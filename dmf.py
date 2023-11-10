import sys
import threading
import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import BSpline,interp1d
from scipy.spatial.transform import Rotation
import cyipopt

#import ase.parallel
from ase.utils import lazyproperty


class GeometricAction(ABC):
    def __init__(
            self,ref_images,coefs=None,
            nsegs=4,dspl=3,
            t_eval=None,w_eval=None,
            n_vel=None,n_trans=None,n_rot=None,
            eps_vel=0.01,eps_rot=0.01,
            trans_rot=True,
            parallel=False):

        #Atoms
        self.ref_images = ref_images
        self.natoms = len(ref_images[0])
        self.masses = ref_images[0].get_masses()
        self.mass_fracs = self.masses/np.sum(self.masses)
        self.atoms_react = ref_images[0].copy()
        self.atoms_prod = ref_images[-1].copy()

        #Constraints
        self.trans_rot = trans_rot
        self.eps_vel = eps_vel
        self.eps_rot = eps_rot

        #Prallel calculation
        self.parallel = parallel

        #B-spline basis functions
        self.nsegs = nsegs
        self.dspl = dspl
        self.nbasis = nsegs + dspl
        t_knot = np.concatenate([
            np.zeros(dspl),
            np.linspace(0.0,1.0,nsegs+1),
            np.ones(dspl)])
        self.t_knot = t_knot
        basis = [
            BSpline(t_knot, np.identity(self.nbasis)[i], dspl)
            for i in range(self.nbasis)]
        d1basis = [b.derivative(nu=1) for b in basis]
        d2basis = [b.derivative(nu=2) for b in basis]
        self.basis = [basis,d1basis,d2basis]

        #t-sequences
        if t_eval is None:
            self.set_t_eval(np.linspace(0.0,1.0,2*nsegs+1))
        else:
            self.set_t_eval(t_eval)

        self.set_w_eval(w_eval)

        if n_vel is None:
            self.n_vel = 4*nsegs
        else:
            self.n_vel = n_vel
        self.t_vel = np.linspace(0.0,1.0,self.n_vel+1)

        if n_trans is None:
            self.n_trans = 2*nsegs
        else:
            self.n_trans = n_trans
        self.t_trans = np.linspace(0.0,1.0,self.n_trans+1)[1:-1]

        if n_rot is None:
            self.n_rot = 2*nsegs
        else:
            self.n_rot = n_rot
        self.t_rot = np.linspace(0.0,1.0,self.n_rot+1)

        #Basis values: [derivative order, basis, t]
        self.P_eval = self.get_basis_values(self.t_eval)
        self.P_vel = self.get_basis_values(self.t_vel)
        self.P_trans = self.get_basis_values(self.t_trans)
        self.P_rot = self.get_basis_values(self.t_rot)

        #Coefficients: [basis, atoms, xyz]
        self.coefs = np.empty([self.nbasis, self.natoms, 3])
        self.angs = np.zeros(3)
        if coefs is not None:
            self.coefs = coefs
        else:
            self.coefs = self.get_coefs_from_ref_images()
        self.coefs0 = self.coefs.copy()
        self.atoms_react.set_positions(self.coefs[0])
        self.atoms_prod.set_positions(self.coefs[-1])

        #Initialize images
        self.images=[]
        for _ in range(self.t_eval.size):
            self.images.append(self.atoms_react.copy())
        self.set_positions()

        #Jacobian of the translation constraints
        self.jac_trans = np.einsum(
            'a,bi,st->isbat',self.mass_fracs,
            self.P_trans[0],np.identity(3))

        self.forces = None
        self.energies = None

    def get_basis_values(self,t_seq):
        return np.array([[[
            b(t) for t in t_seq]
            for b in self.basis[nu]]
            for nu in range(3)])

    def set_t_eval(self,t_eval):
        self.t_eval = t_eval
        self.P_eval = self.get_basis_values(self.t_eval)

    def set_w_eval(self,w_eval=None):
        if w_eval is not None:
            self.w_eval = w_eval
        else:
            w = np.zeros_like(self.t_eval)
            w[0] = 0.5*(self.t_eval[1]-self.t_eval[0])
            w[-1] = 0.5*(self.t_eval[-1]-self.t_eval[-2])
            w[1:-1] = 0.5*(self.t_eval[2:]-self.t_eval[:-2])
            self.w_eval = w

    def get_coefs_from_ref_images(self):
        #Translate and rotate ref_images
        if self.trans_rot:
            prev_image = None
            for image in self.ref_images:
                image.translate(-image.get_center_of_mass())
                if prev_image is not None:
                    pos = image.get_positions()
                    prev_pos = prev_image.get_positions()
                    r = Rotation.align_vectors(
                        prev_pos,pos,weights=self.masses)[0]
                    image.set_positions(r.apply(pos))
                prev_image = image

        nimages = len(self.ref_images)
        pos_ref = np.empty([nimages, self.natoms, 3])
        t_ref = np.zeros(nimages)
        for i,image in enumerate(self.ref_images):
            pos_ref[i] = image.get_positions()
        diff = pos_ref[1:] - pos_ref[:-1]
        l = np.sqrt(
            (self.masses[None,:,None]*diff**2).sum(axis=(1,2)))
        t_ref[1:] = np.cumsum(l)/np.sum(l)

        f = interp1d(t_ref,pos_ref,axis=0)
        t_ref_interp = np.linspace(0.0,1.0,4*self.nsegs+1)[1:-1]
        pos_ref_interp = f(t_ref_interp)
        P_ref_interp0 = self.get_basis_values(t_ref_interp)[0]

        #Solving least-square equations
        A = np.matmul(P_ref_interp0[1:-1],P_ref_interp0[1:-1].T)
        x = pos_ref_interp\
            - np.tensordot(P_ref_interp0[0],pos_ref[0],axes=0)\
            - np.tensordot(P_ref_interp0[-1],pos_ref[-1],axes=0)
        y = np.tensordot(P_ref_interp0[1:-1],x,axes=1).reshape(-1,3*self.natoms)

        coefs = np.empty([self.nbasis, self.natoms, 3])
        coefs[0] = pos_ref[0]
        coefs[-1] = pos_ref[-1]
        coefs[1:-1] = np.linalg.solve(A,y).reshape(-1,self.natoms,3)

        return coefs

    def get_positions(self,t=None,P=None,nu=0):
        if t is None:
            t_temp = self.t_eval
        else:
            t_temp = t
        if P is None:
            P_temp = self.get_basis_values(t_temp)
        else:
            P_temp = P
        return np.tensordot(P_temp[nu].T,self.coefs,1)

    def set_coefs_angs(self,coefs=None,angs=None):
        if coefs is not None:
            self.coefs=coefs
        if angs is not None:
            self.angs = angs
        R=self.get_rot_mats()
        self.coefs[-1]=self.coefs0[-1]@R[0]@R[1]@R[2]

    def get_rot_mats(self):
        R=np.zeros([3,3,3])
        for i in range(3):
            j=(i+1)%3
            k=(i+2)%3
            R[i,i,i]= 1.0
            R[i,j,j]= np.cos(self.angs[i])
            R[i,j,k]=-np.sin(self.angs[i])
            R[i,k,j]= np.sin(self.angs[i])
            R[i,k,k]= np.cos(self.angs[i])
        return R

    def set_positions(self,coefs=None,angs=None):
        self.set_coefs_angs(coefs,angs)
        pos = self.get_positions()
        for i in range(self.t_eval.size):
            self.images[i].set_positions(pos[i])

    def get_consts_trans(self):
        pos = self.get_positions(P=self.P_trans)
        return self.mass_fracs@pos

    def get_jac_trans(self):
        return self.jac_trans

    def get_consts_rot(self):
        pos = self.get_positions(P=self.P_rot)
        return self.mass_fracs@np.cross(pos[:-1],pos[1:])

    def get_jac_rot(self):
        pos = self.get_positions(P=self.P_rot)
        y = np.cross(np.identity(3),pos[...,None,:])
        jac_rot = \
            np.einsum(
                'a,bi,iats->isbat',
                self.mass_fracs,
                self.P_rot[0,:,:-1],
                y[1:]) \
            - np.einsum(
                'a,bi,iats->isbat',
                self.mass_fracs,
                self.P_rot[0,:,1:],
                y[:-1])
        return jac_rot

    def get_consts_vel(self):
        pos = self.get_positions(P=self.P_vel)
        diffs = pos[1:]-pos[:-1]
        d2s = (self.masses[None,:,None]*diffs**2).sum(axis=(1,2))
        return d2s/np.average(d2s)

    def get_jac_vel(self):
        pos = self.get_positions(P=self.P_vel)
        diffs = pos[1:]-pos[:-1]
        d2s = (self.masses[None,:,None]*diffs**2).sum(axis=(1,2))
        diff_P = self.P_vel[0,:,1:]-self.P_vel[0,:,:-1]
        jac_d2s = 2.0*np.einsum(
            'a,bi,ias->ibas',
            self.masses,diff_P,diffs)
        ave_d2s = np.average(d2s)
        return jac_d2s/ave_d2s \
            - np.tensordot(d2s,np.average(jac_d2s,axis=0),0)/(ave_d2s)**2

    def get_jac_fin_rot(self):
        R=self.get_rot_mats()

        dR=np.zeros([3,3,3])
        for i in range(3):
            j=(i+1)%3
            k=(i+2)%3
            dR[i,j,j]=-np.sin(self.angs[i])
            dR[i,j,k]=-np.cos(self.angs[i])
            dR[i,k,j]= np.cos(self.angs[i])
            dR[i,k,k]=-np.sin(self.angs[i])

        jac_rot = np.empty([self.natoms,3,3])
        jac_rot[...,0] = self.coefs0[-1]@dR[0]@R[1]@R[2]
        jac_rot[...,1] = self.coefs0[-1]@R[0]@dR[1]@R[2]
        jac_rot[...,2] = self.coefs0[-1]@R[0]@R[1]@dR[2]

        return jac_rot

    def reshape_jacs(self,jacs):

        def remove_axis(jac):
            if len(jac)==1:
                return jac[0]
            else:
                return jac

        #All constraints are aligned in the 0th axis
        aligned_jac = np.vstack([
            jac.reshape([-1,self.nbasis,self.natoms,3])
            for jac in jacs])
        nc = len(aligned_jac)

        jac_coefs = aligned_jac[:,1:-1,:,:].reshape([nc,-1])

        if self.trans_rot:
            jac_fin_rot = self.get_jac_fin_rot()
            jac_rot = np.tensordot(aligned_jac[:,-1,:,:],jac_fin_rot)
            return remove_axis(np.hstack([jac_coefs,jac_rot]))
        else:
            return remove_axis(jac_coefs)

    def reshape_consts(self,consts):
        return np.hstack([np.ravel(c) for c in consts])

    @lazyproperty
    def f_ends(self):
        forces = np.empty((2, self.natoms, 3))
        if not self.parallel:
            forces[0]=self.atoms_react.get_forces()
            forces[1]=self.atoms_prod.get_forces()
        else:
            def run(image, forces):
                forces[:] = image.get_forces()
            images=[self.atoms_react,self.atoms_prod]
            threads = [threading.Thread(target=run,
                                        args=(images[i],
                                              forces[i:i+1]))
                       for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        return forces

    @lazyproperty
    def e_ends(self):
        f=self.f_ends
        e_react=self.atoms_react.get_potential_energy()
        e_prod =self.atoms_prod.get_potential_energy()
        return np.array([e_react,e_prod])


    @lazyproperty
    def e0(self):
        return np.amin(self.e_ends)

    def get_forces(self):
        eps_t=0.001
        forces = np.empty([self.t_eval.size, self.natoms, 3])
        energies = np.empty(self.t_eval.size)

        inds=[]
        for i in range(self.t_eval.size):
            if self.t_eval[i]<eps_t:
                forces[i] = self.f_ends[0]
                energies[i] = self.e_ends[0]
            elif self.t_eval[i]>1.0-eps_t:
                R=self.get_rot_mats()
                f = self.f_ends[1]
                forces[i] = f@R[0]@R[1]@R[2]
                energies[i] = self.e_ends[1]
            else:
                inds.append(i)

        if not self.parallel:
            for i in inds:
                forces[i] = self.images[i].get_forces()
                energies[i] = self.images[i].get_potential_energy()
        else:
            def run(image, energies, forces):
                forces[:] = image.get_forces()
                energies[:] = image.get_potential_energy()

            threads = [threading.Thread(target=run,
                                        args=(self.images[i],
                                              energies[i:i+1],
                                              forces[i:i+1]))
                       for i in inds]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.energies = energies
        self.forces = forces

        return forces

    @abstractmethod
    def get_objective(self):
        pass

    @abstractmethod
    def get_grad_objective(self):
        pass

    @abstractmethod
    def get_func_en(self,en):
        pass

    def get_norm_vels(self,nu=0):
        pos = self.get_positions(P=self.P_vel)
        diffs = pos[1:]-pos[:-1]

        norm_dx = np.sqrt(
            np.sum(self.masses[None,:,None]*diffs**2,axis=(1,2)))
        dt = self.t_vel[1:]-self.t_vel[:-1]

        t_fd_vel = np.zeros(self.t_vel.size+1)
        t_fd_vel[1:-1] = 0.5*(self.t_vel[1:]+self.t_vel[:-1])
        t_fd_vel[-1] = 1.0

        if nu==0:
            fd_vels = np.zeros(self.t_vel.size + 1)
            fd_vels[1:-1] = norm_dx/dt
            fd_vels[0] = fd_vels[1]
            fd_vels[-1] = fd_vels[-2]

            f = interp1d(t_fd_vel,fd_vels)
            return f(self.t_eval)
        else:
            diff_P_vel0 = self.P_vel[0,:,1:]-self.P_vel[0,:,:-1]
            grad_norm_vel = np.einsum(
                'i,bi,a,ias->ibas',
                1.0/(dt*norm_dx),
                diff_P_vel0,
                self.masses,
                diffs)
            grad_fd_vels = np.zeros(
                [self.t_vel.size+1,self.nbasis,self.natoms,3])
            grad_fd_vels[1:-1] = grad_norm_vel
            grad_fd_vels[0] = grad_norm_vel[0]
            grad_fd_vels[-1] = grad_norm_vel[-1]

            f = interp1d(t_fd_vel,grad_fd_vels,axis=0)
            return f(self.t_eval)

    def get_action(self):

        self.set_positions()
        self.get_forces()

        norm_vels = self.get_norm_vels()
        fe,dfe = self.get_func_en(self.energies)
        action = np.sum(self.w_eval*norm_vels*fe)

        return action

    def get_grad_action(self):

        self.set_positions()
        self.get_forces()

        fe,dfe = self.get_func_en(self.energies)
        norm_vels = self.get_norm_vels()
        grad_norm_vels = self.get_norm_vels(nu=1)

        grad_action = np.tensordot(self.w_eval*fe,grad_norm_vels,1) \
            - np.tensordot(
                self.P_eval[0]*self.w_eval*norm_vels*dfe,
                self.forces,1)

        return grad_action

    def interpolate_energies(
        self,t_eval=None,energies=None,forces=None,coefs=None,
        delta_e=None):

        if t_eval is None:
            t_eval = self.t_eval
        i_fin = np.where(t_eval>0.99)[0][0]
        t_eval = t_eval[:i_fin+1]

        if energies is None:
            energies = self.energies
        energies = energies[:i_fin+1]

        if forces is None:
            forces = self.forces
        forces = forces[:i_fin+1]

        if coefs is None:
            coefs = self.coefs

        P_eval1 = self.get_basis_values(t_eval)[1]
        d_energies = -np.einsum(
            'bi,bas,ias->i',
            P_eval1, coefs, forces)

        t_pows = np.zeros([2*len(t_eval),4])
        for i in range(4):
            t_pows[::2,i] = t_eval**i
            if i<3:
                t_pows[1::2,i+1] = (i+1)*t_eval**i

        ens_dens = np.zeros(2*len(t_eval))
        ens_dens[::2] = energies
        ens_dens[1::2] = d_energies

        polys = np.zeros([len(t_eval)-1,4])
        for i in range(len(t_eval)-1):
            polys[i] = np.linalg.solve(
                t_pows[2*i:2*i+4],ens_dens[2*i:2*i+4])

        if d_energies[np.argmax(energies)]>0.0:
            imax = np.argmax(energies)
        else:
            imax = np.argmax(energies)-1
        tmax = -( polys[imax,2] + np.sqrt(polys[imax,2]**2 \
            -3.0*polys[imax,1]*polys[imax,3])) \
            /(3.0*polys[imax,3])

        tmax_pow = np.array([tmax**i for i in range(4)])
        emax=np.sum(tmax_pow*polys[imax])

        if delta_e is not None:
            e2t = []
            for de in delta_e:
                tlist = np.array([])
                for i in range(len(t_eval)-1):
                    p = P.Polynomial(polys[i])
                    p -= emax-de
                    roots = p.roots()
                    roots = roots.real[abs(roots.imag)<1e-5]
                    roots = roots[(roots>=t_eval[i])&(roots<t_eval[i+1])]
                    tlist = np.append(tlist,roots)
                e2t.append(tlist)
            return polys,tmax,emax,e2t

        return polys,tmax,emax

    def create_problem(self,options={}):
        self.problem = GeoActProblem(self)
        defaults ={
            'tol': 1.0,
            'dual_inf_tol': 0.04,
            'constr_viol_tol': 0.01,
            'compl_inf_tol': 0.01,
            'nlp_scaling_method':'user-scaling',
            'obj_scaling_factor':0.1,
            'limited_memory_initialization':'constant',
            'limited_memory_init_val':2.5,
            'accept_every_trial_step':'yes',
            'output_file':'dmf.out',
            }

        o={**defaults,**options}
        self.problem.add_options(o)
        self.problem.options = o

    def solve_problem(self):
        x0 = self.problem.get_x()
        x,info = self.problem.solve(x0)
        self.problem.set_x(x)

class DirectMaxFlux(GeometricAction):

    def __init__(
        self,
        ref_images,
        beta = 10.0,
        max_alpha = 0.1,
        de  = 0.15,
        ade0 = 0.2,
        di0 = 1.0,
        mu0 = 5.0,
        di1 = 0.2,
        mu1 = 5.0,
        g1  = 0.98,
        adaptive_method=None,
        nmove = 5,
        **kwargs):


        self.beta = beta
        self.max_alpha = max_alpha
        self.de  = de
        self.ade0 = ade0
        self.di0 = di0
        self.mu0 = mu0
        self.di1 = di1
        self.mu1 = mu1
        self.g1  = g1

        self.adaptive_method=adaptive_method
        self.nmove = nmove
        if self.adaptive_method=='full_tmax':
            super().__init__(
                ref_images,
                t_eval=np.linspace(0.0,1.0,nmove+2),
                **kwargs)
        else:
            super().__init__(ref_images,**kwargs)

    def get_forces(self):
        super().get_forces()
        self.energies -= self.e0
        return self.forces

    def get_objective(self):
        return np.log(self.get_action())/self.beta

    def get_grad_objective(self):
        return self.get_grad_action()/self.get_action()/self.beta

    def get_func_en(self,en):
        return np.exp(self.beta*en),self.beta*np.exp(self.beta*en)

class DirectElasticBand(GeometricAction):

    def __init__( self, ref_images, **kwargs):

        super().__init__(ref_images,**kwargs)

        d = self.images[0].get_positions() - self.images[-1].get_positions()
        self.length = np.sqrt(np.sum(d**2 * self.masses[:,None]))

    def get_forces(self):
        super().get_forces()
        self.energies -= self.e0
        return self.forces

    def get_objective(self):
        return self.get_action()/self.length

    def get_grad_objective(self):
        return self.get_grad_action()/self.length

    def get_func_en(self,en):
        return np.where(en>0.0,en,0.0), np.where(en>0.0,1.0,0.0)


class GeoActProblem(cyipopt.Problem):

    def __init__(self,dga):

        if hasattr(dga,'beta'):
            dga.is_dmf = True
        else:
            dga.is_dmf = False
        self.dga = dga

        nvar = (dga.nbasis-2)*3*dga.natoms
        self.nc = nvar
        if dga.trans_rot:
            nvar += 3

        self.var_scales = 1.0

        m_vel = dga.t_vel.size-1
        cl = np.full(m_vel,1.0-dga.eps_vel)
        cu = np.full(m_vel,1.0+dga.eps_vel)

        if dga.trans_rot:
            cl_trans=np.zeros(3*dga.t_trans.size)
            cu_trans=np.zeros(3*dga.t_trans.size)
            m_rot = 3*(dga.t_rot.size-1)
            cl_rot=np.full(m_rot,-dga.eps_rot)
            cu_rot=np.full(m_rot, dga.eps_rot)

            cl = np.hstack([cl,cl_trans,cl_rot])
            cu = np.hstack([cu,cu_trans,cu_rot])

        lb = np.full(nvar,-2.0e19)
        ub = np.full(nvar, 2.0e19)

        super(GeoActProblem, self).__init__(
            n=nvar, m=len(cl),
            lb=lb, ub=ub,
            cl=cl, cu=cu,)

        self.history = self.History()

    class History():
        def __init__(self):
            self.forces=[]
            self.energies=[]
            self.coefs=[]
            self.angs=[]
            self.teval=[]
            self.tmax=[]
            self.images_tmax=[]
            self.beta=[]
            self.weval=[]
            self.teval0=[]
            self.weval0=[]
            self.duals=[]

    def add_options(self,dict_options):
        for item in dict_options.items():
            self.add_option(*item)

    def add_var_scales(self,var_scales):
        self.var_scales = var_scales

    def get_x(self):
        x = self.dga.coefs[1:-1].flatten()
        if self.dga.trans_rot:
            x = np.hstack([x,self.dga.angs])
        return x/self.var_scales

    def set_x(self,x):
        coefs = self.dga.coefs0.copy()
        y = x*self.var_scales
        coefs[1:-1] = y[:self.nc].reshape((-1,self.dga.natoms,3))
        angs = np.zeros(3)
        if self.dga.trans_rot:
            angs = y[self.nc:self.nc+3]

        self.dga.set_positions(coefs,angs)

    def objective(self, x):
        self.set_x(x)
        return self.dga.get_objective()

    def gradient(self,x):
        self.set_x(x)
        grad = self.dga.reshape_jacs(
            [self.dga.get_grad_objective()])
        return grad*self.var_scales

    def constraints(self,x):
        self.set_x(x)
        c_list = [self.dga.get_consts_vel()]
        if self.dga.trans_rot:
            c_list.append(self.dga.get_consts_trans())
            c_list.append(self.dga.get_consts_rot())
        return self.dga.reshape_consts(c_list)

    def jacobian(self,x):
        self.set_x(x)
        j_list = [self.dga.get_jac_vel()]
        if self.dga.trans_rot:
            j_list.append(self.dga.get_jac_trans())
            j_list.append(self.dga.get_jac_rot())
        return self.dga.reshape_jacs(j_list)*self.var_scales

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):

        self.history.forces.append(self.dga.forces)
        self.history.energies.append(self.dga.energies)
        self.history.coefs.append(self.dga.coefs)
        self.history.angs.append(self.dga.angs)
        self.history.teval.append(self.dga.t_eval)
        self.history.weval.append(self.dga.w_eval)
        if self.dga.is_dmf:
            self.history.beta.append(self.dga.beta)
        self.history.duals.append(inf_du)

        polys,tmax,emax_interp = self.dga.interpolate_energies()

        P_tmax = np.array(
            [b(tmax) for b in self.dga.basis[0]])
        image_tmax = self.dga.atoms_react.copy()
        image_tmax.set_positions(
            np.tensordot(P_tmax,self.dga.coefs,1))
        self.history.tmax.append(tmax)
        self.history.images_tmax.append(image_tmax)

        if iter_count==0:
            for image in self.dga.images:
                if hasattr(image,'calc1'):
                    image.calc = image.calc1

        if self.dga.is_dmf:
            if self.dga.adaptive_method=='full_tmax':
                un_di = inf_du \
                    /self.options['obj_scaling_factor'] \
                    /np.amax(self.var_scales)
                tol_di = self.options['dual_inf_tol'] \
                    /np.amax(self.var_scales)

                de  = self.dga.de
                ade0 = self.dga.ade0
                di0 = self.dga.di0
                mu0 = self.dga.mu0
                di1 = self.dga.di1
                mu1 = self.dga.mu1
                g1  = self.dga.g1
                c0  = 0.5         *np.tanh(-2.0*mu0*(un_di-di0)) + 0.5
                c1  = 0.5*(1.0-g1)*np.tanh( 2.0*mu1*(un_di-di1)) + 0.5*(1.0+g1)

                nmove = self.dga.nmove
                barrier = emax_interp - np.amax(self.dga.e_ends)\
                    +self.dga.e0
                de = min(2.0/float(nmove+1)*barrier,de)
                delta_e = de*np.arange(0.5*(nmove%2+1.0),0.5*(nmove+1.0),1.0)
                e2t = self.dga.interpolate_energies(delta_e=delta_e)[3]
                t_cand_m = np.hstack([tl[tl<tmax] for tl in e2t])
                t_cand_p = np.hstack([tl[tl>tmax] for tl in e2t])
                temp_t_eval_m = t_cand_m[
                    np.argsort(np.abs(t_cand_m-tmax))[:nmove//2]]
                temp_t_eval_p = t_cand_p[
                    np.argsort(np.abs(t_cand_p-tmax))[:nmove//2]]
                if nmove%2==1:
                    temp_t_eval_p = np.append(temp_t_eval_p,tmax)
                temp_t_eval = np.sort(np.append(temp_t_eval_m,temp_t_eval_p))

                alpha = c0*self.dga.max_alpha
                t_eval = self.dga.t_eval.copy()
                t_eval[1:-1] = (1.0-alpha)*t_eval[1:-1] + alpha*temp_t_eval

                self.dga.set_t_eval(t_eval)
                self.dga.set_w_eval()

                self.dga.max_alpha *= c1

