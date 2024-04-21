import sys
import threading
import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import BSpline,interp1d
from scipy.spatial.transform import Rotation
import cyipopt

import ase.parallel
from ase.utils import lazyproperty
from ase.neb import IDPP
#from sidpp import SIDPP
from ase.data import covalent_radii


class GeometricAction(ABC,cyipopt.Problem):
    """Class for general variational reaction path optimization.

    This class defines vaiational problems with the objective functional
    I[x(t)] = \int^{1}_{0} dt |v(t)| F(x(t)).

    Attributes:
        images (list of ase.Atoms):
        t_eval (numpy.ndarray):
        w_eval (numpy.ndarray):
        coefs (numpy.ndarray):
        angs (numpy.ndarray):
        energies (numpy.ndarray):
        forces (numpy.ndarray):
        history (History object):
        remove_rotation_and_translation (bool):
        natoms (int):
        nsegs (int):
        dspl (int):
        nbasis (int):
        n_vel (int):
        n_trans (int):
        n_rot (int):
        eps_vel (float):
        eps_rot (float):
        ipopt_options (dict):
    """
    def __init__(self,
        ref_images,
        coefs=None, nsegs=4,dspl=3,
        remove_rotation_and_translation=True,
        mass_weighted=False,
        check_collisions=False,
        parallel=False,world=None,
        t_eval=None,w_eval=None,
        n_vel=None,n_trans=None,n_rot=None,
        eps_vel=0.01,eps_rot=0.01,
        ):
        """__init__ method of GeometricAction.

        Args:
            ref_images (list of ase.Atoms):
            nsegs (int):
            dspl (int):
            remove_rotation_and_translation (bool):
            mass_weighted (bool):
            parallel (bool):
            world (MPI world object):
            t_eval (numpy.ndarray):
            w_eval (numpy.ndarray):
            n_vel (int):
            n_trans (int):
            n_rot (int):
            eps_vel (float):
            eps_rot (float):
        """

        #Atoms
        self.natoms = len(ref_images[0])
        if mass_weighted:
            self._masses = ref_images[0].get_masses()
        else:
            self._masses = np.ones(self.natoms)
        self._mass_fracs = self._masses/np.sum(self._masses)

        self.check_collisions = check_collisions

        #Constraints
        self.remove_rotation_and_translation \
            = remove_rotation_and_translation
        self.eps_vel = eps_vel
        self.eps_rot = eps_rot

        #Prallel calculation
        self.parallel = parallel

        if world is None:
            world = ase.parallel.world
        self._world = world

        #B-spline basis functions
        self.nsegs = nsegs
        self.dspl = dspl
        self.nbasis = nsegs + dspl
        _t_knot = np.concatenate([
            np.zeros(dspl),
            np.linspace(0.0,1.0,nsegs+1),
            np.ones(dspl)])
        self._t_knot = _t_knot
        basis = [
            BSpline(_t_knot, np.identity(self.nbasis)[i], dspl)
            for i in range(self.nbasis)]
        d1basis = [b.derivative(nu=1) for b in basis]
        d2basis = [b.derivative(nu=2) for b in basis]
        self._basis = [basis,d1basis,d2basis]

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
        self._P_eval = self._get_basis_values(self.t_eval)
        self._P_vel = self._get_basis_values(self.t_vel)
        self._P_trans = self._get_basis_values(self.t_trans)
        self._P_rot = self._get_basis_values(self.t_rot)

        #Coefficients: [basis, atoms, xyz]
        self.coefs = np.empty([self.nbasis, self.natoms, 3])
        self.angs = np.zeros(3)
        if coefs is not None:
            self.coefs = coefs
        else:
            self.coefs = self._get_coefs_from_ref_images(ref_images)
        self._coefs0 = self.coefs.copy()

        #Initialize images
        self.images=[]
        for _ in range(self.t_eval.size):
            self.images.append(ref_images[0].copy())
        self.set_positions()

        #Jacobian of the translation constraints
        self._jac_trans = np.einsum(
            'a,bi,st->isbat',self._mass_fracs,
            self._P_trans[0],np.identity(3))

        self.forces = None
        self.energies = None

        self.history = self.History()

        #initialize cyipopt.Problem
        nvar = (self.nbasis-2)*3*self.natoms
        if self.remove_rotation_and_translation:
            nvar += 3

        self.var_scales = 1.0

        m_vel = self.t_vel.size-1
        cl = np.full(m_vel,1.0-self.eps_vel)
        cu = np.full(m_vel,1.0+self.eps_vel)

        if self.remove_rotation_and_translation:
            cl_trans=np.zeros(3*self.t_trans.size)
            cu_trans=np.zeros(3*self.t_trans.size)
            m_rot = 3*(self.t_rot.size-1)
            cl_rot=np.full(m_rot,-self.eps_rot)
            cu_rot=np.full(m_rot, self.eps_rot)

            cl = np.hstack([cl,cl_trans,cl_rot])
            cu = np.hstack([cu,cu_trans,cu_rot])

        lb = np.full(nvar,-2.0e19)
        ub = np.full(nvar, 2.0e19)

        cyipopt.Problem.__init__(self,
            n=nvar, m=len(cl),
            lb=lb, ub=ub,
            cl=cl, cu=cu,)

        #set ipopt options
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

        if self.parallel and self._world.size>1:
            if self._world.rank > 0:
                defaults['print_level'] = 0

        self.ipopt_options = dict()
        self.add_ipopt_options(defaults)

    class History():
        """Object storing histories of some properties.

        Attributes:
            forces (list of numpy.ndarray):
            energies (list of numpy.ndarray):
            coefs (list of numpy.ndarray):
            angs (list of numpy.ndarray):
            tmax (list of float):
            images_tmax (list of ase.Atoms):
            duals (list of float):
        """
        def __init__(self):
            self.forces=[]
            self.energies=[]
            self.coefs=[]
            self.angs=[]
            self.tmax=[]
            self.images_tmax=[]
            self.duals=[]

    def _get_basis_values(self,t_seq):
        return np.array([[[
            b(t) for t in t_seq]
            for b in self._basis[nu]]
            for nu in range(3)])

    def set_t_eval(self,t_eval):
        """Setter of t_eval.
        """
        self.t_eval = t_eval
        self._P_eval = self._get_basis_values(self.t_eval)

    def set_w_eval(self,w_eval=None):
        """Setter of w_eval.
        """
        if w_eval is not None:
            self.w_eval = w_eval
        else:
            w = np.zeros_like(self.t_eval)
            w[0] = 0.5*(self.t_eval[1]-self.t_eval[0])
            w[-1] = 0.5*(self.t_eval[-1]-self.t_eval[-2])
            w[1:-1] = 0.5*(self.t_eval[2:]-self.t_eval[:-2])
            self.w_eval = w

    def _collide(self,image):
        r = covalent_radii[image.arrays['numbers']]
        r_cov = r[None,:]+r[:,None]
        np.fill_diagonal(r_cov,0.0)
        d = image.get_all_distances()
        np.fill_diagonal(d,1.0e5)

        return np.any(d<0.5*r_cov)
        #return np.any(d<0.4)

    def _ignore_collisions(self):
        r = covalent_radii[self.images[0].arrays['numbers']]
        r_cov = r[None,:]+r[:,None]
        np.fill_diagonal(r_cov,0.0)
        for i,image in enumerate(self.images):
            d = image.get_all_distances()
            #if np.any(d<0.25*r_cov):
            if np.any(d<0.5):
                self.w_eval[i] = 0.0
        print(self.w_eval)
        #izeros = np.where(self.w_eval<0.01)[0]
        #if len(izeros)>0:
        #    self.w_eval[izeros[0]:izeros[-1]+1] = 0.0

    def _get_coefs_from_ref_images(self,ref_images):
        ref_images_copy = [image.copy() for image in ref_images]
        #Translate and rotate ref_images
        if self.remove_rotation_and_translation:
            prev_image = None
            for image in ref_images_copy:
                pos = image.get_positions()
                image.translate(-self._mass_fracs@pos)
                if prev_image is not None:
                    pos = image.get_positions()
                    prev_pos = prev_image.get_positions()
                    r = Rotation.align_vectors(
                        prev_pos,pos,weights=self._masses)[0]
                    image.set_positions(r.apply(pos))
                prev_image = image

        nimages = len(ref_images_copy)
        pos_ref = np.empty([nimages, self.natoms, 3])
        t_ref = np.zeros(nimages)
        for i,image in enumerate(ref_images_copy):
            pos_ref[i] = image.get_positions()
        diff = pos_ref[1:] - pos_ref[:-1]
        l = np.sqrt(
            (self._masses[None,:,None]*diff**2).sum(axis=(1,2)))
        t_ref[1:] = np.cumsum(l)/np.sum(l)

        f = interp1d(t_ref,pos_ref,axis=0)
        t_ref_interp = np.linspace(0.0,1.0,4*self.nsegs+1)[1:-1]
        pos_ref_interp = f(t_ref_interp)
        P_ref_interp0 = self._get_basis_values(t_ref_interp)[0]

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
        """Get all positions of images at t or their derivatives.

        Args:
            t (1D numpy.ndarray, optional): All positions of images at t or their derivatives are returned. If not present, t_eval is used. Default is None.
            P (numpy.ndarray, optional): Return of _get_basis_values(t). See source code. Default is None. If both t and P are present, P is used in priority.
            nu (int, optional): Degree of derivative with respect to t. nu must be 0, 1, or 2. Default is 0.

        Returns:
            numpy.ndarray shape (nimages,natoms,3): Positions or their nu-th derivative at t.

        """

        if t is None:
            t_temp = self.t_eval
        else:
            t_temp = t
        if P is None:
            P_temp = self._get_basis_values(t_temp)
        else:
            P_temp = P
        return np.tensordot(P_temp[nu].T,self.coefs,1)

    def set_coefs_angs(self,coefs=None,angs=None):
        """Setter of coefs and angs.
        """
        if coefs is not None:
            self.coefs=coefs
        if angs is not None:
            self.angs = angs
        R=self._get_rot_mats()
        self.coefs[-1]=self._coefs0[-1]@R[0]@R[1]@R[2]

    def _get_rot_mats(self):
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
        """Set positions of all images in self.images. These positions are determined by coefs and angs.

        Args:
            coefs (numpy.ndarray shape (nbasis,natoms,3), optional): If not present, self.coefs is used. Default is None.
            angs (numpy.ndarray shape (3), optional): If not present, self.angs is used. Default is None.
        """
        self.set_coefs_angs(coefs,angs)
        pos = self.get_positions()
        for i in range(self.t_eval.size):
            self.images[i].set_positions(pos[i])

    def _get_consts_trans(self):
        pos = self.get_positions(P=self._P_trans)
        return self._mass_fracs@pos

    def _get_jac_trans(self):
        return self._jac_trans

    def _get_consts_rot(self):
        pos = self.get_positions(P=self._P_rot)
        return self._mass_fracs@np.cross(pos[:-1],pos[1:])

    def _get_jac_rot(self):
        pos = self.get_positions(P=self._P_rot)
        y = np.cross(np.identity(3),pos[...,None,:])
        jac_rot = \
            np.einsum(
                'a,bi,iats->isbat',
                self._mass_fracs,
                self._P_rot[0,:,:-1],
                y[1:]) \
            - np.einsum(
                'a,bi,iats->isbat',
                self._mass_fracs,
                self._P_rot[0,:,1:],
                y[:-1])
        return jac_rot

    def _get_consts_vel(self):
        pos = self.get_positions(P=self._P_vel)
        diffs = pos[1:]-pos[:-1]
        d2s = (self._masses[None,:,None]*diffs**2).sum(axis=(1,2))
        return d2s/np.average(d2s)

    def _get_jac_vel(self):
        pos = self.get_positions(P=self._P_vel)
        diffs = pos[1:]-pos[:-1]
        d2s = (self._masses[None,:,None]*diffs**2).sum(axis=(1,2))
        diff_P = self._P_vel[0,:,1:]-self._P_vel[0,:,:-1]
        jac_d2s = 2.0*np.einsum(
            'a,bi,ias->ibas',
            self._masses,diff_P,diffs)
        ave_d2s = np.average(d2s)
        return jac_d2s/ave_d2s \
            - np.tensordot(d2s,np.average(jac_d2s,axis=0),0)/(ave_d2s)**2

    def _get_jac_fin_rot(self):
        R=self._get_rot_mats()

        dR=np.zeros([3,3,3])
        for i in range(3):
            j=(i+1)%3
            k=(i+2)%3
            dR[i,j,j]=-np.sin(self.angs[i])
            dR[i,j,k]=-np.cos(self.angs[i])
            dR[i,k,j]= np.cos(self.angs[i])
            dR[i,k,k]=-np.sin(self.angs[i])

        jac_rot = np.empty([self.natoms,3,3])
        jac_rot[...,0] = self._coefs0[-1]@dR[0]@R[1]@R[2]
        jac_rot[...,1] = self._coefs0[-1]@R[0]@dR[1]@R[2]
        jac_rot[...,2] = self._coefs0[-1]@R[0]@R[1]@dR[2]

        return jac_rot

    def _reshape_jacs(self,jacs):

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

        if self.remove_rotation_and_translation:
            jac_fin_rot = self._get_jac_fin_rot()
            jac_rot = np.tensordot(aligned_jac[:,-1,:,:],jac_fin_rot)
            return remove_axis(np.hstack([jac_coefs,jac_rot]))
        else:
            return remove_axis(jac_coefs)

    def _reshape_consts(self,consts):
        return np.hstack([np.ravel(c) for c in consts])

    @lazyproperty
    def _f_ends(self):
        forces = np.empty((2, self.natoms, 3))
        if not self.parallel:
            forces[0]=self.images[0].get_forces()
            forces[1]=self.images[-1].get_forces()
        elif self._world.size==1:
            def run(image, forces):
                forces[:] = image.get_forces()
            images=[self.images[0],self.images[-1]]
            threads = [threading.Thread(target=run,
                                        args=(images[i],
                                              forces[i:i+1]))
                       for i in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            nmv = len(self.images)-2
            i = self._world.rank * nmv // self._world.size
            try:
                if i==0:
                    forces[0] = self.images[0].get_forces()
                elif i==1:
                    forces[-1] = self.images[-1].get_forces()
            except Exception:
                error = self._world.sum(1.0)
                raise
            else:
                error = self._world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel DMF failed!')

            root0 = 0
            root1 = self._world.size // nmv
            self._world.broadcast(forces[0], root0)
            self._world.broadcast(forces[-1], root1)

        return forces

    @lazyproperty
    def _e_ends(self):
        f=self._f_ends
        energies = np.empty(2)
        if (not self.parallel) or self._world.size==1:
            energies[0] = self.images[0].get_potential_energy()
            energies[1] = self.images[-1].get_potential_energy()
        else:
            nmv = len(self.images)-2
            root0 = 0
            root1 = self._world.size // nmv
            if self._world.rank == root0:
                energies[0] = self.images[0].get_potential_energy()
            elif self._world.rank == root1:
                energies[1] = self.images[-1].get_potential_energy()

            self._world.broadcast(energies[0:1], root0)
            self._world.broadcast(energies[1:2], root1)

        return energies


    @lazyproperty
    def e0(self):
        return np.amin(self._e_ends)

    def get_forces(self):
        """Get forces of all images in self.images. After calling this method, forces and energies are stored in self.forces and self.energies, respectively.

        Returns:
            numpy.ndarray shape (nimages,natoms,3): Forces of all images in self.images.
        """
        eps_t=0.01
        eps_w=0.001


        forces = np.empty([self.t_eval.size, self.natoms, 3])
        energies = np.empty(self.t_eval.size)
        e0 = self.e0

        inds=[]
        for i in range(self.t_eval.size):
            if self.t_eval[i]<eps_t:
                forces[i] = self._f_ends[0]
                energies[i] = self._e_ends[0]
            elif self.t_eval[i]>1.0-eps_t:
                R=self._get_rot_mats()
                f = self._f_ends[1]
                forces[i] = f@R[0]@R[1]@R[2]
                energies[i] = self._e_ends[1]
            else:
                inds.append(i)

        if not self.parallel:
            for i in inds:
                if self.check_collisions and self._collide(self.images[i]):
                    forces[i] = 0.0
                    energies[i] = e0-1.0e5
                else:
                    forces[i] = self.images[i].get_forces()
                    energies[i] = self.images[i].get_potential_energy()

        elif self._world.size==1:
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

        else:
            nmv = len(self.images)-2
            i = self._world.rank * nmv // self._world.size + 1
            try:
                forces[i] = self.images[i].get_forces()
                energies[i] = self.images[i].get_potential_energy()
            except Exception:
                error = self._world.sum(1.0)
                raise
            else:
                error = self._world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel DMF failed!')

            for i in range(1,nmv+1):
                root = (i-1) * self._world.size // nmv
                self._world.broadcast(energies[i:i + 1], root)
                self._world.broadcast(forces[i], root)


        self.energies = energies
        self.forces = forces


        return forces

    @abstractmethod
    def _get_objective(self):
        pass

    @abstractmethod
    def _get_grad_objective(self):
        pass

    @abstractmethod
    def _get_func_en(self,en):
        pass

    def _get_norm_vels(self,nu=0):
        pos = self.get_positions(P=self._P_vel)
        diffs = pos[1:]-pos[:-1]

        norm_dx = np.sqrt(
            np.sum(self._masses[None,:,None]*diffs**2,axis=(1,2)))
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
            diff_P_vel0 = self._P_vel[0,:,1:]-self._P_vel[0,:,:-1]
            grad_norm_vel = np.einsum(
                'i,bi,a,ias->ibas',
                1.0/(dt*norm_dx),
                diff_P_vel0,
                self._masses,
                diffs)
            grad_fd_vels = np.zeros(
                [self.t_vel.size+1,self.nbasis,self.natoms,3])
            grad_fd_vels[1:-1] = grad_norm_vel
            grad_fd_vels[0] = grad_norm_vel[0]
            grad_fd_vels[-1] = grad_norm_vel[-1]

            f = interp1d(t_fd_vel,grad_fd_vels,axis=0)
            return f(self.t_eval)

    def _get_action(self):

        self.set_positions()
        self.get_forces()

        norm_vels = self._get_norm_vels()
        fe,dfe = self._get_func_en(self.energies)
        action = np.sum(self.w_eval*norm_vels*fe)

        return action

    def _get_grad_action(self):

        self.set_positions()
        self.get_forces()

        fe,dfe = self._get_func_en(self.energies)
        norm_vels = self._get_norm_vels()
        grad_norm_vels = self._get_norm_vels(nu=1)

        grad_action = np.tensordot(self.w_eval*fe,grad_norm_vels,1) \
            - np.tensordot(
                self._P_eval[0]*self.w_eval*norm_vels*dfe,
                self.forces,1)

        return grad_action

    def interpolate_energies(
        self,t_eval=None,energies=None,forces=None,coefs=None,
        delta_e=None):
        """Interpolate the enegy along the path. See Ref. 1 for details.

        Args:
            t_eval (numpy.ndarray shape (\:), optional): Points where energies and forces were evaluated. If not present, self.t_eval is used. Default is None.
            energies (numpy.ndarray shape (len(t_eval)), optional): Energies evaluated at t_eval. If not present, self.energies is used. Default is None.
            forces (numpy.ndarray shape (len(t_eval),natoms,3), optional): Forces evaluated at t_eval. If not present, self.forces is used. Default is None.
            coefs (numpy.ndarray shape (nbasis,natoms,3), optional): Coefficients of B-spline functions. If not present, self.coefs is used. Default is None.
            delta_e (list of float, optional): If present, return points that satisfy :math:`\\tilde{E}(t) = E_{\\max} - \\text{delta_e}`.

        Returns:
            polys (numpy.ndarray shape (len(t_eval)-1,4)): Coefficients of piecewise cubic polynomial.
            t_max (float): Maximum point of :math:`\\tilde{E}(t)`.
            e_max (float): Maximum value of :math:`\\tilde{E}(t)`.
            t_de (list of float): Points that satisfy :math:`\\tilde{E}(t) = E_{\\max} - \\text{delta_e}`.
        """

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

        P_eval1 = self._get_basis_values(t_eval)[1]
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
        t_max = -( polys[imax,2] + np.sqrt(polys[imax,2]**2 \
            -3.0*polys[imax,1]*polys[imax,3])) \
            /(3.0*polys[imax,3])

        t_max_pow = np.array([t_max**i for i in range(4)])
        e_max=np.sum(t_max_pow*polys[imax])

        if delta_e is not None:
            t_de = []
            for de in delta_e:
                tlist = np.array([])
                for i in range(len(t_eval)-1):
                    p = P.Polynomial(polys[i])
                    p -= e_max-de
                    roots = p.roots()
                    roots = roots.real[abs(roots.imag)<1e-5]
                    roots = roots[(roots>=t_eval[i])&(roots<t_eval[i+1])]
                    tlist = np.append(tlist,roots)
                t_de.append(tlist)
            return polys,t_max,e_max,t_de

        return polys,t_max,e_max

    def solve(self,tol='tight'):
        """Solve the variational problem.

        Args:
            tol (float or str, optional): Change IOPOT option dual_inf_tol. If tol is float, dual_inf_tol is set to tol. If tol is either 'tight', 'middle', or 'loose' (keywords used in Ref. 1), dual_inf_tol is set to 0.04, 0.1, or 0.2, respectively. Default is 'tight'.
        """

        if tol:
            if isinstance(tol,float):
                self.add_ipopt_options({'dual_inf_tol':tol})
            elif isinstance(tol,str):
                if tol.strip().upper()=='TIGHT':
                    self.add_ipopt_options({'dual_inf_tol':0.04})
                elif tol.strip().upper()=='MIDDLE':
                    self.add_ipopt_options({'dual_inf_tol':0.1})
                elif tol.strip().upper()=='LOOSE':
                    self.add_ipopt_options({'dual_inf_tol':0.2})

        x0 = self.get_x()
        x,info = super().solve(x0)
        self.set_x(x)
        return x,info


    def add_ipopt_options(self,dict_options):
        """Method for adding ipopt options.
        """
        self.ipopt_options.update(dict_options)
        for item in self.ipopt_options.items():
            self.add_option(*item)

    def get_x(self):
        """Get variables of the variational problem in a 1D array.

        Returns:
            x (1D numpy.ndarray): self.coefs[1:-1].flatten(). If remove_rotation_and_translation is True, self.angs is appended.
        """
        x = self.coefs[1:-1].flatten()
        if self.remove_rotation_and_translation:
            x = np.hstack([x,self.angs])
        return x/self.var_scales

    def set_x(self,x):
        """Set self.coefs and self.angs from x.

        Args:
            x (1D numpy.ndarray): Variables of the variational problem in a 1D array.
        """
        nc = (self.nbasis-2)*3*self.natoms
        coefs = self._coefs0.copy()
        y = x*self.var_scales
        coefs[1:-1] = y[:nc].reshape((-1,self.natoms,3))
        angs = np.zeros(3)
        if self.remove_rotation_and_translation:
            angs = y[-3:]

        self.set_positions(coefs,angs)

    def objective(self, x):
        """Objective function.

        Args:
            x (1D numpy.ndarray): Variables of the variational problem in a 1D array.

        Returns:
            float: Objective function.
        """
        self.set_x(x)
        return self._get_objective()

    def gradient(self,x):
        """Gradient of the objective function.

        Args:
            x (1D numpy.ndarray): Variables of the variational problem in a 1D array.

        Returns:
            numpy.ndarray shape (len(x)): Gradient of the objective function.
        """
        self.set_x(x)
        grad = self._reshape_jacs(
            [self._get_grad_objective()])
        return grad*self.var_scales

    def constraints(self,x):
        """Constraints.
        Args:
            x (1D numpy.ndarray): Variables of the variational problem in a 1D array.

        Returns:
            numpy.ndarray shape (# of constraints): Constraints.
        """
        self.set_x(x)
        c_list = [self._get_consts_vel()]
        if self.remove_rotation_and_translation:
            c_list.append(self._get_consts_trans())
            c_list.append(self._get_consts_rot())
        return self._reshape_consts(c_list)

    def jacobian(self,x):
        """Jacobian of the constraints.
        Args:
            x (1D numpy.ndarray): Variables of the variational problem in a 1D array.

        Returns:
            numpy.ndarray shape (# of constraints,len(x)): Jacobian of the constraints.
        """
        self.set_x(x)
        j_list = [self._get_jac_vel()]
        if self.remove_rotation_and_translation:
            j_list.append(self._get_jac_trans())
            j_list.append(self._get_jac_rot())
        return self._reshape_jacs(j_list)*self.var_scales

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        """Method called at the end of each iteration.
        """

        self.history.forces.append(self.forces)
        self.history.energies.append(self.energies)
        self.history.coefs.append(self.coefs)
        self.history.angs.append(self.angs)
        self.history.duals.append(inf_du)

        polys,tmax,emax_interp = self.interpolate_energies()

        P_tmax = np.array(
            [b(tmax) for b in self._basis[0]])
        image_tmax = self.images[0].copy()
        image_tmax.set_positions(
            np.tensordot(P_tmax,self.coefs,1))
        self.history.tmax.append(tmax)
        self.history.images_tmax.append(image_tmax)



class DirectMaxFlux(GeometricAction):
    """Class for the direct MaxFlux method.

    This class defines the variational problem of the direct MaxFlux method.

    Args:
        ref_images (list of ase.Atoms):
            Reference images for generating an initial path. If coefs is not present, coefs of the initial path is generated by a linear interpolation with ref_images. If coefs is present, only ref_images[0] and ref_images[-1] are used as the endpoints of the path.
        coefs (numpy.ndarray shape (nbasis,natoms,3),optional):
            Coefficients of basis functions. If coefs is present, the interpolation with ref_images is skipped, and coefs is stored as it is. Default is None.
        nsegs (int,optional):
            Determines the number of B-spline functions. See Ref. 1 for details. Default is 4.
        dspl (int,optional):
            Degree of B-spline functions. Default is 3.
        remove_rotation_and_translation (bool,optional):
            Remove redundancy regarding rotaional and translational symmetry. Default is True.
        mass_weighted (bool,optional):
            Use mass-weighted coordinates. False is recommended for a rapid convergence. Default is False.
        parallel (bool,optional):
            Calculate forces of images in parallel. Both Threading and MPI are supported. Default is False. Usage is basically the same as the NEB method in ASE. See `ASE's document <https://wiki.fysik.dtu.dk/ase/ase/neb.html#parallelization-over-images>`_.
        world (MPI world object,optional):
            If not present, ase.parallel.world is used. Default is None.
        t_eval (numpy.ndarray shape (nmove+2),optional):
            Initial energy evaluation points. If not present or update_teval is True, an equally distributed t_eval is used. Default is None.
        w_eval (numpy.ndarray shape (nmove+2),optional):
            Weights of the numerical integuration of the objective function. If not present, w_eval is determined by the trapezoidal formula. Default is None.
        n_vel (int,optional):
            Determines the number of velocity constraints. See Ref. 1 for details. If not present, n_vel is 4*nsegs. Default is None.
        n_trans (int,optional):
            Determines the number of translational constraints. See Ref. 1 for details. If not present, n_trans is 2*nsegs. Default is None.
        n_rot (int,optional):
            Determines the number of rotational constraints. See Ref. 1 for details. If not present, n_rot is 2*nsegs. Default is None.
        eps_vel (float,optional):
            Parameter for loosening velocity constraints. See Ref. 1 for details. Default is 0.01.
        eps_rot (float,optional):
            Parameter for loosening rotational constraints. See Ref. 1 for details. Default is 0.01.
        beta (float,optional):
            Reciprocal temperature of the direct MaxFlux method. Default is 10 [/eV].
        nmove (int,optional):
            The number of movable energy evaluation points (images). Default is 5.
        update_teval (bool,optional):
            Update t_eval at each iteration. Default is False.
        params_t_update (dict,optional):
            Parameters for t_eval updating. See Ref. 1 for details.

    Attributes:
        images (list of ase.Atoms):
            nmove+2 images at t_eval.
        coefs (numpy.ndarray shape (nbasis,natoms,3)):
            Coefficients of B-spline functions.
        angs (numpy.ndarray shape (3)):
            Euler angles of images[-1], which is used when remove_rotation_and_translation is True.
        energies (numpy.ndarray shape (nmove+2)):
            Energies of all images, which are shifted by e0, are stored after calling get_forces().
        e0 (float):
            min(E(images[0]),E(images[-1]))
        forces (numpy.ndarray shape (nmove+2,natoms,3)):
            Forces of each image are stored after calling get_forces().
        history (History object):
            Object that stores histories of some properties. See History object below.
        natoms (int):
            The number of atoms in the system.
        nbasis (int):
            The number of the B-spline functions. nbasis = nsegs + dspl.
        ipopt_options (dict):
            Non-default options of IPOPT. On initialization, {'tol'\:1.0, 'dual_inf_tol'\:0.04, 'constr_viol_tol'\:0.01, 'compl_inf_tol'\:0.01, 'nlp_scaling_method'\:'user-scaling', 'obj_scaling_factor'\:0.1, 'limited_memory_initialization'\:'constant', 'limited_memory_init_val'\:2.5, 'accept_every_trial_step'\:'yes', 'output_file'\:'dmf.out',} is set.
        remove_rotation_and_translation (bool):
            Same as the above parameter.
        nsegs (int):
            Same as the above parameter.
        dspl (int):
            Same as the above parameter.
        t_eval (numpy.ndarray):
            Same as the above parameter.
        w_eval (numpy.ndarray):
            Same as the above parameter.
        n_vel (int):
            Same as the above parameter.
        n_trans (int):
            Same as the above parameter.
        n_rot (int):
            Same as the above parameter.
        eps_vel (float):
            Same as the above parameter.
        eps_rot (float):
            Same as the above parameter.
        beta (float):
            Same as the above parameter.
        nmove (int):
            Same as the above parameter.
        update_teval (bool):
            Same as the above parameter.
        params_t_update (dict):
            Same as the above parameter.
    """

    def __init__(
        self,
        ref_images,
        coefs=None, nsegs=4,dspl=3,
        remove_rotation_and_translation=True,
        mass_weighted=False,
        check_collisions=False,
        parallel=False,world=None,
        t_eval=None,w_eval=None,
        n_vel=None,n_trans=None,n_rot=None,
        eps_vel=0.01,eps_rot=0.01,
        beta = 10.0,
        nmove = 5,
        update_teval = False,
        params_t_update = {
            'max_alpha0':0.1,'de':0.15,
            'dia':1.0,'mua':5.0,
            'dib':0.2,'mub':5.0,'epsb':0.02,},
        ):
        """__init__ method of DirectMaxFlux.

        Args:
            ref_images (list of ase.Atoms):
            coefs (numpy.ndarray,optional):
            nsegs (int,optional):
            dspl (int,optional):
            remove_rotation_and_translation (bool,optional):
            mass_weighted (bool,optional):
            parallel (bool,optional):
            world (MPI world object,optional):
            t_eval (numpy.ndarray,optional):
            w_eval (numpy.ndarray,optional):
            n_vel (int,optional):
            n_trans (int,optional):
            n_rot (int,optional):
            eps_vel (float,optional):
            eps_rot (float,optional):
            beta (float,optional):
            nmove (int,optional):
            update_teval (bool,optional):
            params_t_update (dict,optional):
        """
        args = locals()
        base_params = [
            'ref_images','coefs','nsegs','dspl',
            'remove_rotation_and_translation','mass_weighted',
            'check_collisions',
            'parallel','world','t_eval','w_eval','n_vel',
            'n_trans','n_rot','eps_vel','eps_rot']
        base_args = {k:args[k] for k in base_params}

        self.beta = beta
        self.params_t_update = params_t_update
        self._max_alpha = params_t_update['max_alpha0']

        self.update_teval = update_teval

        self.nmove = nmove

        base_args.update(t_eval=np.linspace(0.0,1.0,nmove+2))

        super().__init__(**base_args)


    class History():
        """Object storing histories of properties listed below.

        Attributes:
            forces (list of numpy.ndarray shape (nmove+2,natoms,3)):
            energies (list of numpy.ndarray shape (nmove+2)):
            coefs (list of numpy.ndarray shape (nbasis,natoms,3)):
            angs (list of numpy.ndarray shape (3)):
            t_eval (list of numpy.ndarray shape (nmove+2)):
            tmax (list of float):
                History of tmax. tmax is the highest energy point of the interpoated energy along the path. See Ref. 1 for details.
            images_tmax (list of ase.Atoms):
                History of the image at t=tmax, an approximation of TS.
            duals (list of float):
                History of the scaled dual infeasibility.
        """
        def __init__(self):
            self.forces=[]
            self.energies=[]
            self.coefs=[]
            self.angs=[]
            self.tmax=[]
            self.images_tmax=[]
            self.duals=[]
            self.t_eval=[]

    def get_forces(self):
        super().get_forces()
        self.energies -= self.e0
        return self.forces

    def _get_objective(self):
        return np.log(self._get_action())/self.beta

    def _get_grad_objective(self):
        return self._get_grad_action()/self._get_action()/self.beta

    def _get_func_en(self,en):
        return np.exp(self.beta*en),self.beta*np.exp(self.beta*en)

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        """Method called at the end of each iteration. Updating of t_eval is defined in this method.
        """

        super().intermediate(alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials)

        if self.update_teval:
            self.history.t_eval.append(self.t_eval)

            polys,tmax,emax_interp = self.interpolate_energies()

            un_di = inf_du \
                /self.ipopt_options['obj_scaling_factor'] \
                /np.amax(self.var_scales)
            tol_di = self.ipopt_options['dual_inf_tol'] \
                /np.amax(self.var_scales)

            de   = self.params_t_update['de']
            dia  = self.params_t_update['dia']
            mua  = self.params_t_update['mua']
            dib  = self.params_t_update['dib']
            mub  = self.params_t_update['mub']
            epsb = self.params_t_update['epsb']

            ca = 0.5*(1.0+np.tanh(-2.0*mua*(un_di-dia)))
            cb = 1.0-0.5*epsb*(1.0+np.tanh(-2.0*mub*(un_di-dib)))

            nmove = self.nmove
            barrier = emax_interp - np.amax(self._e_ends)+self.e0
            de = min(2.0/float(nmove+1)*barrier,de)
            delta_e = de*np.arange(0.5*(nmove%2+1.0),0.5*(nmove+1.0),1.0)
            t_de = self.interpolate_energies(delta_e=delta_e)[3]
            t_cand_m = np.hstack([tl[tl<tmax] for tl in t_de])
            t_cand_p = np.hstack([tl[tl>tmax] for tl in t_de])
            temp_t_eval_m = t_cand_m[
                np.argsort(np.abs(t_cand_m-tmax))[:nmove//2]]
            temp_t_eval_p = t_cand_p[
                np.argsort(np.abs(t_cand_p-tmax))[:nmove//2]]
            if nmove%2==1:
                temp_t_eval_p = np.append(temp_t_eval_p,tmax)
            temp_t_eval = np.sort(np.append(temp_t_eval_m,temp_t_eval_p))

            alpha = ca*self._max_alpha
            t_eval = self.t_eval.copy()
            t_eval[1:-1] = (1.0-alpha)*t_eval[1:-1] + alpha*temp_t_eval

            self.set_t_eval(t_eval)
            self.set_w_eval()

            self._max_alpha *= cb


def interpolate_idpp(
        ref_images,nmove=10,
        output_file='idpp_ipopt.out',
        check_collisions=False,
        soft_core=False,
        dmf_options={}):

    mxflx = DirectMaxFlux(ref_images,
                          nmove=nmove,
                          update_teval=False,
                          check_collisions=check_collisions,
                          **dmf_options)

    d1 = ref_images[0].get_all_distances()
    d2 = ref_images[-1].get_all_distances()
    d = (d2 - d1) / (nmove + 1)

    for i,image in enumerate(mxflx.images):
        if soft_core:
            image.calc = SIDPP(d1+i*d,mic=False)
        else:
            image.calc = IDPP(d1+i*d,mic=False)

    options = {
        'tol': 0.01,
        'dual_inf_tol': 0.01,
        'constr_viol_tol': 0.01,
        'compl_inf_tol': 0.01,
        'nlp_scaling_method':'user-scaling',
        'obj_scaling_factor':0.1,
        'limited_memory_initialization':'constant',
        'limited_memory_init_val':2.5,
        'accept_every_trial_step':'yes',
        'max_iter':500,
        'output_file':output_file,
        }
    mxflx.add_ipopt_options(options)

    w_eval0 = mxflx.w_eval.copy()
    b_scale = 3.0

    #for i in range(5):
    #    mxflx.get_forces()
    #    ens = mxflx.energies.copy()
    #    print(ens)
    #    if np.amax(ens)>0.0:
    #        mxflx.beta=b_scale/np.amax(ens)
    #    else:
    #        mxflx.beta=1.0

    #    if i<3:
    #        mxflx.solve(tol=0.1)
    #    else:
    #        mxflx.solve(tol=0.01)

    if check_collisions:
        for i in range(5):
            mxflx.get_forces()
            ens = mxflx.energies.copy()
            if np.amax(ens)>0.0:
                mxflx.beta=b_scale/np.amax(ens)
            else:
                mxflx.beta=1.0

            if i<3:
                mxflx.solve(tol=0.1)
            else:
                mxflx.solve(tol=0.01)
    else:
        for i in range(1,(nmove+1)//2+4):
            mxflx.get_forces()
            ens = mxflx.energies.copy()
            w_eval = w_eval0.copy()
            if i+1 < nmove+1-i:
                ens[i+1:nmove+1-i] = 0.0
                w_eval[i+1:nmove+1-i] = 0.0
            if np.amax(ens)>0.0:
                mxflx.beta=b_scale/np.amax(ens)
            else:
                mxflx.beta=1.0
            mxflx.set_w_eval(w_eval)

            if i<(nmove+1)//2:
                mxflx.solve(tol=0.01)
            else:
                mxflx.solve(tol=0.01)

    emax=np.amax(mxflx.energies)
    print(f'emax: {emax:.5e}')

    return mxflx
