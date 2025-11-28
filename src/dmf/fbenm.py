import numpy as np

from ase.calculators.calculator import Calculator
from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii


class FB_ENM(Calculator):
    """General form of Flat Bottom Elastic Network Model
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, d_min,d_max,
                 delta_min=None,delta_max=None,delta_scale=0.2):
        Calculator.__init__(self)

        I = np.identity(len(d_min),dtype='bool')

        self.d_min = d_min
        self.d_max = d_max
        if delta_min is not None:
            self.delta_min = delta_min
        else:
            self.delta_min = delta_scale * d_min
        if delta_max is not None:
            self.delta_max = delta_max
        else:
            self.delta_max = delta_scale * d_max

        self.d_min[I] = 0.0
        self.d_max[I] = 0.0
        self.delta_min[I] = 1.0
        self.delta_max[I] = 1.0

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        def vecsum(A,B):
            return np.einsum('ij,ijk->ik',A,B)

        r = atoms.get_positions()
        dr = r[:,np.newaxis,:]-r
        d = np.sqrt(np.einsum('ijk,ijk->ij',dr,dr))
        dwI = d+np.identity(len(d))

        d_rep = np.fmin(0.0,d-self.d_min)
        d_att = np.fmax(0.0,d-self.d_max)

        e_rep = d_rep**2/(self.delta_min)**2
        e_att = d_att**2/(self.delta_max)**2
        f0 = e_rep + e_att

        f1 = 2.0*( d_rep/(self.delta_min)**2 \
                  +d_att/(self.delta_max)**2 )

        grad_en = vecsum(f1/dwI,dr)

        e = 0.5*f0.sum()
        f = -grad_en

        self.results = {'energy': e,
                        'forces': f,
                        'emat_rep':e_rep,
                        'emat_att':e_att}

class FB_ENM_Bonds(FB_ENM):
    """All atom FB-ENM featuring chemical bonds
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self,
                 images,
                 addA=None,
                 delA=None,
                 delta_scale=0.2,
                 bond_scale=1.25,
                 fix_planes=True,
                 d_min_overwrite=None,
                 d_max_overwrite=None,
                 A_overwrite=None,):

        cov_radii = covalent_radii[images[0].arrays['numbers']]
        r_cov = cov_radii + cov_radii[:,None]
        v_radii = vdw_radii[images[0].arrays['numbers']]
        r_vdw = v_radii + v_radii[:,None]

        nimages = len(images)
        natoms = len(images[0])

        d_mins = np.zeros([nimages,natoms,natoms])
        d_maxs = np.zeros([nimages,natoms,natoms])

        if fix_planes:
            addA_p = np.zeros([natoms,natoms],dtype='bool')
            planes = get_planes(images,bond_scale=bond_scale)
            for p in planes:
                addA_p[np.ix_(p,p)] = True

        for i,image in enumerate(images):
            d = image.get_all_distances()
            A = (d/r_cov)<bond_scale
            A = A@A

            if fix_planes:
                A = A|addA_p

            if addA is not None:
                A = A|addA

            if delA is not None:
                A = A&(~delA)

            d_mins[i] = np.where(A,d,np.fmin(d,r_vdw))
            d_maxs[i] = np.where(A,d,2.0*np.max(d))

        d_min = np.min(d_mins,axis=0)
        if d_min_overwrite is not None:
            d_min[A_overwrite] = d_min_overwrite[A_overwrite]

        d_max = np.max(d_maxs,axis=0)
        if d_max_overwrite is not None:
            d_max[A_overwrite] = d_max_overwrite[A_overwrite]

        super().__init__(d_min,d_max,delta_scale=delta_scale)

class CFB_ENM(Calculator):
    """General form of Flat Bottom Elastic Network Model
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, images, bond_scale=1.25,
                 d_corr0=None, corr0_scale=1.10,
                 d_corr1=None, corr1_scale=1.50,
                 d_corr2=None, corr2_scale=1.60,
                 eps=0.05,
                 pivotal=True,
                 single=True,
                 remove_fourmembered=True,
                 ):

        Calculator.__init__(self)

        cov_radii = covalent_radii[images[0].arrays['numbers']]
        r_cov = cov_radii + cov_radii[:,None]

        nimages = len(images)
        natoms = len(images[0])

        d_bonds = np.zeros([nimages,natoms,natoms])

        Js = []
        for i,image in enumerate(images):
            d = image.get_all_distances()
            J = (d/r_cov)<bond_scale
            np.fill_diagonal(J,False)
            Js.append(J)

            d_bonds[i] = np.where(J,d,0.0)

        self.d_bond = np.max(d_bonds,axis=0)

        J_only_r = Js[0]&(~Js[-1])
        J_only_p = Js[-1]&(~Js[0])
        J_both = Js[0]&Js[-1]

        self.quartets = self.get_quartets(
                            J_only_r,J_only_p,J_both,
                            pivotal=pivotal,single=single,
                            remove_fourmembered=remove_fourmembered)

        if d_corr0 is not None:
            self.d_corr0 = d_corr0
        else:
            self.d_corr0 = corr0_scale * self.d_bond

        if d_corr1 is not None:
            self.d_corr1 = d_corr1
        else:
            self.d_corr1 = corr1_scale * self.d_bond

        if d_corr2 is not None:
            self.d_corr2 = d_corr2
        else:
            self.d_corr2 = corr2_scale * self.d_bond

        self.eps = eps

        I = np.identity(natoms,dtype='bool')
        self.d_bond[I] = 0.0
        self.d_corr0[I] = 0.0
        self.d_corr1[I] = 0.0
        self.d_corr2[I] = 0.0

    def get_quartets(self,J_only_r,J_only_p,J_both,
            pivotal=True,single=True,remove_fourmembered=True):

        J2 = J_both@J_both

        if pivotal:
            quartets = []
            if single:
                pivots = np.where((np.sum(J_only_r,axis=1)==1)
                                        &(np.sum(J_only_p,axis=1)==1))[0]
            else:
                pivots = np.where(np.any(J_only_r,axis=1)
                                        &np.any(J_only_p,axis=1))[0]
            for i in pivots:
                only_r = np.where(J_only_r[i])[0]
                only_p = np.where(J_only_p[i])[0]
                for j in only_r:
                    for k in only_p:
                        if (not (remove_fourmembered and J2[j,k])):
                            quartets.append(list(map(int,[i,j,i,k])))

        else:
            pairs_only_r = []
            pairs_only_p = []
            for i in range(len(J_only_r)):
                for j in range(i):
                    if J_only_r[i,j]:
                        pairs_only_r.append([i,j])
                    if J_only_p[i,j]:
                        pairs_only_p.append([i,j])

            quartets = []
            for pr in pairs_only_r:
                for pp in pairs_only_p:
                    q = pr+pp

                    if remove_fourmembered:
                        uniq_idxs = [q[i] for i in range(4) if q.count(q[i])==1]

                        if len(uniq_idxs)==4:
                            is_fourmembered = \
                                (J_both[q[0],q[2]] and J_both[q[1],q[3]]) \
                                or (J_both[q[0],q[3]] and J_both[q[1],q[2]])
                        else:
                            is_fourmembered = J2[uniq_idxs[0],uniq_idxs[1]]

                        if is_fourmembered:
                            continue

                    quartets.append(q)

        return quartets

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        r = atoms.get_positions()
        dr = r[:,np.newaxis,:]-r
        d = np.sqrt(np.einsum('ijk,ijk->ij',dr,dr))

        e = 0.0
        forces = np.zeros([len(atoms),3])

        d_d0 = d - self.d_corr0
        d1_d0 = self.d_corr1 - self.d_corr0
        d2_d0 = self.d_corr2 - self.d_corr0

        for i,t in enumerate(self.quartets):
            pp = (d_d0[t[0],t[1]]*d_d0[t[2],t[3]]
                   - d1_d0[t[0],t[1]]*d1_d0[t[2],t[3]])

            if (d_d0[t[0],t[1]]>0.0 and d_d0[t[2],t[3]]>0.0 and pp>0.0):

                v1 = d_d0[t[2],t[3]]/d[t[0],t[1]]*(r[t[0]]-r[t[1]])
                v2 = d_d0[t[0],t[1]]/d[t[2],t[3]]*(r[t[2]]-r[t[3]])

                dnm = ( d2_d0[t[0],t[1]]*d2_d0[t[2],t[3]]
                      - d1_d0[t[0],t[1]]*d1_d0[t[2],t[3]])

                pp /= dnm
                v1 /= dnm
                v2 /= dnm

                sqrt_pp2 = np.sqrt(pp**2+self.eps**2)

                alpha = pp/sqrt_pp2

                e += sqrt_pp2-self.eps

                forces[t[0]] -= alpha*v1
                forces[t[1]] += alpha*v1
                forces[t[2]] -= alpha*v2
                forces[t[3]] += alpha*v2

        self.results = {'energy': e,
                        'forces': forces}


def get_planes(images,bond_scale=1.25,tol_rmsd=0.05,tol_ang=10.0):

    def rmsd(pos,c4):
        x = pos[c4]
        cent = np.mean(x, axis=0)
        _, _, vh = np.linalg.svd(x-cent)
        v = vh[-1, :]
        d = np.dot(x-cent, v)
        return np.sqrt(np.mean(d**2))

    def is_not_linear(atoms,c4):
        ang0 = atoms.get_angle(*c4[0:3])
        ang1 = atoms.get_angle(*c4[1:4])
        return 180.0-ang0>tol_ang and 180.0-ang1>tol_ang

    def is_cis(atoms,c4):
        dh = atoms.get_dihedral(*c4)
        return np.cos(np.pi/180*dh)>=0.0

    def is_trans(atoms,c4):
        dh = atoms.get_dihedral(*c4)
        return np.cos(np.pi/180*dh)<0.0

    def is_connected(nghs,c4):
        ret = c4[0] in nghs[c4[1]]
        ret = ret and c4[1] in nghs[c4[2]]
        ret = ret and c4[2] in nghs[c4[3]]
        return ret

    def is_connected_center(nghs,c4):
        ret = c4[0] in nghs[c4[1]]
        ret = ret and c4[0] in nghs[c4[2]]
        ret = ret and c4[0] in nghs[c4[3]]
        return ret

    for iimg, atoms in enumerate(images):

        pos = atoms.get_positions()
        cov_radii = covalent_radii[atoms.arrays['numbers']]
        r_cov = cov_radii + cov_radii[:,None]
        d = atoms.get_all_distances()
        A = (d/r_cov)<bond_scale
        np.fill_diagonal(A,False)
        nghs = []
        for l in A:
            nghs.append(np.where(l)[0])

        if iimg==0:
            path = []
            c4s = []
            def next(i):
                if i not in path:
                    path.append(i)
                    if len(path)==4:
                        if path[0]<path[3]:
                            c4s.append(list(path))
                    else:
                        for j in nghs[i]:
                            next(j)
                    path.pop()

            for i in range(len(atoms)):
                next(i)

            c4s_center = []
            for i0 in range(len(atoms)):
                nngh = len(nghs[i0])
                if nngh>=3:
                    for i1 in range(nngh):
                        for i2 in range(i1+1,nngh):
                            for i3 in range(i2+1,nngh):
                                c4s_center.append([i0,
                                                   nghs[i0][i1],
                                                   nghs[i0][i2],
                                                   nghs[i0][i3]])

            pels_cis = [c4 for c4 in c4s if (rmsd(pos,c4)<tol_rmsd
                                             and is_not_linear(atoms,c4)
                                             and is_cis(atoms,c4))]
            pels_trans = [c4 for c4 in c4s if (rmsd(pos,c4)<tol_rmsd
                                             and is_not_linear(atoms,c4)
                                             and is_trans(atoms,c4))]
            pels_center = [c4 for c4 in c4s_center if (rmsd(pos,c4)<tol_rmsd)]

        else:
            pels_cis = [c4 for c4 in pels_cis
                           if (rmsd(pos,c4)<tol_rmsd
                               and is_not_linear(atoms,c4)
                               and is_cis(atoms,c4)
                               and is_connected(nghs,c4))]
            pels_trans = [c4 for c4 in pels_trans
                             if (rmsd(pos,c4)<tol_rmsd
                                 and is_not_linear(atoms,c4)
                                 and is_trans(atoms,c4)
                                 and is_connected(nghs,c4))]
            pels_center = [c4 for c4 in pels_center
                              if (rmsd(pos,c4)<tol_rmsd
                                  and is_connected_center(nghs,c4))]

    pels = [set(pel) for pel in pels_cis+pels_trans+pels_center]


    planes = []
    pels_del = []

    while len(pels)>0:
        if len(pels_del)==0:
            planes.append(pels[-1])
            pels.pop()

        pels_del = [pel for pel in pels if len(planes[-1]&pel)>=3]
        pels = [pel for pel in pels if len(planes[-1]&pel)<3]
        planes[-1] = planes[-1].union(*pels_del)

    return [sorted(p) for p in planes]
