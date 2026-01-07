import threading
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, Union

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
import cyipopt

import ase.parallel

import torch

def _resolve_torch_device(device_spec):
    """Resolve a torch device from a user spec."""
    if device_spec is None:
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            d = torch.device(device_spec)
        except (TypeError, ValueError):
            warnings.warn(f"[torch_dmf] Invalid device spec '{device_spec}', falling back to CPU.")
            d = torch.device("cpu")
    if d.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("[torch_dmf] CUDA requested but not available; falling back to CPU.")
        d = torch.device("cpu")
    return d


@torch.no_grad()
def _interp1d_torch(xq: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolate fp(xp) at xq using pure PyTorch.

    Shapes
    ------
    xp: (M,), fp: (M, *), xq: (K,)
    Returns: (K, *) broadcast along the trailing dims of fp.
    """
    idx = torch.searchsorted(xp, xq, right=False).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    lam = (xq - x0) / (x1 - x0)
    while lam.dim() < fp.dim():
        lam = lam.unsqueeze(-1)
    f0, f1 = fp[idx - 1], fp[idx]
    return f0 + (f1 - f0) * lam


class HistoryBase():
    """
    Container storing the optimization history of the VariationalPathOpt.

    This object collects various physical and numerical quantities evaluated
    along the reaction path during the optimization.  At each IPOPT iteration,
    the ``VariationalPathOpt.intermediate`` method appends the current values
    of these quantities to the corresponding lists below.

    Attributes
    ----------
    forces : list of ndarray
        History of ``VariationalPathOpt.forces``.
    energies : list of ndarray
        History of ``VariationalPathOpt.energies``.
    coefs : list of ndarray
        History of ``VariationalPathOpt.coefs``.
    angs : list of ndarray
        History of ``VariationalPathOpt.angs``.
    tmax : list of float
        History of the location ``t_max`` corresponding to the maximum
        interpolated energy along the path.
    images_tmax : list of ase.Atoms
        History of the atomic structure at ``t = t_max``, providing an
        approximate transition-state geometry at each iteration.
    duals : list of float
        History of the scaled dual infeasibility (IPOPT diagnostic).

    """

    def __init__(self):
        self.forces = []
        self.energies = []
        self.coefs = []
        self.angs = []
        self.tmax = []
        self.images_tmax = []
        self.duals = []


class VariationalPathOpt(ABC, cyipopt.Problem):
    """
    Generic variational path optimization with functional:

        I[x(t)] = ∫_0^1 dt |v(t)| F(x(t))

    Heavy vector operations are kept in torch when helpful; IPOPT-facing
    quantities remain NumPy. Mathematics is unchanged.

    Attributes
    ----------
    images : list[ase.Atoms]
    t_eval, w_eval : np.ndarray
    coefs : np.ndarray, shape (nbasis, natoms, 3)
    angs : np.ndarray, shape (3,)
    energies : np.ndarray
    forces : np.ndarray
    history : HistoryBase
    remove_rotation_and_translation : bool
    natoms, nsegs, dspl, nbasis : int
    n_vel, n_trans, n_rot : int
    eps_vel, eps_rot : float
    ipopt_options : dict

    Optional
    -----------------------
    device : Optional[Union[str, torch.device]]
        Torch device to use (e.g., "cuda:0", "cpu"). If None, auto-select.
    """

    def __init__(
        self,
        ref_images,
        coefs=None, nsegs=4, dspl=3,
        remove_rotation_and_translation=True,
        mass_weighted=False,
        parallel=False, world=None,
        t_eval=None, w_eval=None,
        n_vel=None, n_trans=None, n_rot=None,
        eps_vel=0.01, eps_rot=0.01,
        device=None,
    ):
        self.device = _resolve_torch_device(device)
        _dev = self.device

        # Atoms / masses
        self.natoms = len(ref_images[0])
        if mass_weighted:
            self._masses = ref_images[0].get_masses().astype(np.float64)
        else:
            self._masses = np.ones(self.natoms)
        self._mass_fracs = self._masses / np.sum(self._masses)

        # Constraints
        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.eps_vel = float(eps_vel)
        self.eps_rot = float(eps_rot)

        # Parallel execution
        self.parallel = bool(parallel)
        self._world = ase.parallel.world if world is None else world

        # B-spline basis
        self.nsegs = int(nsegs)
        self.dspl = int(dspl)
        self.nbasis = self.nsegs + self.dspl
        _t_knot = np.concatenate(
            [np.zeros(self.dspl),
             np.linspace(0.0, 1.0, self.nsegs + 1),
             np.ones(self.dspl)]
        )
        self._t_knot = _t_knot
        basis = [BSpline(_t_knot, np.identity(self.nbasis)[i], self.dspl)
                 for i in range(self.nbasis)]
        d1basis = [b.derivative(nu=1) for b in basis]
        d2basis = [b.derivative(nu=2) for b in basis]
        self._basis = [basis, d1basis, d2basis]

        # t sequences and cached basis values
        if t_eval is None:
            self.set_t_eval(np.linspace(0.0, 1.0, 2 * self.nsegs + 1))
        else:
            self.set_t_eval(np.asarray(t_eval))
        self.set_w_eval(w_eval)

        self.n_vel = 4 * self.nsegs if n_vel is None else int(n_vel)
        self.t_vel = np.linspace(0.0, 1.0, self.n_vel + 1)

        self.n_trans = 2 * self.nsegs if n_trans is None else int(n_trans)
        self.t_trans = np.linspace(0.0, 1.0, self.n_trans + 1)[1:-1]

        self.n_rot = 2 * self.nsegs if n_rot is None else int(n_rot)
        self.t_rot = np.linspace(0.0, 1.0, self.n_rot + 1)

        # Basis values [derivative order, basis, t]
        self._P_eval = self._get_basis_values(self.t_eval)
        self._P_vel = self._get_basis_values(self.t_vel)
        self._P_trans = self._get_basis_values(self.t_trans)
        self._P_rot = self._get_basis_values(self.t_rot)

        # Coefficients and angles
        self.coefs = np.empty([self.nbasis, self.natoms, 3])
        self.angs = np.zeros(3)
        if coefs is not None:
            self.coefs[...] = coefs
        else:
            self.coefs[...] = self._get_coefs_from_ref_images(ref_images)
        self._coefs0 = self.coefs.copy()

        # Torch mirrors kept on device
        self._masses_t     = torch.as_tensor(self._masses,     dtype=torch.float64, device=_dev)
        self._mass_fracs_t = torch.as_tensor(self._mass_fracs, dtype=torch.float64, device=_dev)
        self._P_eval_t     = torch.as_tensor(self._P_eval,     dtype=torch.float64, device=_dev)
        self._P_vel_t      = torch.as_tensor(self._P_vel,      dtype=torch.float64, device=_dev)
        self._P_trans_t    = torch.as_tensor(self._P_trans,    dtype=torch.float64, device=_dev)
        self._P_rot_t      = torch.as_tensor(self._P_rot,      dtype=torch.float64, device=_dev)
        self.coefs_t       = torch.as_tensor(self.coefs,       dtype=torch.float64, device=_dev)

        # Initialize images and set coordinates
        self.images = [ref_images[0].copy() for _ in range(self.t_eval.size)]
        self.set_positions()

        # Precompute Jacobian of translation constraints
        self._jac_trans = np.einsum(
            "a,bi,st->isbat", self._mass_fracs,
            self._P_trans[0], np.identity(3)
        )

        self.forces = None
        self.energies = None
        self.history = HistoryBase()

        # IPOPT setup
        nvar = (self.nbasis - 2) * 3 * self.natoms
        if self.remove_rotation_and_translation:
            nvar += 3

        self.var_scales = 1.0

        m_vel = self.t_vel.size - 1
        cl = np.full(m_vel, 1.0 - self.eps_vel)
        cu = np.full(m_vel, 1.0 + self.eps_vel)

        if self.remove_rotation_and_translation:
            cl_trans = np.zeros(3 * self.t_trans.size)
            cu_trans = np.zeros(3 * self.t_trans.size)
            m_rot = 3 * (self.t_rot.size - 1)
            cl_rot = np.full(m_rot, -self.eps_rot)
            cu_rot = np.full(m_rot, self.eps_rot)
            cl = np.hstack([cl, cl_trans, cl_rot])
            cu = np.hstack([cu, cu_trans, cu_rot])

        lb = np.full(nvar, -2.0e19)
        ub = np.full(nvar, 2.0e19)

        cyipopt.Problem.__init__(
            self,
            n=nvar, m=len(cl),
            lb=lb, ub=ub,
            cl=cl, cu=cu,
        )

        defaults = {
            "tol": 1.0,
            "dual_inf_tol": 0.04,
            "constr_viol_tol": 0.01,
            "compl_inf_tol": 0.01,
            "nlp_scaling_method": "user-scaling",
            "obj_scaling_factor": 0.1,
            "limited_memory_initialization": "constant",
            "limited_memory_init_val": 2.5,
            "accept_every_trial_step": "yes",
            "output_file": "pathopt.out",
        }
        if self.parallel and self._world.size > 1 and self._world.rank > 0:
            defaults["print_level"] = 0
        self.ipopt_options = {}
        self.add_ipopt_options(defaults)

    def _get_basis_values(self, t_seq: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [[b(t) for t in t_seq] for b in self._basis[nu]]
                for nu in range(3)
            ]
        )

    def set_t_eval(self, t_eval: np.ndarray):
        """Set integration nodes and refresh basis caches."""
        self.t_eval = np.asarray(t_eval)
        self._P_eval = self._get_basis_values(self.t_eval)
        self._P_eval_t = torch.as_tensor(self._P_eval, dtype=torch.float64, device=self.device)

    def set_w_eval(self, w_eval: Optional[np.ndarray] = None):
        """Set trapezoidal weights used in the action integral."""
        if w_eval is not None:
            self.w_eval = np.asarray(w_eval)
        else:
            w = np.zeros_like(self.t_eval)
            w[0] = 0.5 * (self.t_eval[1] - self.t_eval[0])
            w[-1] = 0.5 * (self.t_eval[-1] - self.t_eval[-2])
            w[1:-1] = 0.5 * (self.t_eval[2:] - self.t_eval[:-2])
            self.w_eval = w

    def _get_coefs_from_ref_images(self, ref_images) -> np.ndarray:
        """
        Build initial B-spline coefficients from reference images
        (with optional rigid motion removal). Keeps NumPy output for IPOPT.
        """
        ref_images_copy = [image.copy() for image in ref_images]

        if self.remove_rotation_and_translation:
            prev_image = None
            for image in ref_images_copy:
                pos = image.get_positions()
                image.translate(-self._mass_fracs @ pos)
                if prev_image is not None:
                    pos = image.get_positions()
                    prev_pos = prev_image.get_positions()
                    r = Rotation.align_vectors(prev_pos, pos, weights=self._masses)[0]
                    image.set_positions(r.apply(pos))
                prev_image = image

        nimages = len(ref_images_copy)
        pos_ref = np.empty([nimages, self.natoms, 3])
        t_ref = np.zeros(nimages)
        for i, image in enumerate(ref_images_copy):
            pos_ref[i] = image.get_positions().astype(np.float64)
        diff = pos_ref[1:] - pos_ref[:-1]
        l = np.sqrt((self._masses[None, :, None] * diff ** 2).sum(axis=(1, 2)))
        t_ref[1:] = np.cumsum(l) / np.sum(l)

        t_ref_interp = np.linspace(0.0, 1.0, 4 * self.nsegs + 1)[1:-1]
        pos_ref_interp = _interp1d_torch(
            torch.as_tensor(t_ref_interp, dtype=torch.float64, device=self.device),
            torch.as_tensor(t_ref, dtype=torch.float64, device=self.device),
            torch.as_tensor(pos_ref, dtype=torch.float64, device=self.device)
        ).cpu().numpy().astype(np.float64)

        P_ref_interp0 = self._get_basis_values(t_ref_interp)[0]
        A = np.matmul(P_ref_interp0[1:-1], P_ref_interp0[1:-1].T)
        x = pos_ref_interp \
            - np.tensordot(P_ref_interp0[0], pos_ref[0], axes=0) \
            - np.tensordot(P_ref_interp0[-1], pos_ref[-1], axes=0)
        y = np.tensordot(P_ref_interp0[1:-1], x, axes=1).reshape(-1, 3 * self.natoms)

        coefs = np.empty([self.nbasis, self.natoms, 3])
        coefs[0] = pos_ref[0]
        coefs[-1] = pos_ref[-1]
        coefs[1:-1] = np.linalg.solve(A, y).reshape(-1, self.natoms, 3)
        return coefs

    def get_positions(self, t=None, P=None, nu=0) -> np.ndarray:
        """
        Return positions or derivatives at nodes t.

        Returns
        -------
        (nimages, natoms, 3) np.ndarray
        """
        t_temp = self.t_eval if t is None else t
        P_temp = self._get_basis_values(t_temp) if P is None else P
        return np.tensordot(P_temp[nu].T, self.coefs, axes=1)

    def set_coefs_angs(self, coefs=None, angs=None):
        """Update internal coefficients/angles and their torch mirrors."""
        if coefs is not None:
            self.coefs = np.asarray(coefs)
            self.coefs_t = torch.as_tensor(self.coefs, dtype=torch.float64, device=self.device)
        if angs is not None:
            self.angs = np.asarray(angs)

        R = self._get_rot_mats()
        self.coefs[-1] = self._coefs0[-1] @ R[0] @ R[1] @ R[2]
        # Keep last layer in sync without reallocating the entire tensor
        self.coefs_t[-1].copy_(torch.as_tensor(self.coefs[-1], dtype=torch.float64, device=self.device))

    def _get_positions_torch(self, P_t: torch.Tensor, nu=0) -> torch.Tensor:
        """Torch evaluation: tensordot(P[nu].T, coefs)."""
        return torch.tensordot(P_t[nu].T.contiguous(), self.coefs_t, dims=1)

    def _get_rot_mats(self) -> np.ndarray:
        """3 intrinsic rotations for end-image alignment (NumPy)."""
        R = np.zeros([3, 3, 3])
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            R[i, i, i] = 1.0
            R[i, j, j] = np.cos(self.angs[i])
            R[i, j, k] = -np.sin(self.angs[i])
            R[i, k, j] = np.sin(self.angs[i])
            R[i, k, k] = np.cos(self.angs[i])
        return R

    def set_positions(self, coefs=None, angs=None):
        """Push current coefs/angs into all images."""
        self.set_coefs_angs(coefs, angs)
        pos = self.get_positions()
        for i in range(self.t_eval.size):
            self.images[i].set_positions(pos[i])

    # --- Constraints & Jacobians --------------------------------
    def _get_consts_trans(self) -> np.ndarray:
        pos = self.get_positions(P=self._P_trans)
        return self._mass_fracs @ pos

    def _get_jac_trans(self) -> np.ndarray:
        return self._jac_trans

    def _get_consts_rot(self) -> np.ndarray:
        pos = self.get_positions(P=self._P_rot)
        return self._mass_fracs @ np.cross(pos[:-1], pos[1:])

    def _get_jac_rot(self) -> np.ndarray:
        pos = self.get_positions(P=self._P_rot)
        y = np.cross(np.identity(3), pos[..., None, :])
        jac_rot = (
            np.einsum("a,bi,iats->isbat", self._mass_fracs, self._P_rot[0, :, :-1], y[1:])
            - np.einsum("a,bi,iats->isbat", self._mass_fracs, self._P_rot[0, :, 1:], y[:-1])
        )
        return jac_rot

    def _get_consts_vel(self) -> np.ndarray:
        pos_t = self._get_positions_torch(self._P_vel_t)
        diffs = pos_t[1:] - pos_t[:-1]
        d2s = torch.sum(self._masses_t[None, :, None] * diffs ** 2, dim=(1, 2))
        out = (d2s / torch.mean(d2s)).cpu().numpy().astype(np.float64)
        return out

    def _get_jac_vel(self) -> np.ndarray:
        pos_t = self._get_positions_torch(self._P_vel_t)
        diffs = pos_t[1:] - pos_t[:-1]                                               # (m, nat, 3)
        d2s = torch.sum(self._masses_t[None, :, None] * diffs ** 2, dim=(1, 2))      # (m,)
        diff_P = self._P_vel_t[0, :, 1:] - self._P_vel_t[0, :, :-1]                  # (nbasis, m)

        jac_d2s = 2.0 * torch.einsum("a,bi,ias->ibas", self._masses_t, diff_P, diffs)  # (m, nbasis, nat, 3)
        ave_d2s = torch.mean(d2s)
        jac = jac_d2s / ave_d2s - torch.tensordot(d2s, torch.mean(jac_d2s, dim=0), dims=0) / (ave_d2s * ave_d2s)
        return jac.cpu().numpy().astype(np.float64)

    def _get_jac_fin_rot(self) -> np.ndarray:
        R = self._get_rot_mats()
        dR = np.zeros([3, 3, 3])
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            dR[i, j, j] = -np.sin(self.angs[i])
            dR[i, j, k] = -np.cos(self.angs[i])
            dR[i, k, j] = np.cos(self.angs[i])
            dR[i, k, k] = -np.sin(self.angs[i])
        jac_rot = np.empty([self.natoms, 3, 3])
        jac_rot[..., 0] = self._coefs0[-1] @ dR[0] @ R[1] @ R[2]
        jac_rot[..., 1] = self._coefs0[-1] @ R[0] @ dR[1] @ R[2]
        jac_rot[..., 2] = self._coefs0[-1] @ R[0] @ R[1] @ dR[2]
        return jac_rot

    def _reshape_jacs(self, jacs):
        def remove_axis(jac):
            return jac[0] if len(jac) == 1 else jac

        aligned_jac = np.vstack([jac.reshape([-1, self.nbasis, self.natoms, 3]) for jac in jacs])
        nc = len(aligned_jac)
        jac_coefs = aligned_jac[:, 1:-1, :, :].reshape([nc, -1])

        if self.remove_rotation_and_translation:
            jac_fin_rot = self._get_jac_fin_rot()
            jac_rot = np.tensordot(aligned_jac[:, -1, :, :], jac_fin_rot)
            return remove_axis(np.hstack([jac_coefs, jac_rot]))
        else:
            return remove_axis(jac_coefs)

    def _reshape_consts(self, consts):
        return np.hstack([np.ravel(c) for c in consts]).astype(np.float64)

    # --- Endpoints energies/forces -----------

    @cached_property
    def _f_ends(self):
        forces = np.empty((2, self.natoms, 3))

        if not self.parallel or self._world.size == 1:
            forces[0] = self.images[0].get_forces()
            forces[1] = self.images[-1].get_forces()
            return forces

        # MPI case
        nmv = len(self.images) - 2
        i = self._world.rank * nmv // self._world.size
        try:
            if i == 0:
                forces[0] = self.images[0].get_forces()
            elif i == 1:
                forces[-1] = self.images[-1].get_forces()
        except Exception:
            error = self._world.sum(1.0)
            raise
        else:
            error = self._world.sum(0.0)
            if error:
                raise RuntimeError("Parallel DMF failed!")

        root0 = 0
        root1 = self._world.size // nmv
        self._world.broadcast(forces[0], root0)
        self._world.broadcast(forces[-1], root1)
        return forces

    @cached_property
    def _e_ends(self):
        _ = self._f_ends  # ensure forces are computed
        energies = np.empty(2)
        if (not self.parallel) or self._world.size == 1:
            energies[0] = self.images[0].get_potential_energy()
            energies[1] = self.images[-1].get_potential_energy()
            return energies

        nmv = len(self.images) - 2
        root0 = 0
        root1 = self._world.size // nmv
        if self._world.rank == root0:
            energies[0] = self.images[0].get_potential_energy()
        elif self._world.rank == root1:
            energies[1] = self.images[-1].get_potential_energy()
        self._world.broadcast(energies[0:1], root0)
        self._world.broadcast(energies[1:2], root1)
        return energies

    @cached_property
    def e0(self) -> float:
        return float(np.amin(self._e_ends))

    def get_forces(self) -> np.ndarray:
        """
        Compute forces for all images. Stores `self.forces` and `self.energies`.
        """
        eps_t = 0.01

        forces = np.empty([self.t_eval.size, self.natoms, 3])
        energies = np.empty(self.t_eval.size)

        inds = []
        for i, t in enumerate(self.t_eval):
            if t < eps_t:
                forces[i] = self._f_ends[0]
                energies[i] = self._e_ends[0]
            elif t > 1.0 - eps_t:
                R = self._get_rot_mats()
                f = self._f_ends[1]
                forces[i] = f @ R[0] @ R[1] @ R[2]
                energies[i] = self._e_ends[1]
            else:
                inds.append(i)

        if not self.parallel:
            for i in inds:
                forces[i] = self.images[i].get_forces()
                energies[i] = self.images[i].get_potential_energy()
        elif self._world.size == 1:
            def run(image, out_energy, out_forces):
                out_forces[:] = image.get_forces()
                out_energy[:] = image.get_potential_energy()
            threads = [threading.Thread(target=run,
                                        args=(self.images[i], energies[i:i+1], forces[i:i+1]))
                       for i in inds]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        else:
            nmv = len(self.images) - 2
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
                    raise RuntimeError("Parallel DMF failed!")
            for i in range(1, nmv + 1):
                root = (i - 1) * self._world.size // nmv
                self._world.broadcast(energies[i:i + 1], root)
                self._world.broadcast(forces[i], root)

        self.energies = energies
        self.forces = forces
        return forces

    # --- Action (objective) ------------------------------------------------------
    @abstractmethod
    def _get_objective(self) -> float:
        ...

    @abstractmethod
    def _get_grad_objective(self) -> np.ndarray:
        ...

    @abstractmethod
    def _get_func_en(self, en: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def _get_norm_vels(self, nu: int = 0):
        """
        Finite difference norm of velocities ||dx/dt|| with mass weighting.
        nu=0 returns values on self.t_eval, nu=1 returns gradient wrt coefs.
        """
        pos_t = self._get_positions_torch(self._P_vel_t)
        diffs = pos_t[1:] - pos_t[:-1]
        norm_dx = torch.sqrt(torch.sum(self._masses_t[None, :, None] * diffs ** 2, dim=(1, 2)))
        dt = torch.as_tensor(self.t_vel[1:] - self.t_vel[:-1], dtype=torch.float64, device=self.device)

        if nu == 0:
            fd_vels = torch.zeros(self.t_vel.size + 1, dtype=torch.float64, device=self.device)
            fd_vels[1:-1] = norm_dx / dt
            fd_vels[0] = fd_vels[1]
            fd_vels[-1] = fd_vels[-2]

            t_fd_vel = torch.zeros_like(fd_vels)
            t_fd_vel[1:-1] = 0.5 * (torch.as_tensor(self.t_vel[1:], dtype=torch.float64, device=self.device) +
                                    torch.as_tensor(self.t_vel[:-1], dtype=torch.float64, device=self.device))
            t_fd_vel[-1] = torch.tensor(1.0, dtype=torch.float64, device=self.device)

            return _interp1d_torch(
                torch.as_tensor(self.t_eval, dtype=torch.float64, device=self.device),
                t_fd_vel, fd_vels
            ).cpu().numpy().astype(np.float64)

        diff_P_vel0 = self._P_vel_t[0, :, 1:] - self._P_vel_t[0, :, :-1]
        grad_norm = torch.einsum("i,bi,a,ias->ibas", 1.0 / (dt * norm_dx), diff_P_vel0, self._masses_t, diffs)
        grad_fd_vels = torch.zeros(
            (self.t_vel.size + 1, self.nbasis, self.natoms, 3),
            dtype=torch.float64, device=self.device
        )
        grad_fd_vels[1:-1] = grad_norm
        grad_fd_vels[0] = grad_norm[0]
        grad_fd_vels[-1] = grad_norm[-1]

        t_fd_vel = torch.zeros(self.t_vel.size + 1, dtype=torch.float64, device=self.device)
        t_fd_vel[1:-1] = 0.5 * (torch.as_tensor(self.t_vel[1:], dtype=torch.float64, device=self.device) +
                                torch.as_tensor(self.t_vel[:-1], dtype=torch.float64, device=self.device))
        t_fd_vel[-1] = torch.tensor(1.0, dtype=torch.float64, device=self.device)

        return _interp1d_torch(
            torch.as_tensor(self.t_eval, dtype=torch.float64, device=self.device),
            t_fd_vel, grad_fd_vels
        ).cpu().numpy().astype(np.float64)

    def _get_action(self) -> float:
        self.set_positions()
        self.get_forces()
        norm_vels = self._get_norm_vels()
        fe, _ = self._get_func_en(self.energies)
        return float(np.sum(self.w_eval * norm_vels * fe))

    def _get_grad_action(self) -> np.ndarray:
        self.set_positions()
        self.get_forces()
        fe, dfe = self._get_func_en(self.energies)
        norm_vels = self._get_norm_vels()
        grad_norm_vels = self._get_norm_vels(nu=1)
        grad_action = (np.tensordot(self.w_eval * fe, grad_norm_vels, axes=1)
                       - np.tensordot(self._P_eval[0] * self.w_eval * norm_vels * dfe,
                                      self.forces, axes=1))
        return grad_action.astype(np.float64)

    # --- Energy interpolation ----------------------------------
    def interpolate_energies(
        self, t_eval=None, energies=None, forces=None, coefs=None,
        delta_e=None
    ):
        """
        Piecewise-cubic interpolation of energy along the path (Ref. 1).
        """
        if t_eval is None:
            t_eval = self.t_eval
        i_fin = np.where(t_eval > 0.99)[0][0]
        t_eval = t_eval[:i_fin + 1]

        if energies is None:
            energies = self.energies
        energies = energies[:i_fin + 1]

        if forces is None:
            forces = self.forces
        forces = forces[:i_fin + 1]

        if coefs is None:
            coefs = self.coefs

        P_eval1 = self._get_basis_values(t_eval)[1]
        d_energies = -np.einsum("bi,bas,ias->i", P_eval1, coefs, forces)

        t_pows = np.zeros([2 * len(t_eval), 4])
        for i in range(4):
            t_pows[::2, i] = t_eval ** i
            if i < 3:
                t_pows[1::2, i + 1] = (i + 1) * t_eval ** i

        ens_dens = np.zeros(2 * len(t_eval))
        ens_dens[::2] = energies
        ens_dens[1::2] = d_energies

        polys = np.zeros([len(t_eval) - 1, 4])
        for i in range(len(t_eval) - 1):
            polys[i] = np.linalg.solve(t_pows[2 * i:2 * i + 4], ens_dens[2 * i:2 * i + 4])

        if d_energies[np.argmax(energies)] > 0.0:
            imax = np.argmax(energies)
        else:
            imax = np.argmax(energies) - 1

        if imax == -1:
            t_max = 0.0
            e_max = energies[0]
        elif imax == i_fin:
            t_max = 1.0
            e_max = energies[-1]
        else:
            t_max = -(polys[imax, 2] + np.sqrt(polys[imax, 2] ** 2 - 3.0 * polys[imax, 1] * polys[imax, 3])) \
                    / (3.0 * polys[imax, 3])
            t_max_pow = np.array([t_max ** i for i in range(4)])
            e_max = np.sum(t_max_pow * polys[imax])

        if delta_e is not None:
            t_de = []
            for de in delta_e:
                tlist = np.array([])
                for i in range(len(t_eval) - 1):
                    poly = P.Polynomial(polys[i])
                    poly -= e_max - de
                    roots = poly.roots()
                    roots = roots.real[np.abs(roots.imag) < 1e-5]
                    roots = roots[(roots >= t_eval[i]) & (roots < t_eval[i + 1])]
                    tlist = np.append(tlist, roots)
                t_de.append(tlist)
            return polys, t_max, e_max, t_de

        return polys, t_max, e_max

    # --- IPOPT glue --------------------------------------------------------------
    def solve(self, tol="tight"):
        """
        Solve with IPOPT. `tol` can be float, 'tight', 'middle', or 'loose'.
        """
        if tol:
            if isinstance(tol, float):
                self.add_ipopt_options({"dual_inf_tol": tol})
            elif isinstance(tol, str):
                tt = tol.strip().upper()
                if tt == "TIGHT":
                    self.add_ipopt_options({"dual_inf_tol": 0.04})
                elif tt == "MIDDLE":
                    self.add_ipopt_options({"dual_inf_tol": 0.1})
                elif tt == "LOOSE":
                    self.add_ipopt_options({"dual_inf_tol": 0.2})

        x0 = self.get_x()
        x, info = super().solve(x0)
        self.set_x(x)
        return x, info

    def add_ipopt_options(self, dict_options):
        self.ipopt_options.update(dict_options)
        for item in self.ipopt_options.items():
            self.add_option(*item)

    def get_x(self) -> np.ndarray:
        """Flatten variables (coefs[1:-1], optionally angles)."""
        x = self.coefs[1:-1].ravel()
        if self.remove_rotation_and_translation:
            x = np.hstack([x, self.angs])
        return x.astype(np.float64)

    def set_x(self, x: np.ndarray):
        """Inverse of get_x: update coefs/angles and push to images."""
        nc = (self.nbasis - 2) * 3 * self.natoms
        coefs = self._coefs0.copy()
        coefs[1:-1] = x[:nc].reshape((-1, self.natoms, 3))
        angs = np.zeros(3)
        if self.remove_rotation_and_translation:
            angs = x[-3:]
        self.set_positions(coefs, angs)

    def objective(self, x: np.ndarray) -> float:
        self.set_x(x)
        return self._get_objective()

    def gradient(self, x: np.ndarray) -> np.ndarray:
        self.set_x(x)
        grad = self._reshape_jacs([self._get_grad_objective()])
        return (grad * self.var_scales).astype(np.float64)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        self.set_x(x)
        c_list = [self._get_consts_vel()]
        if self.remove_rotation_and_translation:
            c_list.append(self._get_consts_trans())
            c_list.append(self._get_consts_rot())
        return self._reshape_consts(c_list)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        self.set_x(x)
        j_list = [self._get_jac_vel()]
        if self.remove_rotation_and_translation:
            j_list.append(self._get_jac_trans())
            j_list.append(self._get_jac_rot())
        return (self._reshape_jacs(j_list) * self.var_scales).astype(np.float64)

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm, regularization_size,
                     alpha_du, alpha_pr, ls_trials):
        """Called every iteration by IPOPT."""
        self.history.forces.append(self.forces)
        self.history.energies.append(self.energies)
        self.history.coefs.append(self.coefs)
        self.history.angs.append(self.angs)
        self.history.duals.append(inf_du)

        polys, tmax, emax_interp = self.interpolate_energies()
        P_tmax = np.array([b(tmax) for b in self._basis[0]])
        image_tmax = self.images[0].copy()
        image_tmax.set_positions(np.tensordot(P_tmax, self.coefs, axes=1))
        self.history.tmax.append(tmax)
        self.history.images_tmax.append(image_tmax)


class HistoryDMF():
    """
    Container storing the optimization history of the ``DirectMaxFlux`` method.

    This object collects various physical and numerical quantities evaluated
    along the reaction path during the optimization.  At each IPOPT iteration,
    the ``DirectMaxFlux.intermediate`` method appends the current values of
    these quantities to the corresponding lists below.

    Attributes
    ----------
    forces : list of ndarray
        History of ``DirectMaxFlux.forces``.
    energies : list of ndarray
        History of ``DirectMaxFlux.energies``.
    coefs : list of ndarray
        History of ``DirectMaxFlux.coefs``.
    angs : list of ndarray
        History of ``DirectMaxFlux.angs``.
    t_eval : list of ndarray
        History of ``DirectMaxFlux.t_eval``.
    tmax : list of float
        History of the location ``t_max`` corresponding to the maximum
        interpolated energy along the path.
    images_tmax : list of ase.Atoms
        History of the atomic structure at ``t = t_max``, providing an
        approximate transition-state geometry at each iteration.
    duals : list of float
        History of the scaled dual infeasibility (IPOPT diagnostic).

    """

    def __init__(self):
        self.forces = []
        self.energies = []
        self.coefs = []
        self.angs = []
        self.t_eval = []
        self.tmax = []
        self.images_tmax = []
        self.duals = []


class DirectMaxFlux(VariationalPathOpt):
    """
    Direct MaxFlux variational problem (identical equations).

    Parameters
    ----------
    ref_images : list[ase.Atoms]
    coefs : np.ndarray, optional
    nsegs, dspl : int
    remove_rotation_and_translation : bool
    mass_weighted : bool
    parallel : bool
    world : MPI world object
    t_eval, w_eval : np.ndarray
    n_vel, n_trans, n_rot : int
    eps_vel, eps_rot : float
    beta : float
    nmove : int
    update_teval : bool
    params_t_update : dict

    Optional
    --------
    device : Optional[Union[str, torch.device]]
        Torch device for internal tensors.
    """

    def __init__(
        self,
        ref_images,
        coefs=None, nsegs: int = 4, dspl: int = 3,
        remove_rotation_and_translation: bool = True,
        mass_weighted: bool = False,
        parallel: bool = False, world=None,
        t_eval: Optional[np.ndarray] = None,
        w_eval: Optional[np.ndarray] = None,
        n_vel: Optional[int] = None,
        n_trans: Optional[int] = None,
        n_rot: Optional[int] = None,
        eps_vel: float = 0.01,
        eps_rot: float = 0.01,
        beta: float = 10.0,
        nmove: int = 5,
        update_teval: bool = False,
        params_t_update: Optional[dict] = None,
        device=None,
    ):
        if params_t_update is None:
            params_t_update = {}
        _ptu_def = dict(
            max_alpha0=0.1, de=0.15,
            dia=1.0, mua=5.0,
            dib=0.2, mub=5.0, epsb=0.02
        )
        for k, v in _ptu_def.items():
            params_t_update.setdefault(k, v)

        t_eval_init = np.linspace(0.0, 1.0, nmove + 2)

        base_args = dict(
            ref_images=ref_images,
            coefs=coefs,
            nsegs=nsegs,
            dspl=dspl,
            remove_rotation_and_translation=remove_rotation_and_translation,
            mass_weighted=mass_weighted,
            parallel=parallel,
            world=world,
            t_eval=t_eval_init,
            w_eval=w_eval,
            n_vel=n_vel,
            n_trans=n_trans,
            n_rot=n_rot,
            eps_vel=eps_vel,
            eps_rot=eps_rot,
            device=device,
        )

        self.beta: float = float(beta)
        self.params_t_update: dict = params_t_update
        self._max_alpha: float = params_t_update["max_alpha0"]
        self.update_teval: bool = bool(update_teval)
        self.nmove: int = int(nmove)

        super().__init__(**base_args)
        self.history = HistoryDMF()

    def get_forces(self):
        super().get_forces()
        self.energies = (self.energies - self.e0).astype(np.float64)

        return self.forces

    def _get_objective(self):
        return float(np.log(self._get_action()) / self.beta)

    def _get_grad_objective(self):
        return (self._get_grad_action() / self._get_action() / self.beta).astype(np.float64)

    def _get_func_en(self, en: np.ndarray):
        expbe = np.exp(self.beta * en).astype(np.float64)
        return expbe, (self.beta * expbe).astype(np.float64)

    def intermediate(
        self, alg_mod, iter_count, obj_value,
        inf_pr, inf_du, mu, d_norm, regularization_size,
        alpha_du, alpha_pr, ls_trials,
    ):
        super().intermediate(
            alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials,
        )
        if not self.update_teval:
            return

        un_di = inf_du / self.ipopt_options["obj_scaling_factor"] / np.amax(self.var_scales)
        tol_di = self.ipopt_options["dual_inf_tol"] / np.amax(self.var_scales)

        polys, tmax, emax_interp = self.interpolate_energies()
        self.history.t_eval.append(self.t_eval.copy())

        p = self.params_t_update
        ca = 0.5 * (1.0 + np.tanh(-2.0 * p["mua"] * (un_di - p["dia"])))
        cb = 1.0 - 0.5 * p["epsb"] * (1.0 + np.tanh(-2.0 * p["mub"] * (un_di - p["dib"])))

        barrier = emax_interp - np.amax(self._e_ends) + self.e0
        de_unit = min(2.0 / (self.nmove + 1.0) * barrier, p["de"])
        delta_e = de_unit * np.arange(
            0.5 * ((self.nmove % 2) + 1.0),
            0.5 * (self.nmove + 1.0),
            1.0,
        )

        t_de_lists = self.interpolate_energies(delta_e=delta_e)[3]
        t_cand_m = np.hstack([tl[tl < tmax] for tl in t_de_lists])
        t_cand_p = np.hstack([tl[tl > tmax] for tl in t_de_lists])
        temp_t_eval_m = t_cand_m[np.argsort(np.abs(t_cand_m - tmax))[: self.nmove // 2]]
        temp_t_eval_p = t_cand_p[np.argsort(np.abs(t_cand_p - tmax))[: self.nmove // 2]]
        if self.nmove % 2 == 1:
            temp_t_eval_p = np.append(temp_t_eval_p, tmax)
        temp_t_eval = np.sort(np.append(temp_t_eval_m, temp_t_eval_p))

        alpha = ca * self._max_alpha
        t_eval_new = self.t_eval.copy()
        t_eval_new[1:-1] = (1.0 - alpha) * t_eval_new[1:-1] + alpha * temp_t_eval
        self.set_t_eval(t_eval_new)
        self.set_w_eval()

        self._max_alpha *= cb
