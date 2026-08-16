import threading
from typing import Optional, Tuple, Union

import numpy as np

from ase.calculators.calculator import Calculator
from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii

import torch
from ._torch_config import _resolve_torch_device, _resolve_torch_dtype


@torch.no_grad()
def _calc_dist_mat(pos, *, device, dtype) -> torch.Tensor:
    pos_t = torch.as_tensor(pos, dtype=dtype, device=device)
    return torch.cdist(pos_t,pos_t)


def _actual_device(device) -> torch.device:
    """Resolve a bare CUDA device to the indexed device used for allocation."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


class _SharedConstant(np.ndarray):
    """Constant shared by every copy of a calculator, so it cannot be written."""

    def __setitem__(self, key, value):
        if not self.flags.writeable:
            raise ValueError(
                "This constant is shared with every copy of this calculator and "
                "with its device cache, so it is read-only. Build a new "
                "calculator to use different bounds or wall widths.")
        super().__setitem__(key, value)


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view(_SharedConstant)
    view.setflags(write=False)
    return view


class _FBConstantsOnDevice:
    """One device-resident copy of the FB-ENM constants, shared by a cache group.

    A cache group is a calculator and the copies made from it: they share these
    arrays, so one upload serves all of them.  Calculators built independently
    have their own group, and therefore their own device copy.  Without this the
    same four N x N matrices were re-uploaded on every force call.  The inverse
    squared wall widths are stored rather than the widths, since only they are
    read.

    The device copy is dropped by ``FB_ENM.release_device_cache()``.  The host
    constants are read-only because copies share both them and this holder;
    construct a new calculator to use different bounds or wall widths.
    """

    __slots__ = ("d_min", "d_max", "delta_min", "delta_max", "_lock",
                 "_key", "_d_min_t", "_d_max_t", "_inv_dmin2_t", "_inv_dmax2_t")

    def __init__(self, d_min, d_max, delta_min, delta_max):
        self.d_min = d_min
        self.d_max = d_max
        self.delta_min = delta_min
        self.delta_max = delta_max
        self._lock = threading.Lock()
        self._key = None
        self._d_min_t = None
        self._d_max_t = None
        self._inv_dmin2_t = None
        self._inv_dmax2_t = None

    # Device tensors and the lock are runtime state, not part of the value.
    def __getstate__(self):
        return {"d_min": self.d_min, "d_max": self.d_max,
                "delta_min": self.delta_min, "delta_max": self.delta_max}

    def __setstate__(self, state):
        self.__init__(state["d_min"], state["d_max"],
                      state["delta_min"], state["delta_max"])

    @torch.no_grad()
    def get(self, device, dtype):
        device = _actual_device(device)
        key = (device, dtype)
        with self._lock:
            if self._key != key or self._d_min_t is None:
                # Drop a previous key before allocating its replacement so a
                # dtype/device switch cannot transiently retain two 4 x N^2 sets.
                self._key = None
                self._d_min_t = None
                self._d_max_t = None
                self._inv_dmin2_t = None
                self._inv_dmax2_t = None
                d_min_t = torch.as_tensor(self.d_min, dtype=dtype, device=device)
                d_max_t = torch.as_tensor(self.d_max, dtype=dtype, device=device)
                # ``torch.tensor`` gives these writable storage even on CPU;
                # they become the inverse-width cache in place.  On CUDA this
                # is still one host-to-device transfer per source matrix.
                inv_dmin2_t = torch.tensor(self.delta_min, dtype=dtype, device=device)
                inv_dmax2_t = torch.tensor(self.delta_max, dtype=dtype, device=device)
                inv_dmin2_t.square_().reciprocal_()
                inv_dmax2_t.square_().reciprocal_()
                self._d_min_t = d_min_t
                self._d_max_t = d_max_t
                self._inv_dmin2_t = inv_dmin2_t
                self._inv_dmax2_t = inv_dmax2_t
                self._key = key
            return (self._d_min_t, self._d_max_t,
                    self._inv_dmin2_t, self._inv_dmax2_t)

    def release(self):
        with self._lock:
            self._key = None
            self._d_min_t = None
            self._d_max_t = None
            self._inv_dmin2_t = None
            self._inv_dmax2_t = None


@torch.no_grad()
def _adjacency_squared(J: torch.Tensor) -> torch.Tensor:
    """``(J @ J) > 0`` for a boolean adjacency, without the dense product.

    Marks the pairs sharing at least one common neighbor, i.e. the bond and angle
    pairs.  A dense ``J @ J`` costs 2 N^3 flops, which dominates the FB-ENM setup
    for large systems and is especially expensive in float64 on GPUs with a low
    float64 rate, while a chemical adjacency has a handful of neighbors per atom.
    This scatters each atom's neighbor list against itself instead,
    O(N * maxdeg^2), and reproduces the thresholded product exactly.
    """
    n = J.shape[0]
    e = torch.nonzero(J, as_tuple=False)
    out = torch.zeros_like(J)
    if e.numel() == 0:
        return out

    k, nb = e[:, 0], e[:, 1]
    deg = torch.bincount(k, minlength=n)
    maxdeg = int(deg.max())
    start = torch.cumsum(deg, 0) - deg
    pos = torch.arange(e.shape[0], device=J.device) - start[k]

    nbr = torch.zeros((n, maxdeg), dtype=torch.long, device=J.device)
    valid = torch.zeros((n, maxdeg), dtype=torch.bool, device=J.device)
    nbr[k, pos] = nb
    valid[k, pos] = True

    for a in range(maxdeg):
        for b in range(maxdeg):
            m = valid[:, a] & valid[:, b]
            out[nbr[m, a], nbr[m, b]] = True
    return out


class FB_ENM(Calculator):
    """
    ASE calculator for the Flat-bottom Elastic Network Model (FB-ENM).

    A lightweight structure-based potential for generating
    collision-free plausible paths used in combination with the
    direct MaxFlux method. Each pair interaction has a flat
    bottom defined by (d_min, d_max). This class implements the
    general pairwise model.

    This implementation follows the model introduced in:

        Koda & Saito, J. Chem. Theory Comput. 2024, 20, 7176–7187.

    See the original paper for theoretical details.

    Parameters
    ----------
    d_min, d_max : array-like, shape (N, N)
        Flat-bottom bounds for each pair distance.
    delta_min, delta_max : array-like, shape (N, N) or None
        Quadratic wall widths. If None, generated via ``delta_scale``.
    delta_scale : float
        Scale factor used when delta arrays are autogenerated.
    return_energy_mats : bool
        If True, include ``emat_rep`` and ``emat_att`` in results.
        Unlike the NumPy implementation, these matrices are omitted by default.
    device : str or torch.device, optional
        Torch device used only for the distance-matrix evaluation.
    dtype : str or torch.dtype, optional
        Torch floating-point dtype for internal calculations
        (``float32`` or ``float64``). Default: ``float64``.

    Notes
    -----
    - ``results['energy']`` and ``results['forces']`` are returned as CPU scalars/NumPy arrays.
    - When ``return_energy_mats=True``, ``results['emat_rep']`` and
      ``results['emat_att']`` are returned as Torch tensors on ``device``.

    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        d_min,
        d_max,
        delta_min: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_max: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_scale: float = 0.2,
        return_energy_mats: bool = False,
        device=None,
        dtype=None,
        _copy_from=None,
    ):
        super().__init__()

        if _copy_from is not None:
            device = _copy_from.device
            dtype = _copy_from.torch_dtype
        self.device = _resolve_torch_device(device)
        self.torch_dtype = _resolve_torch_dtype(dtype)
        self.np_dtype = np.float64
        self._return_energy_mats = bool(return_energy_mats)

        if _copy_from is not None:
            self.d_min = _copy_from.d_min
            self.d_max = _copy_from.d_max
            self.delta_min = _copy_from.delta_min
            self.delta_max = _copy_from.delta_max
            self._const = _copy_from._const
            return

        def to_np(x) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(self.np_dtype, copy=False)
            return np.asarray(x, dtype=self.np_dtype)

        d_min_np = to_np(d_min).copy()
        d_max_np = to_np(d_max).copy()

        if delta_min is not None:
            delta_min_np = to_np(delta_min).copy()
        else:
            delta_min_np = (float(delta_scale) * d_min_np).copy()

        if delta_max is not None:
            delta_max_np = to_np(delta_max).copy()
        else:
            delta_max_np = (float(delta_scale) * d_max_np).copy()

        # Diagonal is unused; set to safe values to avoid divide-by-zero
        np.fill_diagonal(d_min_np, 0.0)
        np.fill_diagonal(d_max_np, 0.0)
        np.fill_diagonal(delta_min_np, 1.0)
        np.fill_diagonal(delta_max_np, 1.0)

        # The holder keeps writable private sources so PyTorch can create CPU
        # views without copying or warning.  Users see read-only views: copies
        # share these constants, so mutation would invalidate every device copy.
        self._const = _FBConstantsOnDevice(d_min_np, d_max_np, delta_min_np, delta_max_np)
        self.d_min = _readonly_view(d_min_np)
        self.d_max = _readonly_view(d_max_np)
        self.delta_min = _readonly_view(delta_min_np)
        self.delta_max = _readonly_view(delta_max_np)

    def __setstate__(self, state):
        self.__dict__.update(state)
        # NumPy pickle does not preserve a view's writeable flag.  Reconnect the
        # public views to the holder's canonical arrays and restore immutability.
        self.d_min = _readonly_view(self._const.d_min)
        self.d_max = _readonly_view(self._const.d_max)
        self.delta_min = _readonly_view(self._const.delta_min)
        self.delta_max = _readonly_view(self._const.delta_max)

    def copy(self):
        """
        Return a copy of this FB-ENM calculator.

        Returns
        -------
        FB_ENM
            A new calculator with the same flat-bottom bounds and wall-width
            parameters.  Results and atoms stay per instance; the four read-only
            constant matrices and the device copy of them are shared with this
            instance, so releasing the device cache through either one releases
            both.

        """
        return FB_ENM(
            self.d_min,
            self.d_max,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            return_energy_mats=self._return_energy_mats,
            device=self.device,
            dtype=self.torch_dtype,
            _copy_from=self,
        )

    def release_device_cache(self, empty_cache: bool = False):
        """Drop the device copy of the constants.

        Always releases; ``empty_cache`` additionally returns the freed blocks
        from PyTorch's caching allocator to the driver, which is a separate
        concern.
        """
        if self._const is not None:
            self._const.release()
        if empty_cache and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _dmf_device_cache_key(self):
        """Return the CUDA cache group used to coordinate path evaluation."""
        return self._const if self.device.type == "cuda" else None

    @torch.no_grad()
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)

        # 1) Distance matrix (Torch on self.device)
        pos = atoms.get_positions()
        pos_t = torch.as_tensor(pos, dtype=self.torch_dtype, device=self.device)
        d = torch.cdist(pos_t, pos_t)

        d_min_t, d_max_t, inv_dmin2, inv_dmax2 = self._const.get(
            self.device, self.torch_dtype)

        # 2) Flat-bottom penalties (Torch on device)
        d_rep = torch.sub(d, d_min_t)
        d_rep.clamp_(max=0.0)

        # Build the repulsive energy term before allocating the attractive
        # distance and energy matrices.  The elementwise multiply is in place,
        # so no third N x N expression temporary is created.
        e_rep = torch.mul(d_rep, d_rep)
        e_rep.mul_(inv_dmin2)

        d_att = torch.sub(d, d_max_t)
        d_att.clamp_(min=0.0)
        e_att = torch.mul(d_att, d_att)
        e_att.mul_(inv_dmax2)

        if self._return_energy_mats:
            # symmetric matrix -> half to count each pair once
            energy = 0.5 * torch.sum(e_rep + e_att)
        else:
            # Preserve the original elementwise-add-then-reduce order while
            # reusing the repulsive energy buffer.
            e_rep.add_(e_att)
            energy = 0.5 * torch.sum(e_rep)
            del e_rep, e_att

        # dE/dd  (up to a sign handled in force assembly)
        d_rep.mul_(inv_dmin2)
        d_att.mul_(inv_dmax2)
        d_rep.add_(d_att).mul_(2.0)
        f1 = d_rep
        del d_att

        # 3) Forces (Torch on device)
        # Avoid divide-by-zero on diagonal; diagonal term is zero anyway.
        d.fill_diagonal_(1.0)
        f1.div_(d)
        # Ensure exact zeros on diagonal
        f1.fill_diagonal_(0.0)

        # Vectorized assembly:
        # F_i = sum_j (f1_ij * (r_j - r_i))
        s = torch.sum(f1, dim=1)  # (N,)
        forces = f1 @ pos_t
        forces.sub_(pos_t * s[:, None])  # (N,3)

        self.results = {
            'energy': float(energy.item()),
            'forces': forces.cpu().numpy(),
        }
        if self._return_energy_mats:
            # Keep energy matrices on device to avoid host transfer
            self.results['emat_rep'] = e_rep
            self.results['emat_att'] = e_att
        elif self.device.type == "cuda":
            # Drop large temporaries eagerly for large-N systems.
            del d_min_t, d_max_t, inv_dmin2, inv_dmax2
            del f1, s, forces, d, pos_t, energy



class FB_ENM_Bonds(FB_ENM):
    """
    ASE calculator for bond-aware FB-ENM.

    Builds an FB-ENM automatically from multiple reference images.
    Bond lengths and bond angles preserved across all images are
    strongly constrained, while other distances receive weak bounds.

    Also supports optional planar constraints for π-like fragments.
    For details on how these constraints are defined, refer to:

        Koda & Saito, J. Chem. Theory Comput. 2024, 20, 7176–7187.

    Parameters
    ----------
    images : list of ase.Atoms
        Reference structures.
        Here, natoms = len(images[0]).
    addA : boolean ndarray of shape (natoms, natoms), optional
        Manually add strong constraints. Pairs marked True are added.
        Default: None.
    delA : boolean ndarray of shape (natoms, natoms), optional
        Manually remove strong constraints. Pairs marked True are removed.
        If both `addA` and `delA` are provided, `delA` takes precedence.
        Default: None.
    delta_scale : float, optional
        Scale for delta_min/max in the parent FB_ENM.
        Default: 0.2.
    bond_scale : float, optional
        Threshold for bond detection
        (bond is detected when d_ij < bond_scale × (r_cov[i] + r_cov[j])).
        Default: 1.25.
    fix_planes : bool, optional
        Whether to add additional constraints to preserve planar groups.
        Default: True.
    d_min_overwrite : float or ndarray of shape (natoms, natoms), optional
        Values used to overwrite selected entries of d_min.
        Only pairs where `A_overwrite` is True are replaced.
        Default: None.
    d_max_overwrite : float or ndarray of shape (natoms, natoms), optional
        Values used to overwrite selected entries of d_max.
        Only pairs where `A_overwrite` is True are replaced.
        Default: None.
    A_overwrite : boolean ndarray of shape (natoms, natoms), optional
        Boolean mask specifying which atom pairs should be overwritten.
        For any (i, j) where `A_overwrite[i, j] == True`, the following
        assignments are applied::

            d_min[A_overwrite] = d_min_overwrite[A_overwrite]
            d_max[A_overwrite] = d_max_overwrite[A_overwrite]

        Must be supplied when using either overwrite option.
        Default: None.
    device : str or torch.device, optional
        Torch device for internal tensors. If None, auto-select.
    dtype : str or torch.dtype, optional
        Torch floating-point dtype for internal calculations
        (``float32`` or ``float64``). Default: ``float64``.

    Notes
    -----
    - Weak constraints follow van der Waals and system-size bounds.

    """


    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        images: list,
        addA: Optional[np.ndarray] = None,
        delA: Optional[np.ndarray] = None,
        delta_scale: float = 0.2,
        bond_scale: float = 1.25,
        fix_planes: bool = True,
        d_min_overwrite: Optional[np.ndarray] = None,
        d_max_overwrite: Optional[np.ndarray] = None,
        A_overwrite: Optional[np.ndarray] = None,
        device=None,
        dtype=None,
    ):
        self.device = _resolve_torch_device(device)
        self.torch_dtype = _resolve_torch_dtype(dtype)
        self.np_dtype = np.float64
        _dev = self.device
        _tdt = self.torch_dtype

        numbers = images[0].arrays['numbers']
        r_cov_atom = torch.as_tensor(
            covalent_radii[numbers], dtype=_tdt, device=_dev
        )
        r_vdw_atom = torch.as_tensor(
            vdw_radii[numbers], dtype=_tdt, device=_dev
        )
        r_cov = r_cov_atom[:, None] + r_cov_atom[None, :]
        r_cov.mul_(float(bond_scale))
        r_vdw = r_vdw_atom[:, None] + r_vdw_atom[None, :]

        natoms = len(images[0])

        if fix_planes:
            addA_p = torch.zeros([natoms,natoms],dtype=torch.bool,device=_dev)
            planes = _get_planes(images, bond_scale=bond_scale, device=self.device, dtype=self.torch_dtype)
            for p in planes:
                idx = torch.as_tensor(p,device=_dev)
                addA_p[idx.unsqueeze(1),idx] = True
        else:
            addA_p = None

        addA_mask = torch.as_tensor(addA,dtype=torch.bool,device=_dev) if addA is not None else None
        delA_mask = torch.as_tensor(delA,dtype=torch.bool,device=_dev) if delA is not None else None

        d_min = torch.full([natoms,natoms],torch.inf,dtype=_tdt,device=_dev)
        d_max = torch.zeros([natoms,natoms],dtype=_tdt,device=_dev)

        with torch.no_grad():
            for image in images:
                d = _calc_dist_mat(image.get_positions(), device=self.device, dtype=self.torch_dtype)
                J = d < r_cov
                A = _adjacency_squared(J)
                del J

                if fix_planes:
                    A.logical_or_(addA_p)
                if addA_mask is not None:
                    A.logical_or_(addA_mask)
                if delA_mask is not None:
                    A.logical_and_(~delA_mask)

                cand_min = torch.where(A,d,torch.minimum(d,r_vdw))
                cand_max = torch.where(A,d,2.0*torch.amax(d))
                del d, A
                torch.minimum(d_min, cand_min, out=d_min)
                torch.maximum(d_max, cand_max, out=d_max)
                del cand_min, cand_max

            if (d_min_overwrite is not None) and (A_overwrite is not None):
                mask = torch.as_tensor(A_overwrite,dtype=torch.bool,device=_dev)
                d_min = torch.where(mask,torch.as_tensor(d_min_overwrite,dtype=_tdt,device=_dev),d_min)
            if (d_max_overwrite is not None) and (A_overwrite is not None):
                mask = torch.as_tensor(A_overwrite,dtype=torch.bool,device=_dev)
                d_max = torch.where(mask,torch.as_tensor(d_max_overwrite,dtype=_tdt,device=_dev),d_max)

        del numbers, r_cov_atom, r_vdw_atom, r_cov, r_vdw, natoms, addA_p, addA_mask, delA_mask
        super().__init__(d_min, d_max, delta_scale=delta_scale, device=device, dtype=self.torch_dtype)
        del d_min, d_max


class CFB_ENM(Calculator):
    """
    Correlated Flat-Bottom ENM (CFB-ENM).

    Extends FB-ENM by adding correlation terms between
    bond-breaking and bond-forming atom pairs. This enforces
    coordinated structural changes that FB-ENM cannot capture alone.

    The correlation model and selection of correlated quartets follow
    the formulation described in the companion paper:

        Koda & Saito, J. Chem. Theory Comput. 21, 3513-3522 (2025).

    See the original paper for the mathematical definitions.

    Parameters
    ----------
    images : list of ase.Atoms
        Reference structures.
        Here, natoms = len(images[0]).
    bond_scale : float
        Threshold to detect bonds from distances. Default: 1.25.
    d_corr0, d_corr1, d_corr2 : ndarray of shape (natoms, natoms) or None, optional
        Flat-bottom thresholds for the correlated pair-pair potential.
        See the original paper for their definitions.
        If None, they are generated by scaling the largest bond length.
        Default: None.
    corr0_scale : float
        Scaling factors used when d_corr arrays are not provided.
        See the original paper for their definitions.
        Default: 1.10.
    corr1_scale : float
        Scaling factors used when d_corr arrays are not provided.
        See the original paper for their definitions.
        Default: 1.50.
    corr2_scale : float
        Scaling factors used when d_corr arrays are not provided.
        See the original paper for their definitions.
        Default: 1.60.
    eps : float
        Small value added for numerical stability in the correlation term.
        See the original paper for their definitions.
        Default: 0.05.
    pivotal : bool
        If True, restrict correlation to pivot-based patterns.
        See the original paper for the definition of "pivot".
        Default: True.
    single : bool
        If True, require exactly one breaking and one forming bond at pivot.
        See the original paper for the definition of "single pivot".
        Default: True.
    remove_fourmembered : bool
        Exclude quasi-four-membered-ring patterns that lead to artifacts.
        See the original paper for the definition of "quasi-four-membered ring".
        Default: True.
    device : str or torch.device, optional
        Torch device for internal tensors. If None, auto-select.
    dtype : str or torch.dtype, optional
        Torch floating-point dtype for internal calculations
        (``float32`` or ``float64``). Default: ``float64``.

    Notes
    -----
    - CFB-ENM adds only correlation energy; FB-ENM must be used
      together if flat-bottom bond/angle constraints are required.
      That is, use CFB_ENM alongside FB_ENM_Bonds in a SumCalculator as::

            from ase.calculators.mixing import SumCalculator
            calc = SumCalculator([
                     FB_ENM_Bonds(...),
                     CFB_ENM(...)])
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        images=None,
        d_bond: Optional[np.ndarray] = None,
        bond_scale: float = 1.25,
        d_corr0: Optional[np.ndarray] = None,
        corr0_scale: float = 1.10,
        d_corr1: Optional[np.ndarray] = None,
        corr1_scale: float = 1.50,
        d_corr2: Optional[np.ndarray] = None,
        corr2_scale: float = 1.60,
        eps: float = 0.05,
        quartets: Optional[list] = None,
        pivotal: bool = True,
        single: bool = True,
        remove_fourmembered: bool = True,
        device=None,
        dtype=None,
        _copy_from=None,
    ):

        Calculator.__init__(self)
        if _copy_from is not None:
            device = _copy_from.device
            dtype = _copy_from.torch_dtype
        self.device = _resolve_torch_device(device)
        self.torch_dtype = _resolve_torch_dtype(dtype)
        self.np_dtype = np.float64
        _dev = self.device
        _tdt = self.torch_dtype

        need_bond = d_bond is None
        need_quartets = quartets is None

        if _copy_from is not None:
            self.d_bond = _copy_from.d_bond
            self.d_corr0 = _copy_from.d_corr0
            self.d_corr1 = _copy_from.d_corr1
            self.d_corr2 = _copy_from.d_corr2
            self.quartets = [list(q) for q in _copy_from.quartets]
            self.eps = _copy_from.eps
            natoms = self.d_bond.shape[0]
        elif need_bond or need_quartets:
            if images is None:
                raise ValueError("images are required when d_bond or quartets are not provided.")

            numbers = images[0].arrays['numbers']
            r_cov_np = covalent_radii[numbers][:, None] + covalent_radii[numbers]
            r_cov_t = torch.as_tensor(r_cov_np, dtype=_tdt, device=_dev)
            bond_scale_t = torch.tensor(float(bond_scale), dtype=_tdt, device=_dev)

            natoms = len(images[0])
            # A running maximum rather than an (nimages, N, N) stack, since only
            # the maximum over images is used.  Likewise only the first and last
            # image's adjacency enters the quartet search, so the intermediate
            # ones are not retained.
            d_bond_max = None
            J_first = None
            J_last = None
            with torch.no_grad():
                for i, image in enumerate(images):
                    d = _calc_dist_mat(image.get_positions(), device=self.device, dtype=self.torch_dtype)
                    J = (d / r_cov_t) < bond_scale_t
                    J.fill_diagonal_(False)
                    if i == 0:
                        J_first = J
                    J_last = J
                    d_masked = torch.where(J, d, torch.zeros_like(d))
                    if d_bond_max is None:
                        d_bond_max = d_masked
                    else:
                        d_bond_max = torch.maximum(d_bond_max, d_masked)
                        del d_masked
                    del d

            if need_bond:
                self.d_bond = d_bond_max.cpu().numpy()
            else:
                self.d_bond = np.asarray(d_bond, dtype=self.np_dtype).copy()
            del d_bond_max

            if need_quartets:
                J_only_r = J_first & (~J_last)
                J_only_p = J_last & (~J_first)
                J_both = J_first & J_last
                self.quartets = self._get_quartets(
                    J_only_r,
                    J_only_p,
                    J_both,
                    pivotal=pivotal,
                    single=single,
                    remove_fourmembered=remove_fourmembered,
                )
            else:
                self.quartets = [list(map(int, q)) for q in quartets]
        else:
            self.d_bond = np.asarray(d_bond, dtype=self.np_dtype).copy()
            self.quartets = [list(map(int, q)) for q in quartets]
            natoms = self.d_bond.shape[0]

        if _copy_from is None:
            if d_corr0 is not None:
                self.d_corr0 = np.asarray(d_corr0, dtype=self.np_dtype).copy()
            else:
                self.d_corr0 = corr0_scale * self.d_bond

            if d_corr1 is not None:
                self.d_corr1 = np.asarray(d_corr1, dtype=self.np_dtype).copy()
            else:
                self.d_corr1 = corr1_scale * self.d_bond

            if d_corr2 is not None:
                self.d_corr2 = np.asarray(d_corr2, dtype=self.np_dtype).copy()
            else:
                self.d_corr2 = corr2_scale * self.d_bond

            self.eps = eps

            # Shared constants arrive with this already applied, and the mask is
            # an N x N boolean array.
            I = np.identity(natoms, dtype="bool")
            self.d_bond[I] = 0.0
            self.d_corr0[I] = 0.0
            self.d_corr1[I] = 0.0
            self.d_corr2[I] = 0.0

        if self.quartets:
            q_np = np.asarray(self.quartets, dtype=np.int64)
            q_i_np = q_np[:, 0]
            q_j_np = q_np[:, 1]
            q_k_np = q_np[:, 2]
            q_l_np = q_np[:, 3]

            d00_ij_np = self.d_corr0[q_i_np, q_j_np]
            d00_kl_np = self.d_corr0[q_k_np, q_l_np]
            d10_ij_np = self.d_corr1[q_i_np, q_j_np] - d00_ij_np
            d10_kl_np = self.d_corr1[q_k_np, q_l_np] - d00_kl_np
            d20_ij_np = self.d_corr2[q_i_np, q_j_np] - d00_ij_np
            d20_kl_np = self.d_corr2[q_k_np, q_l_np] - d00_kl_np

            self.quartets_t = torch.as_tensor(q_np,dtype=torch.long,device=_dev)
            self._q_i_t = torch.as_tensor(q_i_np, dtype=torch.long, device=_dev)
            self._q_j_t = torch.as_tensor(q_j_np, dtype=torch.long, device=_dev)
            self._q_k_t = torch.as_tensor(q_k_np, dtype=torch.long, device=_dev)
            self._q_l_t = torch.as_tensor(q_l_np, dtype=torch.long, device=_dev)
            self._d00_ij_t = torch.as_tensor(d00_ij_np, dtype=_tdt, device=_dev)
            self._d00_kl_t = torch.as_tensor(d00_kl_np, dtype=_tdt, device=_dev)
            self._d10_ij_t = torch.as_tensor(d10_ij_np, dtype=_tdt, device=_dev)
            self._d10_kl_t = torch.as_tensor(d10_kl_np, dtype=_tdt, device=_dev)
            self._d20_ij_t = torch.as_tensor(d20_ij_np, dtype=_tdt, device=_dev)
            self._d20_kl_t = torch.as_tensor(d20_kl_np, dtype=_tdt, device=_dev)
            self._dnm_t = self._d20_ij_t * self._d20_kl_t - self._d10_ij_t * self._d10_kl_t
        else:
            self.quartets_t = torch.zeros((0,4),dtype=torch.long,device=_dev)
            self._q_i_t = torch.zeros((0,), dtype=torch.long, device=_dev)
            self._q_j_t = torch.zeros((0,), dtype=torch.long, device=_dev)
            self._q_k_t = torch.zeros((0,), dtype=torch.long, device=_dev)
            self._q_l_t = torch.zeros((0,), dtype=torch.long, device=_dev)
            self._d00_ij_t = torch.zeros((0,), dtype=_tdt, device=_dev)
            self._d00_kl_t = torch.zeros((0,), dtype=_tdt, device=_dev)
            self._d10_ij_t = torch.zeros((0,), dtype=_tdt, device=_dev)
            self._d10_kl_t = torch.zeros((0,), dtype=_tdt, device=_dev)
            self._d20_ij_t = torch.zeros((0,), dtype=_tdt, device=_dev)
            self._d20_kl_t = torch.zeros((0,), dtype=_tdt, device=_dev)
            self._dnm_t = torch.zeros((0,), dtype=_tdt, device=_dev)

        self.d_bond = _readonly_view(self.d_bond)
        self.d_corr0 = _readonly_view(self.d_corr0)
        self.d_corr1 = _readonly_view(self.d_corr1)
        self.d_corr2 = _readonly_view(self.d_corr2)

    def release_device_cache(self, empty_cache: bool = False):
        """No device copy is held here; ``empty_cache`` is an allocator action only."""
        if empty_cache and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.d_bond = _readonly_view(self.d_bond)
        self.d_corr0 = _readonly_view(self.d_corr0)
        self.d_corr1 = _readonly_view(self.d_corr1)
        self.d_corr2 = _readonly_view(self.d_corr2)

    def copy(self, images=None):
        """
        Return a copy of this CFB-ENM calculator.

        Parameters
        ----------
        images : list of ase.Atoms
            The same reference images that were used to construct the original
            ``CFB_ENM`` instance.

        Returns
        -------
        CFB_ENM
            A new ``CFB_ENM`` instance with the same correlation parameters and
            quartet list.  Results and atoms stay per instance; the four
            read-only constant matrices are shared with this instance.

        """
        return type(self)(
            images=images,
            _copy_from=self,
        )

    def _get_quartets(self,J_only_r,J_only_p,J_both,
            pivotal=True,single=True,remove_fourmembered=True):

        if isinstance(J_both,torch.Tensor):
            J_both_b = J_both.to(dtype=torch.bool)
        else:
            J_both_b = torch.as_tensor(np.asarray(J_both, dtype=bool), device=self.device)
        J2 = _adjacency_squared(J_both_b).cpu().numpy()
        del J_both_b

        J_only_r = J_only_r.cpu().numpy() if isinstance(J_only_r,torch.Tensor) else np.asarray(J_only_r,bool)
        J_only_p = J_only_p.cpu().numpy() if isinstance(J_only_p,torch.Tensor) else np.asarray(J_only_p,bool)
        J_both = J_both.cpu().numpy() if isinstance(J_both,torch.Tensor) else np.asarray(J_both,bool)

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

    @torch.no_grad()
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        r = atoms.get_positions()

        natoms = len(atoms)
        if self.quartets_t.numel()==0:
            self.results = {'energy': 0.0,
                            'forces': np.zeros([natoms,3], dtype=self.np_dtype)}
            del natoms, r
            return

        pos = torch.as_tensor(r, dtype=self.torch_dtype, device=self.device)
        i = self._q_i_t
        j = self._q_j_t
        k = self._q_k_t
        l = self._q_l_t

        diff_ij = pos[i]-pos[j]
        diff_kl = pos[k]-pos[l]
        d_ij = torch.linalg.norm(diff_ij,dim=1)
        d_kl = torch.linalg.norm(diff_kl,dim=1)

        d00_ij = self._d00_ij_t
        d00_kl = self._d00_kl_t
        d10_ij = self._d10_ij_t
        d10_kl = self._d10_kl_t
        d20_ij = self._d20_ij_t
        d20_kl = self._d20_kl_t

        dd0_ij = d_ij-d00_ij
        dd0_kl = d_kl-d00_kl

        pp = dd0_ij*dd0_kl-d10_ij*d10_kl
        ok = (dd0_ij>0.0)&(dd0_kl>0.0)&(pp>0.0)
        if not torch.any(ok):
            self.results = {'energy': 0.0,
                            'forces': np.zeros([natoms,3], dtype=self.np_dtype)}
            del natoms, pos, i, j, k, l, diff_ij, diff_kl, d_ij, d_kl, d00_ij, d00_kl, d10_ij, d10_kl, d20_ij, d20_kl, dd0_ij, dd0_kl, pp, ok
            return

        i = i[ok]; j = j[ok]; k = k[ok]; l = l[ok]
        diff_ij = diff_ij[ok]; diff_kl = diff_kl[ok]
        d_ij = d_ij[ok]; d_kl = d_kl[ok]
        dd0_ij = dd0_ij[ok]; dd0_kl = dd0_kl[ok]
        d10_ij = d10_ij[ok]; d10_kl = d10_kl[ok]
        d20_ij = d20_ij[ok]; d20_kl = d20_kl[ok]
        pp = pp[ok]

        dnm = self._dnm_t[ok]
        pp = pp/dnm
        eps = float(self.eps)
        sqrt_pp2 = torch.sqrt(pp*pp+eps*eps)
        alpha = pp/sqrt_pp2
        energy = torch.sum(sqrt_pp2-eps).item()

        v1 = (dd0_kl/d_ij).unsqueeze(1)*(diff_ij/dnm.unsqueeze(1))
        v2 = (dd0_ij/d_kl).unsqueeze(1)*(diff_kl/dnm.unsqueeze(1))

        forces = torch.zeros([natoms,3], dtype=self.torch_dtype, device=self.device)
        forces.index_add_(0,i,-alpha.unsqueeze(1)*v1)
        forces.index_add_(0,j, alpha.unsqueeze(1)*v1)
        forces.index_add_(0,k,-alpha.unsqueeze(1)*v2)
        forces.index_add_(0,l, alpha.unsqueeze(1)*v2)

        self.results = {'energy': float(energy),
                        'forces': forces.cpu().numpy()}
        del natoms, pos, i, j, k, l, diff_ij, diff_kl, d_ij, d_kl, d00_ij, d00_kl, d10_ij, d10_kl, d20_ij, d20_kl, dd0_ij, dd0_kl, pp, dnm, eps, sqrt_pp2, alpha, v1, v2, forces


@torch.no_grad()
def _quartet_planarity_mask(
    pos: torch.Tensor,
    quartets_t: torch.Tensor,
    tol_rmsd: float,
    chunk_size: int = 32768,
) -> torch.Tensor:
    n = int(quartets_t.shape[0])
    if n == 0:
        return torch.zeros((0,), dtype=torch.bool, device=pos.device)

    keep = torch.zeros((n,), dtype=torch.bool, device=pos.device)
    tol = float(tol_rmsd)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        q = quartets_t[i0:i1]
        x = pos[q]  # (B, 4, 3)
        xc = x - torch.mean(x, dim=1, keepdim=True)
        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        v = vh[:, -1, :]  # (B, 3)
        d = torch.sum(xc * v.unsqueeze(1), dim=2)
        rmsd = torch.sqrt(torch.mean(d * d, dim=1))
        keep[i0:i1] = rmsd < tol
    return keep


@torch.no_grad()
def _quartet_chain_geom_masks(
    pos: torch.Tensor,
    quartets_t: torch.Tensor,
    tol_rmsd: float,
    tol_ang: float,
    chunk_size: int = 32768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(quartets_t.shape[0])
    if n == 0:
        empty = torch.zeros((0,), dtype=torch.bool, device=pos.device)
        return empty, empty, empty

    mask_plane = torch.zeros((n,), dtype=torch.bool, device=pos.device)
    mask_not_linear = torch.zeros((n,), dtype=torch.bool, device=pos.device)
    mask_cis = torch.zeros((n,), dtype=torch.bool, device=pos.device)

    eps = torch.finfo(pos.dtype).eps
    tol_rmsd_f = float(tol_rmsd)
    # 180-angle > tol_ang  <=> angle < 180-tol_ang
    cos_thr = float(np.cos(np.deg2rad(180.0 - float(tol_ang))))

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        q = quartets_t[i0:i1]
        x = pos[q]  # (B, 4, 3)

        # Planarity via best-fit plane RMSD (same definition as previous implementation).
        xc = x - torch.mean(x, dim=1, keepdim=True)
        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        v = vh[:, -1, :]
        d = torch.sum(xc * v.unsqueeze(1), dim=2)
        rmsd = torch.sqrt(torch.mean(d * d, dim=1))
        plane_ok = rmsd < tol_rmsd_f

        # Non-linearity from two internal angles (i-j-k and j-k-l).
        v01 = x[:, 0, :] - x[:, 1, :]
        v21 = x[:, 2, :] - x[:, 1, :]
        v12 = x[:, 1, :] - x[:, 2, :]
        v32 = x[:, 3, :] - x[:, 2, :]
        cos0 = torch.sum(v01 * v21, dim=1) / (torch.linalg.norm(v01, dim=1) * torch.linalg.norm(v21, dim=1) + eps)
        cos1 = torch.sum(v12 * v32, dim=1) / (torch.linalg.norm(v12, dim=1) * torch.linalg.norm(v32, dim=1) + eps)
        not_linear_ok = (cos0 > cos_thr) & (cos1 > cos_thr)

        # cis/trans from cosine of the dihedral angle.
        b1 = x[:, 1, :] - x[:, 0, :]
        b2 = x[:, 2, :] - x[:, 1, :]
        b3 = x[:, 3, :] - x[:, 2, :]
        n1 = torch.cross(b1, b2, dim=1)
        n2 = torch.cross(b2, b3, dim=1)
        cos_dih = torch.sum(n1 * n2, dim=1) / (torch.linalg.norm(n1, dim=1) * torch.linalg.norm(n2, dim=1) + eps)
        cis_ok = cos_dih >= 0.0

        mask_plane[i0:i1] = plane_ok
        mask_not_linear[i0:i1] = not_linear_ok
        mask_cis[i0:i1] = cis_ok

    return mask_plane, mask_not_linear, mask_cis


def _quartets_to_tensor(quartets: list, device: torch.device) -> torch.Tensor:
    if len(quartets) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)
    return torch.as_tensor(quartets, dtype=torch.long, device=device)


def _apply_mask_to_quartets(quartets: list, mask_t: torch.Tensor) -> list:
    if len(quartets) == 0:
        return []
    idxs = torch.nonzero(mask_t, as_tuple=False).squeeze(1).cpu().tolist()
    return [quartets[i] for i in idxs]


def _get_planes(images, bond_scale=1.25, tol_rmsd=0.05, tol_ang=10.0, *, device=None, dtype=None):

    device = _resolve_torch_device(device)
    torch_dtype = _resolve_torch_dtype(dtype)
    bond_scale_t = torch.tensor(float(bond_scale), dtype=torch_dtype, device=device)

    for iimg, atoms in enumerate(images):

        pos = torch.as_tensor(atoms.get_positions(), dtype=torch_dtype, device=device)
        cov_radii = covalent_radii[atoms.arrays['numbers']]
        r_cov = torch.as_tensor(cov_radii, dtype=torch_dtype, device=device)
        r_cov = r_cov + r_cov.unsqueeze(1)
        d = _calc_dist_mat(pos, device=device, dtype=torch_dtype)
        A = (d/r_cov)<bond_scale_t
        A.fill_diagonal_(False)
        # One transfer for the whole adjacency, then split it per row on the host:
        # a row-by-row `.cpu().tolist()` costs one device synchronization per atom.
        rows, cols = np.nonzero(A.cpu().numpy())
        bounds = np.searchsorted(rows, np.arange(len(atoms) + 1))
        nghs = [cols[bounds[i]:bounds[i + 1]].tolist() for i in range(len(atoms))]

        if iimg==0:
            path = []
            c4s = []

            def walk(i):
                if i not in path:
                    path.append(i)
                    if len(path)==4:
                        if path[0]<path[3]:
                            c4s.append(list(path))
                    else:
                        for j in nghs[i]:
                            walk(j)
                    path.pop()

            for i in range(len(atoms)):
                walk(i)

            c4s_center = []
            for i0 in range(len(atoms)):
                nngh = len(nghs[i0])
                if nngh>=3:
                    for i1 in range(nngh):
                        for i2 in range(i1+1,nngh):
                            for i3 in range(i2+1,nngh):
                                c4s_center.append([i0, nghs[i0][i1], nghs[i0][i2], nghs[i0][i3]])

            q_chain = _quartets_to_tensor(c4s, device)
            if q_chain.numel() > 0:
                conn_chain = (
                    A[q_chain[:, 0], q_chain[:, 1]]
                    & A[q_chain[:, 1], q_chain[:, 2]]
                    & A[q_chain[:, 2], q_chain[:, 3]]
                )
                plane_ok, not_linear_ok, cis_ok = _quartet_chain_geom_masks(
                    pos, q_chain, tol_rmsd, tol_ang
                )
                chain_common = conn_chain & plane_ok & not_linear_ok
                pels_cis = _apply_mask_to_quartets(c4s, chain_common & cis_ok)
                pels_trans = _apply_mask_to_quartets(c4s, chain_common & (~cis_ok))
            else:
                pels_cis = []
                pels_trans = []

            q_center = _quartets_to_tensor(c4s_center, device)
            if q_center.numel() > 0:
                conn_center = (
                    A[q_center[:, 0], q_center[:, 1]]
                    & A[q_center[:, 0], q_center[:, 2]]
                    & A[q_center[:, 0], q_center[:, 3]]
                )
                plane_center = _quartet_planarity_mask(pos, q_center, tol_rmsd)
                pels_center = _apply_mask_to_quartets(c4s_center, conn_center & plane_center)
            else:
                pels_center = []

        else:
            q_cis = _quartets_to_tensor(pels_cis, device)
            if q_cis.numel() > 0:
                conn_cis = (
                    A[q_cis[:, 0], q_cis[:, 1]]
                    & A[q_cis[:, 1], q_cis[:, 2]]
                    & A[q_cis[:, 2], q_cis[:, 3]]
                )
                plane_ok, not_linear_ok, cis_ok = _quartet_chain_geom_masks(
                    pos, q_cis, tol_rmsd, tol_ang
                )
                pels_cis = _apply_mask_to_quartets(
                    pels_cis, conn_cis & plane_ok & not_linear_ok & cis_ok
                )
            else:
                pels_cis = []

            q_trans = _quartets_to_tensor(pels_trans, device)
            if q_trans.numel() > 0:
                conn_trans = (
                    A[q_trans[:, 0], q_trans[:, 1]]
                    & A[q_trans[:, 1], q_trans[:, 2]]
                    & A[q_trans[:, 2], q_trans[:, 3]]
                )
                plane_ok, not_linear_ok, cis_ok = _quartet_chain_geom_masks(
                    pos, q_trans, tol_rmsd, tol_ang
                )
                pels_trans = _apply_mask_to_quartets(
                    pels_trans, conn_trans & plane_ok & not_linear_ok & (~cis_ok)
                )
            else:
                pels_trans = []

            q_center = _quartets_to_tensor(pels_center, device)
            if q_center.numel() > 0:
                conn_center = (
                    A[q_center[:, 0], q_center[:, 1]]
                    & A[q_center[:, 0], q_center[:, 2]]
                    & A[q_center[:, 0], q_center[:, 3]]
                )
                plane_center = _quartet_planarity_mask(pos, q_center, tol_rmsd)
                pels_center = _apply_mask_to_quartets(pels_center, conn_center & plane_center)
            else:
                pels_center = []

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
