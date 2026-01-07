from typing import Optional, Union

import numpy as np

from ase.calculators.calculator import Calculator
from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii

import torch

from . import dmf
from .dmf import _resolve_torch_device

@torch.no_grad()
def _calc_dist_mat(pos, *, device) -> torch.Tensor:
    """
    Compute pairwise distances using torch.cdist on the current device.

    Parameters
    ----------
    pos : (N, 3) array-like
        Atomic positions. NumPy or Tensor is accepted.

    Returns
    -------
    torch.Tensor, shape (N, N), dtype=torch.float64, on `device`
    """
    pos_t = torch.as_tensor(pos, dtype=torch.float64, device=device)
    return torch.cdist(pos_t, pos_t)


class FB_ENM(Calculator):
    """
    Flexible boundary ENM with repulsive/attractive quadratic penalties.

    GPU optimizations:
    - All pairwise distances and per-pair energies assembled on `device`.
    - Forces accumulated using upper-triangular indexing (no N×N×3 tensors).
    - Single transfer to CPU for ASE `results`.

    Parameters
    ----------
    d_min, d_max : Union[np.ndarray, torch.Tensor], shape (N, N)
        Target lower/upper distances. Diagonal is ignored.
    delta_min, delta_max : Union[np.ndarray, torch.Tensor], shape (N, N), optional
        Widths for repulsive/attractive quadratics. If not given, set to
        `delta_scale * d_min` / `delta_scale * d_max`. Diagonals are set to 1.
    delta_scale : float
        Factor for automatic delta construction.
    return_energy_mats : bool, default True
        If False, omit 'emat_rep'/'emat_att' from results to reduce transfers.

    Optional
    --------
    device : Optional[Union[str, torch.device]]
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        d_min,
        d_max,
        delta_min: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_max: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_scale: float = 0.2,
        return_energy_mats: bool = True,
        device=None,
    ):
        super().__init__()

        self.device = _resolve_torch_device(device)
        _dev = self.device
        _tdt = torch.float64

        self._return_energy_mats = bool(return_energy_mats)

        # Normalize inputs to torch tensors on `device`
        def to_t(x):
            return torch.as_tensor(x, dtype=_tdt, device=_dev)

        d_min_t = to_t(d_min).clone()
        d_max_t = to_t(d_max).clone()

        # Build deltas if needed
        if delta_min is None:
            delta_min_t = (float(delta_scale) * d_min_t).clone()
        else:
            delta_min_t = to_t(delta_min).clone()
        if delta_max is None:
            delta_max_t = (float(delta_scale) * d_max_t).clone()
        else:
            delta_max_t = to_t(delta_max).clone()

        # Ensure diagonals are harmless
        with torch.no_grad():
            d_min_t.fill_diagonal_(0.0)
            d_max_t.fill_diagonal_(0.0)
            delta_min_t.fill_diagonal_(1.0)
            delta_max_t.fill_diagonal_(1.0)

        self.d_min = d_min_t
        self.d_max = d_max_t
        self.delta_min = delta_min_t
        self.delta_max = delta_max_t

        # Precompute triangular indices for energy/force accumulation
        n = d_min_t.shape[0]
        iu, ju = torch.triu_indices(n, n, offset=1, device=_dev).unbind(0)
        self._iu = iu
        self._ju = ju

    @torch.no_grad()
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)

        pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=self.device)  # (N,3)
        d = torch.cdist(pos, pos)                                                   # (N,N)

        # Per-pair deviations
        d_rep = torch.minimum(torch.zeros_like(d), d - self.d_min)                  # <= 0
        d_att = torch.clamp(d - self.d_max, min=0.0)                                # >= 0

        # Common factors
        inv_dmin2 = 1.0 / (self.delta_min * self.delta_min)
        inv_dmax2 = 1.0 / (self.delta_max * self.delta_max)

        # Scalar total energy (upper triangle only)
        if self._return_energy_mats:
            e_rep = (d_rep * d_rep) * inv_dmin2
            e_att = (d_att * d_att) * inv_dmax2
            energy = (e_rep[self._iu, self._ju] + e_att[self._iu, self._ju]).sum().item()
        else:
            energy = (((d_rep * d_rep) * inv_dmin2 + (d_att * d_att) * inv_dmax2)[self._iu, self._ju]
                      .sum()
                      .item())

        # Force accumulation using i<j pairs only:
        # f1_ij = 2*(d_rep/δ_min^2 + d_att/δ_max^2)
        f1 = 2.0 * (d_rep * inv_dmin2 + d_att * inv_dmax2)                          # (N,N)
        i = self._iu
        j = self._ju
        rij = pos[i] - pos[j]                                                       # (M,3)
        dij = torch.linalg.norm(rij, dim=1)                                         # (M,)
        uij = rij / dij.unsqueeze(1)                                                # (M,3)
        coeff = f1[i, j].unsqueeze(1)                                               # (M,1)
        contrib = coeff * uij                                                       # (M,3)

        forces = torch.zeros_like(pos)                                              # (N,3)
        forces.index_add_(0, i, -contrib)                                           # F_i += -coeff*uij
        forces.index_add_(0, j,  contrib)                                           # F_j += +coeff*uij

        # Pack results: ASE expects NumPy on CPU
        self.results = {
            "energy": float(energy),
            "forces": forces.cpu().numpy().astype(np.float64),
        }
        if self._return_energy_mats:
            # Diagnostics (optional; large). Leave identical to original when requested.
            e_rep = (d_rep * d_rep) * inv_dmin2
            e_att = (d_att * d_att) * inv_dmax2
            self.results["emat_rep"] = e_rep.cpu().numpy().astype(np.float64)
            self.results["emat_att"] = e_att.cpu().numpy().astype(np.float64)


# FB_ENM_Bonds
class FB_ENM_Bonds(FB_ENM):
    """
    ENM with bond/plane-aware distance envelopes aggregated over images.

    Builds per-pair lower/upper distance bounds on the GPU via incremental
    reduction per image (memory efficient), then initializes the parent
    `FB_ENM` with torch tensors (no CPU round-trips).

    Parameters
    ----------
    images : list[ase.Atoms]
    addA, delA : np.ndarray[bool], optional
        Manual additions/removals to the adjacency mask.
    delta_scale : float
    bond_scale : float
        Bond detection threshold relative to covalent radii sum.
    fix_planes : bool
        If True, co-planar atom sets are kept connected.
    d_min_overwrite, d_max_overwrite : np.ndarray, optional
        Overwrite subsets with `A_overwrite` mask.
    A_overwrite : np.ndarray[bool], optional
    two_hop_mode : {"dense","sparse"}, default "dense"
        Build 2-hop connectivity A=(J @ J)>0 densely or via sparse CSR mm.

    Optional
    --------
    device : Optional[Union[str, torch.device]]
    """

    implemented_properties = ["energy", "forces"]

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
        two_hop_mode: str = "dense",
        device=None,
    ):
        self.device = _resolve_torch_device(device)
        _dev = self.device
        _tdt = torch.float64

        numbers = images[0].arrays["numbers"]
        r_cov_np = covalent_radii[numbers][:, None] + covalent_radii[numbers]
        r_vdw_np = vdw_radii[numbers][:, None] + vdw_radii[numbers]

        r_cov = torch.as_tensor(r_cov_np, dtype=_tdt, device=_dev)
        r_vdw = torch.as_tensor(r_vdw_np, dtype=_tdt, device=_dev)
        bond_scale_t = torch.tensor(float(bond_scale), dtype=_tdt, device=_dev)

        nat = len(images[0])

        # Optional “plane” constraints → adjacency mask
        addA_plane = torch.zeros((nat, nat), dtype=torch.bool, device=_dev)
        if fix_planes:
            for plane in _get_planes(images, bond_scale=bond_scale, device=self.device):
                idx = torch.as_tensor(plane, device=_dev)
                addA_plane[idx.unsqueeze(1), idx] = True

        addA_mask = None if addA is None else torch.as_tensor(addA, dtype=torch.bool, device=_dev)
        delA_mask = None if delA is None else torch.as_tensor(delA, dtype=torch.bool, device=_dev)

        # Incremental reduction across images to avoid (nimg, nat, nat) tensors
        d_min = torch.full((nat, nat), torch.inf, dtype=_tdt, device=_dev)
        d_max = torch.zeros((nat, nat), dtype=_tdt, device=_dev)

        with torch.no_grad():
            for atoms in images:
                d = _calc_dist_mat(atoms.get_positions(), device=self.device)  # (nat, nat), on device

                J = (d / r_cov) < bond_scale_t

                if two_hop_mode.lower() == "sparse":
                    Jf = J.to(_tdt)
                    A2 = torch.sparse.mm(Jf.to_sparse_csr(), Jf.to_sparse_csr()).to_dense()
                    A = A2 > 0
                else:
                    A = (J.to(_tdt) @ J.to(_tdt)) > 0

                if fix_planes:
                    A |= addA_plane
                if addA_mask is not None:
                    A |= addA_mask
                if delA_mask is not None:
                    A &= ~delA_mask

                cand_min = torch.where(A, d, torch.minimum(d, r_vdw))
                cand_max = torch.where(A, d, 2.0 * torch.amax(d))

                d_min = torch.minimum(d_min, cand_min)
                d_max = torch.maximum(d_max, cand_max)

            # Optional overwrite in masked regions
            if (d_min_overwrite is not None) and (A_overwrite is not None):
                mask = torch.as_tensor(A_overwrite, dtype=torch.bool, device=_dev)
                d_min = torch.where(mask, torch.as_tensor(d_min_overwrite, dtype=_tdt, device=_dev), d_min)
            if (d_max_overwrite is not None) and (A_overwrite is not None):
                mask = torch.as_tensor(A_overwrite, dtype=torch.bool, device=_dev)
                d_max = torch.where(mask, torch.as_tensor(d_max_overwrite, dtype=_tdt, device=_dev), d_max)

        # Initialize parent with torch tensors (no round-trip to NumPy here)
        super().__init__(d_min, d_max, delta_scale=delta_scale, device=device)


# CFB_ENM — correlated FB_ENM
class CFB_ENM(Calculator):
    """
    Correlated flexible-boundary ENM.

    GPU optimizations:
    - Distances, per-quartet terms, and force accumulation are vectorized on `device`.
    - Single CPU transfer at the end for ASE `results`.

    Parameters
    ----------
    images : list[ase.Atoms]
    bond_scale : float
    d_corr{0,1,2} : np.ndarray, optional
        Per-pair correlation distances. If None, built from observed bond
        distances scaled by corr{0,1,2}_scale.
    eps : float
        Smoothness parameter in sqrt(pp^2 + eps^2).
    pivotal, single, remove_fourmembered : bool
        Options for quartet enumeration.
    two_hop_mode : {"dense","sparse"}, default "dense"
        How to build two-hop connectivity inside quartet enumeration.

    Optional
    --------
    device : Optional[Union[str, torch.device]]
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self, images,
        bond_scale=1.25,
        d_corr0=None, corr0_scale=1.10,
        d_corr1=None, corr1_scale=1.50,
        d_corr2=None, corr2_scale=1.60,
        eps=0.05,
        pivotal=True,
        single=True,
        remove_fourmembered=True,
        two_hop_mode: str = "dense",
        device=None,
    ):
        super().__init__()

        self.device = _resolve_torch_device(device)
        _dev = self.device
        _tdt = torch.float64

        self.two_hop_mode = two_hop_mode

        numbers = images[0].arrays["numbers"]
        r_cov_np = covalent_radii[numbers][:, None] + covalent_radii[numbers]
        self.r_cov_t = torch.as_tensor(r_cov_np, dtype=_tdt, device=_dev)
        self.bond_scale_t = torch.tensor(float(bond_scale), dtype=_tdt, device=_dev)

        nimg, nat = len(images), len(images[0])
        d_bonds = torch.zeros((nimg, nat, nat), dtype=_tdt, device=_dev)

        Js = []
        with torch.no_grad():
            for idx, img in enumerate(images):
                d = _calc_dist_mat(img.get_positions(), device=self.device)
                J = (d / self.r_cov_t) < self.bond_scale_t
                J.fill_diagonal_(False)
                Js.append(J)
                d_bonds[idx] = torch.where(J, d, torch.zeros_like(d))

        d_bond_np = torch.max(d_bonds, dim=0).values.cpu().numpy()

        # Edges
        J_only_r = (Js[0] & (~Js[-1]))
        J_only_p = (Js[-1] & (~Js[0]))
        J_both = (Js[0] & Js[-1])

        quartets = self.get_quartets(
            J_only_r, J_only_p, J_both,
            pivotal=pivotal, single=single,
            remove_fourmembered=remove_fourmembered,
        )
        self.quartets_t = (
            torch.as_tensor(quartets, dtype=torch.long, device=_dev)
            if quartets else
            torch.empty((0, 4), dtype=torch.long, device=_dev)
        )

        # Build correlation distances (NumPy inputs -> torch tensors)
        if d_corr0 is None:
            d_corr0 = corr0_scale * d_bond_np
        if d_corr1 is None:
            d_corr1 = corr1_scale * d_bond_np
        if d_corr2 is None:
            d_corr2 = corr2_scale * d_bond_np

        I = np.identity(nat, dtype=bool)
        for arr in (d_bond_np, d_corr0, d_corr1, d_corr2):
            arr[I] = 0.0

        self.d_corr0_t = torch.as_tensor(d_corr0, dtype=_tdt, device=_dev)
        self.d_corr1_t = torch.as_tensor(d_corr1, dtype=_tdt, device=_dev)
        self.d_corr2_t = torch.as_tensor(d_corr2, dtype=_tdt, device=_dev)
        self.eps = float(eps)

    def get_quartets(
        self, J_only_r, J_only_p, J_both,
        pivotal=True, single=True, remove_fourmembered=True
    ):
        # Compute two-hop connectivity of shared-bond graph
        if isinstance(J_both, torch.Tensor):
            if self.two_hop_mode.lower() == "sparse":
                Jf = J_both.to(torch.float64)
                J2_t = torch.sparse.mm(Jf.to_sparse_csr(), Jf.to_sparse_csr()).to_dense() > 0
            else:
                J2_t = (J_both.to(torch.float64) @ J_both.to(torch.float64)) > 0
            J2_np = J2_t.cpu().numpy()
        else:
            J_both_t = torch.as_tensor(J_both, dtype=torch.bool, device=self.device)
            if self.two_hop_mode.lower() == "sparse":
                Jf = J_both_t.to(torch.float64)
                J2_np = (torch.sparse.mm(Jf.to_sparse_csr(), Jf.to_sparse_csr()).to_dense() > 0).cpu().numpy()
            else:
                J2_np = ((J_both_t.to(torch.float64) @ J_both_t.to(torch.float64)) > 0).cpu().numpy()

        J_only_r_np = J_only_r.cpu().numpy() if isinstance(J_only_r, torch.Tensor) else np.asarray(J_only_r, bool)
        J_only_p_np = J_only_p.cpu().numpy() if isinstance(J_only_p, torch.Tensor) else np.asarray(J_only_p, bool)
        J_both_np = J_both.cpu().numpy() if isinstance(J_both, torch.Tensor) else np.asarray(J_both, bool)

        if pivotal:
            quartets = []
            if single:
                pivots = np.where(
                    (np.sum(J_only_r_np, axis=1) == 1) &
                    (np.sum(J_only_p_np, axis=1) == 1)
                )[0]
            else:
                pivots = np.where(np.any(J_only_r_np, axis=1) & np.any(J_only_p_np, axis=1))[0]

            for i in pivots:
                only_r = np.where(J_only_r_np[i])[0]
                only_p = np.where(J_only_p_np[i])[0]
                for j in only_r:
                    for k in only_p:
                        if not (remove_fourmembered and J2_np[j, k]):
                            quartets.append([i, j, i, k])
        else:
            pairs_only_r = []
            pairs_only_p = []
            n = J_only_r_np.shape[0]
            for i in range(n):
                for j in range(i):
                    if J_only_r_np[i, j]:
                        pairs_only_r.append([i, j])
                    if J_only_p_np[i, j]:
                        pairs_only_p.append([i, j])

            quartets = []
            for pr in pairs_only_r:
                for pp in pairs_only_p:
                    q = pr + pp  # [i, j, k, l]
                    if remove_fourmembered:
                        uniq_idxs = [q[idx] for idx in range(4) if q.count(q[idx]) == 1]
                        if len(uniq_idxs) == 4:
                            is_four = ((J_both_np[q[0], q[2]] and J_both_np[q[1], q[3]])
                                       or (J_both_np[q[0], q[3]] and J_both_np[q[1], q[2]]))
                        else:
                            is_four = J2_np[uniq_idxs[0], uniq_idxs[1]]
                        if is_four:
                            continue
                    quartets.append(q)

        return quartets

    @torch.no_grad()
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)

        nat = len(atoms)
        if self.quartets_t.numel() == 0:
            self.results = {"energy": 0.0, "forces": np.zeros((nat, 3))}
            return

        pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=self.device)

        q = self.quartets_t  # (M,4)
        i = q[:, 0]; j = q[:, 1]; k = q[:, 2]; l = q[:, 3]

        # Pair vectors/distances
        diff_ij = pos[i] - pos[j]                                   # (M,3)
        diff_kl = pos[k] - pos[l]                                   # (M,3)
        d_ij = torch.linalg.norm(diff_ij, dim=1)                    # (M,)
        d_kl = torch.linalg.norm(diff_kl, dim=1)                    # (M,)

        # Correlation distances
        d00_ij = self.d_corr0_t[i, j]; d00_kl = self.d_corr0_t[k, l]
        d10_ij = self.d_corr1_t[i, j] - d00_ij
        d10_kl = self.d_corr1_t[k, l] - d00_kl
        d20_ij = self.d_corr2_t[i, j] - d00_ij
        d20_kl = self.d_corr2_t[k, l] - d00_kl

        dd0_ij = d_ij - d00_ij
        dd0_kl = d_kl - d00_kl

        pp = dd0_ij * dd0_kl - d10_ij * d10_kl
        ok = (dd0_ij > 0.0) & (dd0_kl > 0.0) & (pp > 0.0)
        if not torch.any(ok):
            self.results = {"energy": 0.0, "forces": np.zeros((nat, 3))}
            return

        # Effective subset
        i = i[ok]; j = j[ok]; k = k[ok]; l = l[ok]
        diff_ij = diff_ij[ok]; diff_kl = diff_kl[ok]
        d_ij = d_ij[ok]; d_kl = d_kl[ok]
        dd0_ij = dd0_ij[ok]; dd0_kl = dd0_kl[ok]
        d10_ij = d10_ij[ok]; d10_kl = d10_kl[ok]
        d20_ij = d20_ij[ok]; d20_kl = d20_kl[ok]
        pp = pp[ok]

        denom = d20_ij * d20_kl - d10_ij * d10_kl
        pp_div = pp / denom
        eps_val = float(self.eps)
        sqrt_pp2 = torch.sqrt(pp_div * pp_div + eps_val * eps_val)
        alpha = pp_div / sqrt_pp2
        energy = torch.sum(sqrt_pp2 - eps_val).item()

        v1 = (dd0_kl / d_ij).unsqueeze(1) * (diff_ij / denom.unsqueeze(1))
        v2 = (dd0_ij / d_kl).unsqueeze(1) * (diff_kl / denom.unsqueeze(1))

        forces = torch.zeros((nat, 3), dtype=torch.float64, device=self.device)
        forces.index_add_(0, i, -alpha.unsqueeze(1) * v1)
        forces.index_add_(0, j,  alpha.unsqueeze(1) * v1)
        forces.index_add_(0, k, -alpha.unsqueeze(1) * v2)
        forces.index_add_(0, l,  alpha.unsqueeze(1) * v2)

        self.results = {
            "energy": float(energy),
            "forces": forces.cpu().numpy().astype(np.float64),
        }


# Planarity detection
def _get_planes(images, bond_scale=1.25, tol_rmsd=0.05, tol_ang=10.0, *, device=None):
    """
    Find atom sets that remain (near-)planar across given images.
    Returned as a list of sorted index lists. Used to stabilize ENM masks.
    """

    def rmsd(pos_t: torch.Tensor, c4):
        x = pos_t[c4]
        cent = torch.mean(x, dim=0, keepdim=True)
        u, s, vh = torch.linalg.svd(x - cent, full_matrices=False)
        v = vh[-1]
        d = torch.matmul(x - cent, v)
        return torch.sqrt(torch.mean(d * d)).item()

    def is_not_linear(atoms, c4):
        return (
            180.0 - atoms.get_angle(*c4[:3]) > tol_ang
            and 180.0 - atoms.get_angle(*c4[1:4]) > tol_ang
        )

    def is_cis(atoms, c4):
        return np.cos(np.deg2rad(atoms.get_dihedral(*c4))) >= 0.0

    def is_trans(atoms, c4):
        return np.cos(np.deg2rad(atoms.get_dihedral(*c4))) < 0.0

    def is_connected(nghs, c4):
        return (
            c4[0] in nghs[c4[1]]
            and c4[1] in nghs[c4[2]]
            and c4[2] in nghs[c4[3]]
        )

    def is_connected_center(nghs, c4):
        return (
            c4[0] in nghs[c4[1]]
            and c4[0] in nghs[c4[2]]
            and c4[0] in nghs[c4[3]]
        )

    device = _resolve_torch_device(device)
    bond_scale_t = torch.tensor(float(bond_scale), dtype=torch.float64, device=device)

    for i_img, atoms in enumerate(images):
        pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float64, device=device)
        cov_radii = covalent_radii[atoms.arrays["numbers"]]
        r_cov = torch.as_tensor(cov_radii, dtype=torch.float64, device=device)
        r_cov = r_cov + r_cov.unsqueeze(1)

        d = _calc_dist_mat(pos, device=device)
        A = (d / r_cov) < bond_scale_t
        A.fill_diagonal_(False)

        nghs = [row.nonzero(as_tuple=False).squeeze(1).cpu().tolist() for row in A]

        if i_img == 0:
            path, c4s = [], []

            def dfs(i):
                if i not in path:
                    path.append(i)
                    if len(path) == 4:
                        if path[0] < path[3]:
                            c4s.append(list(path))
                    else:
                        for j in nghs[i]:
                            dfs(j)
                    path.pop()

            for i in range(len(atoms)):
                dfs(i)

            c4s_center = []
            for i0 in range(len(atoms)):
                neighbors = nghs[i0]
                if len(neighbors) >= 3:
                    for i1 in range(len(neighbors)):
                        for i2 in range(i1 + 1, len(neighbors)):
                            for i3 in range(i2 + 1, len(neighbors)):
                                c4s_center.append([i0, neighbors[i1], neighbors[i2], neighbors[i3]])

            pels_cis = [
                c4 for c4 in c4s
                if rmsd(pos, c4) < tol_rmsd and is_not_linear(atoms, c4) and is_cis(atoms, c4)
            ]
            pels_trans = [
                c4 for c4 in c4s
                if rmsd(pos, c4) < tol_rmsd and is_not_linear(atoms, c4) and is_trans(atoms, c4)
            ]
            pels_center = [c4 for c4 in c4s_center if rmsd(pos, c4) < tol_rmsd]
        else:
            pels_cis = [
                c4 for c4 in pels_cis
                if rmsd(pos, c4) < tol_rmsd and is_not_linear(atoms, c4) and is_cis(atoms, c4) and is_connected(nghs, c4)
            ]
            pels_trans = [
                c4 for c4 in pels_trans
                if rmsd(pos, c4) < tol_rmsd and is_not_linear(atoms, c4) and is_trans(atoms, c4) and is_connected(nghs, c4)
            ]
            pels_center = [
                c4 for c4 in pels_center
                if rmsd(pos, c4) < tol_rmsd and is_connected_center(nghs, c4)
            ]

    pels = [set(pel) for pel in pels_cis + pels_trans + pels_center]
    planes, pels_del = [], []
    while pels:
        if not pels_del:
            planes.append(pels.pop())
        pels_del = [pel for pel in pels if len(planes[-1] & pel) >= 3]
        pels = [pel for pel in pels if len(planes[-1] & pel) < 3]
        planes[-1].update(*pels_del)

    return [sorted(p) for p in planes]
