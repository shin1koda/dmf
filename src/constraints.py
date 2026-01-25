import warnings

import numpy as np
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths


class HarmonicFixAtoms(Calculator):
    """
    Harmonic position restraint on a subset of atoms.

    E = 1/2 * k_fix * sum_i |r_i - r_i^ref|^2
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, indices, ref_positions, k_fix=300.0):
        super().__init__()
        idx = np.asarray(indices, dtype=int).ravel()
        if idx.size == 0:
            raise ValueError("HarmonicFixAtoms requires at least one index.")
        ref_pos = np.asarray(ref_positions, dtype=float)
        if ref_pos.shape != (idx.size, 3):
            raise ValueError(
                f"ref_positions must have shape ({idx.size}, 3), got {ref_pos.shape}"
            )
        self.indices = idx
        self.ref_positions = ref_pos
        self.k_fix = float(k_fix)

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions().astype(float)
        disp = pos[self.indices] - self.ref_positions
        energy = 0.5 * self.k_fix * np.sum(disp ** 2)
        forces = np.zeros_like(pos, dtype=float)
        forces[self.indices] = -self.k_fix * disp
        self.results = {
            "energy": float(energy),
            "forces": forces,
        }


class HarmonicFixBondLengths(Calculator):
    """
    Harmonic bond-length restraint for selected atom pairs.

    E = 1/2 * k_bond * sum_ij (d_ij - d_ij^ref)^2
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, pairs, ref_distances, k_bond=300.0):
        super().__init__()
        pairs = np.asarray(pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")
        ref_distances = np.asarray(ref_distances, dtype=float).ravel()
        if ref_distances.shape[0] != pairs.shape[0]:
            raise ValueError(
                "ref_distances must have same length as pairs"
            )
        self.pairs = pairs
        self.ref_distances = ref_distances
        self.k_bond = float(k_bond)

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions().astype(float)
        i_idx = self.pairs[:, 0]
        j_idx = self.pairs[:, 1]
        rij = pos[i_idx] - pos[j_idx]
        dist = np.linalg.norm(rij, axis=1)
        diff = dist - self.ref_distances
        energy = 0.5 * self.k_bond * np.sum(diff ** 2)
        inv_dist = np.zeros_like(dist)
        nonzero = dist > 0.0
        inv_dist[nonzero] = 1.0 / dist[nonzero]
        force_scale = -self.k_bond * diff * inv_dist
        fij = (force_scale[:, None] * rij).astype(float)
        forces = np.zeros_like(pos, dtype=float)
        np.add.at(forces, i_idx, fij)
        np.add.at(forces, j_idx, -fij)
        self.results = {
            "energy": float(energy),
            "forces": forces,
        }


def _iter_constraints(atoms):
    constraints = getattr(atoms, "constraints", None)
    if constraints is None:
        return []
    if isinstance(constraints, (list, tuple)):
        return list(constraints)
    return [constraints]


def build_harmonic_constraint_calculators(atoms, k_fix=300.0, k_bond=300.0):
    """
    Build harmonic-restraint calculators from ASE constraints.

    Supported constraints:
    - FixAtoms
    - FixBondLength / FixBondLengths
    """

    calculators = []
    constraints = _iter_constraints(atoms)
    remaining_constraints = []
    for constraint in constraints:
        if isinstance(constraint, FixAtoms):
            indices = np.asarray(constraint.index, dtype=int)
            ref_positions = atoms.get_positions()[indices]
            calculators.append(
                HarmonicFixAtoms(indices, ref_positions, k_fix=k_fix)
            )
        elif isinstance(constraint, FixBondLengths):
            pairs = np.asarray(constraint.pairs, dtype=int)
            if constraint.bondlengths is None:
                pos = atoms.get_positions()
                rij = pos[pairs[:, 0]] - pos[pairs[:, 1]]
                ref_distances = np.linalg.norm(rij, axis=1)
            else:
                ref_distances = np.asarray(constraint.bondlengths, dtype=float)
            calculators.append(
                HarmonicFixBondLengths(pairs, ref_distances, k_bond=k_bond)
            )
            remaining_constraints.append(constraint)
        else:
            warnings.warn(
                f"Unsupported ASE constraint {type(constraint).__name__}; "
                "ignored for harmonic restraints."
            )
            remaining_constraints.append(constraint)
    if remaining_constraints != constraints:
        atoms.set_constraint(remaining_constraints or None)
    return calculators


def add_harmonic_constraints(atoms, calculators, k_fix=300.0, k_bond=300.0):
    """
    Append harmonic constraint calculators to a calculator list.
    """

    calcs = list(calculators) if calculators is not None else []
    calcs.extend(build_harmonic_constraint_calculators(atoms, k_fix=k_fix, k_bond=k_bond))
    return calcs
