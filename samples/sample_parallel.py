from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.gaussian import Gaussian
from ase.io import read, write

from dmf import DirectMaxFlux, interpolate_fbenm

# This example assumes Open MPI and Gaussian 16. # Load and configure them before running this script. In particular, # set the required Gaussian environment variables and create # GAUSS_SCRDIR on every compute node used by the calculation.

NMOVE = 3

# Hypothetical node and CPU assignments for five images.
nodes = [
    "node001",
    "node001",
    "node002",
    "node003",
    "node002",
]

cpus = [
    ",".join(str(i) for i in range(8))
] * 5

g_options = dict(
    method="B3LYP",
    basis="6-31G(d)",
    ioplist=["2/11=1", "2/12=1"],
    scf="yqc",
)


def make_gaussian_calc_factory(
    nodes: list[str],
    cpus: list[str],
    g_options: Mapping[str, Any],
) -> Callable[[int], Gaussian]:
    """
    Create a Gaussian calculator factory for DirectMaxFlux images.

    The node names and CPU lists are assumed to be assigned to each image
    in advance.
    """
    base_options = dict(g_options)

    def calc_factory(i: int) -> Gaussian:
        """Create the Gaussian calculator assigned to image i."""
        img_dir = Path(f"image{i:02}")
        img_dir.mkdir(exist_ok=True)

        options = dict(base_options)

        label = options.get("label", "gaussian")
        chk = options.get("chk", "gaussian.chk")

        options["label"] = str(img_dir / label)
        options["chk"] = str(chk)
        options["cpu"] = cpus[i]

        ncpus = len(cpus[i].split(","))
        options["mem"] = f"{1200 * ncpus}MB"

        options["command"] = (
            f"mpirun -np 1 "
            f"--host {nodes[i]} "
            f"--bind-to none "
            f"g16 < PREFIX.com > PREFIX.log"
        )

        return Gaussian(**options)

    return calc_factory


calc_factory = make_gaussian_calc_factory(
    nodes=nodes,
    cpus=cpus,
    g_options=g_options,
)

# Read reactant and product structures.
ref_images = [
    read("react.xyz"),
    read("prod.xyz"),
]

coefs_path = Path("sample_ini_coefs.npy")

if coefs_path.is_file():
    coefs = np.load(coefs_path)
else:
    # Generate the initial path using FB-ENM.
    mxflx_fbenm = interpolate_fbenm(
        ref_images,
        correlated=True,
    )

    write("sample_ini.traj", mxflx_fbenm.images)

    coefs = mxflx_fbenm.coefs.copy()
    np.save("sample_ini_coefs.npy", coefs)

# Set up the Direct MaxFlux variational problem.
mxflx = DirectMaxFlux(
    ref_images,
    coefs=coefs,
    nmove=NMOVE,
    update_teval=True,
    calc_factory=calc_factory,
    parallel=True,
)

# Perform the initial electronic-structure calculations.
mxflx.get_forces()

# Reuse checkpoint files in subsequent calculations.
for image in mxflx.images:
    image.calc.set(guess="read")

# Solve the variational problem.
mxflx.add_ipopt_options(
    {"output_file": "sample_ipopt.out"}
)
mxflx.solve(tol="middle")

# Save the optimized path and the history of x(tmax).
write("sample_fin.traj", mxflx.images)
write("sample_tmax.traj", mxflx.history.images_tmax)
