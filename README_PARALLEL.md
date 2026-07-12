# Parallel energy/force evaluation

## Overview

PyDMF supports parallel energy and force evaluations at the energy evaluation points along the path. As of v1.2.1, PyDMF implements only thread-based parallelization. In practice, PyDMF runs multiple ASE calculators concurrently.

Users are therefore responsible for assigning computational resources to each calculator and for avoiding conflicts in output and intermediate files.

A convenient way to configure the calculators is to define a callable named `calc_factory`. The callable receives an image index `i` and returns an ASE calculator configured for that image:

```python
calc = calc_factory(i)
```

When `calc_factory` is passed to `DirectMaxFlux`, PyDMF calls it for each image and automatically attaches the returned calculator to that image. This makes it possible to keep node selection, CPU allocation, memory settings, file names, and working-directory settings in one place.

## Details

Parallel evaluation is enabled by passing `parallel=True` when constructing a `DirectMaxFlux` object.

The evaluations are parallelized as follows:

* In the initial step, the energy and force calculations for the two endpoint images, `images[0]` and `images[nimages - 1]`, are performed in parallel.
* In subsequent steps, the energy and force calculations for the internal images, from `images[1]` to `images[nimages - 2]`, are performed in parallel. The number of internal images is `nimages - 2`, which is equal to `nmove`.

The resource assignments for the endpoint images and the internal images may therefore overlap, because these two groups are evaluated at different stages.


## Example: Gaussian calculators

The following simplified example corresponds to `samples/sample_parallel.py`. Although Gaussian is used here as an example, the main point is to illustrate how computational resources can be assigned to individual images and how conflicts in output and intermediate files can be avoided.

The example assumes that the node names and CPU lists have already been assigned to the individual images. Each Gaussian calculation:

* is launched on a specified node using Open MPI,
* uses a specified list of CPU IDs (on some systems, specifying only nprocshared may be sufficient),
* writes its files to a separate `imageXX` directory,
* uses a separate checkpoint file, and
* receives 1200 MB of Gaussian memory per assigned CPU.

This example assumes Open MPI and Gaussian 16. The required MPI and Gaussian environments must be configured before running the script. In particular, the required Gaussian environment variables must be set, and `GAUSS_SCRDIR` must exist on every compute node used by the calculation.

The working directory must also be visible from all compute nodes, for example through an NFS- or Lustre-mounted filesystem.

```python
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ase.calculators.gaussian import Gaussian
from dmf import DirectMaxFlux


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

mxflx = DirectMaxFlux(
    [react, prod],
    nmove=NMOVE,
    calc_factory=calc_factory,
    parallel=True,
    ...
)
```

When the `DirectMaxFlux` object is initialized, PyDMF effectively performs the following assignment for each image:

```python
for i, image in enumerate(mxflx.images):
    image.calc = calc_factory(i)
```

The factory approach is useful because resource allocation and calculator construction are defined independently of the path-optimization code. The same `DirectMaxFlux` workflow can therefore be used with different node layouts, CPU assignments, ASE calculators, or HPC environments by replacing only the factory.

Depending on the MPI implementation and the local environment, it may be necessary to explicitly export required environment variables to MPI-launched processes. The exact MPI command and environment setup are system dependent.

