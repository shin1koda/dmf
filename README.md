# Direct MaxFlux method
A Python implementation of the direct MaxFlux method for transition state search

example.py
```python
from ase import Atoms
from ase.io import write, read
from ase.calculators.mopac import MOPAC
import numpy as np
import dmf

# read idpp-interpolated images created from react.xyz and prod.xyz
# see ASE's document for how to generate it
ref_images=read('idpp_images.traj',index=':')

# avoid mass-weighted coordinate
# normal coordinate is preferable for a rapid convergence
w = np.ones(len(ref_images[0]))
for image in ref_images:
    image.set_masses(w)

# set up direct MaxFlux object
mxflx = dmf.DirectMaxFlux(
    ref_images,
    nmove=3,
    adaptive_method='full_tmax',
    )

# set up calculators
for i,image in enumerate(mxflx.images):
    image.calc = MOPAC(label=f'example{i:02}')
mxflx.atoms_react.calc = MOPAC(label=f'example_react')
mxflx.atoms_prod.calc = MOPAC(label=f'example_prod')

# create and solve the variational problem
mxflx.create_problem()
mxflx.problem.add_option('output_file','example.out')
mxflx.solve_problem()

# final path
write('example_fin.traj',mxflx.images)
# history of x(tmax)
write('example_tmax.traj',mxflx.problem.history.images_tmax)
```

## Requirements

- NumPy
- SciPy
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [cyipopt](https://cyipopt.readthedocs.io/en/stable/)

## Documentation

Coming soon.

## Limitations

Currently, the implementation is limited as described below. Additional features will be added in the future.

- Only non-periodic systems are supported.
- Only parallel energy/force evaluation by threading can be used (MPI is not supported).
