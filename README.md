# Direct MaxFlux method
A Python implementation of the direct MaxFlux method for transition state search

sample.py
```python
from ase.io import write, read
from ase.calculators.emt import EMT
from dmf import DirectMaxFlux

# read idpp-interpolated images created from react.xyz and prod.xyz
# see ASE's document for how to generate it
ref_images = read('idpp_images.traj',index=':')

# set up a variational problem of the direct MaxFlux method
mxflx = DirectMaxFlux(ref_images,nmove=3,update_teval=True)

# set up calculators
for image in mxflx.images:
    image.calc = EMT()

# solve the variational problem
mxflx.add_ipopt_options({'output_file':'sample_ipopt.out'})
mxflx.solve()

# write final path
write('sample_fin.traj',mxflx.images)
# write history of x(tmax)
write('sample_tmax.traj',mxflx.history.images_tmax)
```

## Requirements

- NumPy
- SciPy
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [cyipopt](https://cyipopt.readthedocs.io/en/stable/)

## Documentation

See this [GitHub Pages](https://shin1koda.github.io/dmf/).

## Limitations

Currently, only non-periodic systems are supported.
