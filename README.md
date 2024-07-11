# Direct MaxFlux method
A Python implementation of the direct MaxFlux method for transition state search

sample.py
```python
import numpy as np
from ase.io import write, read
from ase.calculators.emt import EMT
from dmf import DirectMaxFlux, interpolate_fbenm

# read react.xyz and prod.xyz
ref_images = [read('react.xyz'), read('prod.xyz')]

# generate initial path by FB-ENM
mxflx_fbenm = interpolate_fbenm(ref_images,correlated=True)

# write initial path and its coefficients
write('sample_ini.traj',mxflx_fbenm.images)
coefs = mxflx_fbenm.coefs.copy()
np.save('sample_ini_coefs',coefs)

# set up a variational problem of the direct MaxFlux method
mxflx = DirectMaxFlux(ref_images,coefs=coefs,nmove=3,update_teval=True)

# set up calculators
for image in mxflx.images:
    image.calc = EMT()

# solve the variational problem
mxflx.add_ipopt_options({'output_file':'sample_ipopt.out'})
mxflx.solve(tol='middle')

# write final path and history of x(tmax)
write('sample_fin.traj',mxflx.images)
write('sample_tmax.traj',mxflx.history.images_tmax)
```

## Requirements

- NumPy
- SciPy
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [cyipopt](https://cyipopt.readthedocs.io/en/stable/)

## Installation

 - If you want to try the direct MaxFlux method once for now, it is sufficient to install the above requirements and copy dmf.py to the directory where your script is located.
 - If you want to install this module on your system, you can install it using a command like:

```
pip install git+https://github.com/shin1koda/dmf.git
```

## Documentation

See this [GitHub Pages](https://shin1koda.github.io/dmf/).

## Limitations

Currently, only non-periodic systems are supported.

## Citation

 1. S.-i. Koda and  S. Saito, Locating Transition States by Variational Reaction Path Optimization with an Energy-Derivative-Free Objective Function, JCTC, 20, 2798–2811 (2024). [doi: 10.1021/acs.jctc.3c01246](https://doi.org/10.1021/acs.jctc.3c01246)
 1. S.-i. Koda and  S. Saito, A Flat-bottom Elastic Network Model for Generating Improved Plausible Reaction Paths, ChemRxiv (2024). [doi: 10.26434/chemrxiv-2024-h15dh](https://doi.org/10.26434/chemrxiv-2024-h15dh)
 1. S.-i. Koda and  S. Saito, A Correlated Flat-bottom Elastic Network Model for Improved Bond Rearrangement in Reaction Paths, ChemRxiv (2024). [doi: ](https://doi.org/)

Please cite:

 - Ref. 1 when you use the direct MaxFlux method
 - Ref. 2 when you use the flat-bottom elasitic network model
 - Ref. 3 when you use the correlated flat-bottom elasitic network model
