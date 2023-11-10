from ase import Atoms
from ase.io import write, read
from ase.calculators.mopac import MOPAC
import numpy as np
import dmf

ref_images=read('idpp_images.traj',index=':')

#avoid mass-weighted coordinate
w = np.ones(len(ref_images[0]))
for image in ref_images:
    image.set_masses(w)

mxflx = dmf.DirectMaxFlux(
    ref_images,
    nmove=3,
    adaptive_method='full_tmax',
    )

#set calcs
for i,image in enumerate(mxflx.images):
    image.calc = MOPAC(label=f'example{i:02}')
mxflx.atoms_react.calc = MOPAC(label=f'example_react')
mxflx.atoms_prod.calc = MOPAC(label=f'example_prod')

#create and solve the variational problem
mxflx.create_problem()
mxflx.problem.add_option('output_file','example.out')
mxflx.solve_problem()

#final path
write('example_fin.traj',mxflx.images)
#history of x(tmax)
write('example_tmax.traj',mxflx.problem.history.images_tmax)
