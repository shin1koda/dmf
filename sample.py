import numpy as np
from ase.io import write, read
from ase.calculators.emt import EMT
from dmf import DirectMaxFlux, interpolate_fbenm

# read react.xyz and prod.xyz
ref_images = [read('react.xyz'), read('prod.xyz')]

# generate initial path by FB-ENM
mxflx_fbenm = interpolate_fbenm(ref_images)

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
