from ase.io import write, read
from ase.calculators.emt import EMT
from dmf import DirectMaxFlux

# read idpp-interpolated images created from react.xyz and prod.xyz
# see ASE's document for how to generate it
ref_images = read('idpp_images.traj',index=':')

# set up direct MaxFlux object
mxflx = DirectMaxFlux(ref_images,nmove=3,update_teval=True)

# set up calculators
for i,image in enumerate(mxflx.images):
    image.calc = EMT()

# create and solve the variational problem
mxflx.create_problem()
mxflx.problem.add_option('output_file','sample_ipopt.out')
mxflx.solve_problem()

# write final path
write('sample_fin.traj',mxflx.images)
# write history of x(tmax)
write('sample_tmax.traj',mxflx.problem.history.images_tmax)
