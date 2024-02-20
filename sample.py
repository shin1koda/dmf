from ase.io import write, read
from ase.calculators.emt import EMT
from dmf import DirectMaxFlux

ref_images = read('idpp_images.traj',index=':')

mxflx = DirectMaxFlux(ref_images,nmove=3,
    adaptive_method='full_tmax',)

#set calcs
for i,image in enumerate(mxflx.images):
    image.calc = EMT()

#create and solve the variational problem
mxflx.create_problem()
mxflx.problem.add_option('output_file','sample_ipopt.out')
mxflx.solve_problem()

#final path
write('sample_fin.traj',mxflx.images)
#history of x(tmax)
write('sample_tmax.traj',mxflx.problem.history.images_tmax)
