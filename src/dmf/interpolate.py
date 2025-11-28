import numpy as np
from ase.calculators.mixing import SumCalculator
from .dmf import DirectMaxFlux
from .fbenm import FB_ENM_Bonds, CFB_ENM


def interpolate_fbenm(
        ref_images,nmove=10,
        output_file='fbenm_ipopt.out',
        correlated=True,
        sequential=True,
        fbenm_only_endpoints=True,
        fbenm_options={},
        cfbenm_options={},
        dmf_options={},
        ):

    mxflx = DirectMaxFlux(ref_images,
                          nmove=nmove,
                          update_teval=False,
                          **dmf_options)

    if fbenm_only_endpoints:
        fbenm_images = [ref_images[0].copy(),ref_images[-1].copy()]
    else:
        fbenm_images = [image.copy() for image in ref_images]

    for i,image in enumerate(mxflx.images):
        if correlated:
            image.calc = SumCalculator([
                             FB_ENM_Bonds(fbenm_images, **fbenm_options),
                             CFB_ENM(fbenm_images,**cfbenm_options)])
        else:
            image.calc = FB_ENM_Bonds(fbenm_images, **fbenm_options)

    options ={
        'tol': 0.1,
        'dual_inf_tol': 0.01,
        'constr_viol_tol': 0.01,
        'compl_inf_tol': 0.01,
        'nlp_scaling_method':'user-scaling',
        'obj_scaling_factor':0.1,
        'limited_memory_initialization':'constant',
        'limited_memory_init_val':2.5,
        'accept_every_trial_step':'yes',
        'output_file':output_file,
        'max_iter':200,
        }
    mxflx.add_ipopt_options(options)


    if sequential:
        b_scale = 3.0
        w_eval0 = mxflx.w_eval.copy()
        for i in range((nmove+1)//2):
            mxflx.get_forces()
            ens = mxflx.energies.copy()
            w_eval = w_eval0.copy()
            ens[i+2:nmove-i]=0.0
            w_eval[i+2:nmove-i]=0.0
            if np.amax(ens)>0.0:
                mxflx.beta=b_scale/np.amax(ens)
            else:
                mxflx.beta=1.0
            mxflx.set_w_eval(w_eval)

            mxflx.solve(tol=0.1)

    b_scale = 5.0
    for _ in range(5):
        mxflx.get_forces()
        ens = mxflx.energies.copy()
        if np.amax(ens)>0.0:
            mxflx.beta = b_scale/np.amax(ens)
        else:
            mxflx.beta = 1.0

        mxflx.solve(tol=0.1)

    return mxflx
