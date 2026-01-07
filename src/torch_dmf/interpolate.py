import numpy as np
from ase.calculators.mixing import SumCalculator

from .dmf import DirectMaxFlux
from .fbenm import FB_ENM_Bonds, CFB_ENM


def interpolate_fbenm(
    ref_images,
    nmove=10,
    output_file="fbenm_ipopt.out",
    correlated=True,
    sequential=True,
    fbenm_only_endpoints=True,
    fbenm_options={},
    cfbenm_options={},
    dmf_options={},
    device=None,
):
    """
    Build a DMF object with ENM calculators and optimize the path.

    All ENM internals now execute on GPU when available. IPOPT/ASE remain
    unchanged. Numerical behavior matches the original version.

    Optional
    --------
    device : Optional[Union[str, torch.device]]
        Torch device for all internal tensors (DMF and ENM). If None, use default.
    """
    dmf_kw = dict(dmf_options) if dmf_options is not None else {}
    fbenm_kw = dict(fbenm_options) if fbenm_options is not None else {}
    cfbenm_kw = dict(cfbenm_options) if cfbenm_options is not None else {}
    if device is None:
        device = dmf_kw.pop("device", device)
        device = fbenm_kw.pop("device", device)
        device = cfbenm_kw.pop("device", device)
    else:
        dmf_kw.pop("device", None)
        fbenm_kw.pop("device", None)
        cfbenm_kw.pop("device", None)

    mxflx = DirectMaxFlux(
        ref_images,
        nmove=nmove,
        update_teval=False,
        device=device,
        **dmf_kw,
    )

    fbenm_images = (
        [ref_images[0].copy(), ref_images[-1].copy()]
        if fbenm_only_endpoints
        else [img.copy() for img in ref_images]
    )

    for image in mxflx.images:
        calcs = []

        if correlated:
            calcs.append(
                FB_ENM_Bonds(
                    fbenm_images,
                    device=device,
                    **fbenm_kw,
                )
            )
            calcs.append(
                CFB_ENM(
                    fbenm_images,
                    device=device,
                    **cfbenm_kw,
                )
            )
        else:
            calcs.append(
                FB_ENM_Bonds(
                    fbenm_images,
                    device=device,
                    **fbenm_kw,
                )
            )

        if len(calcs) == 1:
            image.calc = calcs[0]
        else:
            image.calc = SumCalculator(calcs)

    mxflx.add_ipopt_options(
        {
            "tol": 0.1,
            "dual_inf_tol": 0.01,
            "constr_viol_tol": 0.01,
            "compl_inf_tol": 0.01,
            "nlp_scaling_method": "user-scaling",
            "obj_scaling_factor": 0.1,
            "limited_memory_initialization": "constant",
            "limited_memory_init_val": 2.5,
            "accept_every_trial_step": "yes",
            "output_file": output_file,
            "max_iter": 200,
        }
    )

    if sequential:
        b_scale = 3.0
        w_eval0 = mxflx.w_eval.copy()
        for i in range((nmove + 1) // 2):
            mxflx.get_forces()
            ens = mxflx.energies.copy()
            w_eval = w_eval0.copy()
            ens[i + 2: nmove - i] = 0.0
            w_eval[i + 2: nmove - i] = 0.0
            mxflx.beta = b_scale / np.amax(ens) if np.amax(ens) > 0.0 else 1.0
            mxflx.set_w_eval(w_eval)
            mxflx.solve(tol=0.1)

    b_scale = 5.0
    for _ in range(5):
        mxflx.get_forces()
        ens = mxflx.energies.copy()
        mxflx.beta = b_scale / np.amax(ens) if np.amax(ens) > 0.0 else 1.0
        mxflx.solve(tol=0.1)

    return mxflx
