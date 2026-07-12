def test_import_cyipopt():
    import cyipopt

def test_import_ase():
    import ase

def test_import_dmf():
    import dmf

def test_interpolate_fbenm_basic():

    from ase import Atoms
    from dmf import interpolate_fbenm
    import numpy as np
    from numpy.testing import assert_allclose

    r = Atoms('HOH', positions=[[0, 0, -1], [0, 1, 0], [0, 0, 1]])
    p = Atoms('HOH', positions=[[0, 0.5, 0], [1, 0, 0], [0, -0.5, 0]])
    result = interpolate_fbenm([r, p], nmove=5, output_file="tests/fbenm_ipopt.out")

    assert hasattr(result, "coefs")

    expected_coefs = np.load("tests/coefs.npy")

    assert_allclose(
        result.coefs,
        expected_coefs,
        rtol=1e-5,
        atol=1e-8
    )

def test_maxflux():

    import numpy as np
    from ase.io import write, read
    from ase.calculators.emt import EMT
    from dmf import DirectMaxFlux, interpolate_fbenm

    # read react.xyz and prod.xyz
    ref_images = [read('samples/react.xyz'), read('samples/prod.xyz')]

    # generate initial path by FB-ENM
    mxflx_fbenm = interpolate_fbenm(ref_images,correlated=True)
    coefs = mxflx_fbenm.coefs.copy()

    # set up a variational problem of the direct MaxFlux method
    mxflx = DirectMaxFlux(ref_images,coefs=coefs,nmove=3,update_teval=True)

    # set up calculators
    for image in mxflx.images:
        image.calc = EMT()

    # solve the variational problem
    mxflx.add_ipopt_options({'output_file':'tests/sample_ipopt.out'})
    mxflx.solve(tol='middle')
