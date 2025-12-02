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
    mxflx = interpolate_fbenm([r, p], nmove=5, output_file="tests/fbenm_ipopt.out")

    expected_coefs = np.load("tests/coefs.npy")

    assert_allclose(
        mxflx.coefs,
        expected_coefs,
        rtol=1e-5,
        atol=1e-8
    )

