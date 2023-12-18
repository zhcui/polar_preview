#!/usr/bin/env python

"""
Test Thermal average of polaron transformed h1 and h2.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

def test_h1():
    import numpy as np
    from scipy import linalg as la
    from pyscf import lib
    from polar.lang_firsov import thermal_average as ta

    np.set_printoptions(4, linewidth=1000, suppress=True)

    print ("*" * 79)
    print ("h1 averaging")
    print ()

    nmode = 3
    nao = 10

    lams_diag = np.random.random((nmode, nao))
    lams = np.zeros((nmode, nao, nao))
    for x in range(nmode):
        lams[x][range(nao), range(nao)] = lams_diag[x]
    print ("lams diag\n%s"%lams_diag)

    h1 = np.random.random((nao, nao)) - 0.5
    h1 = h1 + h1.T
    h1[range(nao), range(nao)] *= 5
    print ("bare h1\n%s"%h1)

    diff = lib.direct_sum('xq - xp -> xpq', lams_diag, lams_diag)
    diff **= 2
    factor = diff.sum(axis=0)
    diff = None
    factor *= (-0.5)
    factor = np.exp(factor)
    H1 = h1 * factor
    print ("polaron h1\n%s"%H1)
    print ("error: ", la.norm(H1 - h1))
    print ("-" * 79)

    H1_exact = ta.bch_h1_exact(h1, lams, order=6, H1_ref=H1)
    H1_2 = ta.bch_h1(h1, lams, order=6, H1_ref=H1)
    print ("Compare exact and approx formuala for diagonal lams:", la.norm(H1_exact - H1_2))
    assert la.norm(H1_exact - H1_2) < 1e-10

    H1_exp_ref = ta.bch_h1_exp_ref(h1, lams)
    print ("H1 exp ref\n%s"%H1_exp_ref)
    diff = la.norm(H1_exp_ref - H1)
    print ("diff exp ref and expansion", diff)
    assert diff < 1e-2

    H1_exp = ta.bch_h1_exp(h1, lams)
    diff = la.norm(H1_exp - H1_exp_ref)
    print ("diff exp fast and exp ref", diff)
    assert diff < 1e-10

    # ****************************************************************************
    # non-local parameters
    # ****************************************************************************

    print ("*" * 79)
    print ("non-local paramters")
    lams = np.random.random((nmode, nao, nao)) - 0.5
    lams = lams + lams.transpose(0, 2, 1)
    lams *= 0.1
    lams[:, range(nao), range(nao)] *= 5

    H1_exact = ta.bch_h1_exact(h1, lams, order=10)
    H1_2 = ta.bch_h1(h1, lams, order=10)
    H1_exp_ref = ta.bch_h1_exp_ref(h1, lams)
    H1_exp = ta.bch_h1_exp(h1, lams)

    print ("h1\n%s"%h1)
    print ("H1 exact\n%s"%H1_exact)
    print ("H1 approx\n%s"%H1_2)
    print ("error approx - exact", la.norm(H1_exact - H1_2))

    print ("-" * 79)
    print ("H1 exp vs expansion")
    print (la.norm(H1_exp - H1_2))
    print ("H1 exp diff")
    print (la.norm(H1_exp - H1_exp_ref))
    assert la.norm(H1_exp - H1_exp_ref) < 1e-10

def test_h2():
    import numpy as np
    from scipy import linalg as la
    from pyscf import lib
    from polar.lang_firsov import thermal_average as ta
    from pyscf import gto, ao2mo
    from libdmet.utils import cholesky

    np.set_printoptions(4, linewidth=1000, suppress=True)

    print ("*" * 79)
    print ("ERI averaging")
    print ()

    nmode = 2
    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 1.1',  # in Angstrom
        basis = '321g',
        verbose = 4,
        dump_input=False)
    mf = mol.HF()
    mf.kernel()
    nao = mol.nao_nr()
    h2 = ao2mo.restore(1, mf._eri, nao)

    cderi = cholesky.get_cderi_rhf(ao2mo.restore(4, h2, nao), nao)
    h2_re = np.einsum('Lpq, Lrs -> pqrs', cderi, cderi)

    lams_diag = np.random.random((nmode, nao)) #* 10
    lams = np.zeros((nmode, nao, nao))
    for x in range(nmode):
        lams[x][range(nao), range(nao)] = lams_diag[x]

    factor = 0.0
    for x in range(nmode):
        diff = lib.direct_sum('q - p + s - r -> pqrs', lams_diag[x], lams_diag[x],
                              lams_diag[x], lams_diag[x])
        diff **= 2
        factor += diff
        diff = None
    factor *= (-0.5)
    factor = np.exp(factor)
    H2 = h2 * factor

    print ("lams\n%s"%lams)
    print ("h2\n%s"%h2[0, 0])
    print ("polaron h2\n%s"%H2[0, 0])
    print ("diff: ", la.norm(H2 - h2))
    print ("-" * 79)

    H2_exact = ta.bch_h2_exact(h2, lams, order=10, H2_ref=H2)
    H2_2 = ta.bch_h2(h2, lams, order=10, H2_ref=H2)

    H2_exp_ref = ta.bch_h2_exp_ref(h2, lams)
    H2_exp = ta.bch_h2_exp(h2, lams)

    print ("H2 exact vs H2")
    print (la.norm(H2_2 - H2_exact))
    assert la.norm(H2_2 - H2_exact) < 1e-10

    print ("H2 approx vs H2")
    print (la.norm(H2_2 - H2))

    print ("H2 exp fast vs H2")
    print (la.norm(H2_exp - H2))
    assert la.norm(H2_exp - H2) < 1e-10

    print ("H2 exp ref vs exp fast")
    print (la.norm(H2_exp_ref - H2_exp))
    assert la.norm(H2_exp_ref - H2_exp) < 1e-10

    # ****************************************************************************
    # non-local parameters
    # ****************************************************************************

    np.random.seed(100)

    nmode = 3
    nao = 5

    h2 = np.random.random((nao, nao, nao, nao)) - 0.5
    h2 = h2 + h2.transpose(0, 1, 3, 2)
    h2 = h2 + h2.transpose(1, 0, 2, 3)
    h2 = h2 + h2.transpose(2, 3, 0, 1)

    print ("*" * 79)
    print ("non-local paramters")
    lams = np.random.random((nmode, nao, nao)) - 0.5
    lams = lams + lams.transpose(0, 2, 1)
    lams *= 0.1
    lams[:, range(nao), range(nao)] *= 2

    H2_exact = ta.bch_h2_exact(h2, lams, order=8)
    H2_2 = ta.bch_h2(h2, lams, order=8)
    H2_exp_ref = ta.bch_h2_exp_ref(h2, lams)
    H2_exp = ta.bch_h2_exp(h2, lams)

    print ("h2\n%s"%h2[0, 0])
    print ("H2 exact\n%s"%H2_exact[0, 0])
    print ("H2 approx\n%s"%H2_2[0, 0])
    print ("error approx - exact", la.norm(H2_exact - H2_2))

    print ("-" * 79)
    print ("H2 exp\n%s"%H2_exp[0, 0])
    print ("H2 exp vs expansion")
    print (np.max(np.abs(H2_exp - H2_2)))
    print ("H2 exp diff")
    print (la.norm(H2_exp - H2_exp_ref))
    assert la.norm(H2_exp - H2_exp_ref) < 1e-5

if __name__ == "__main__":
    test_h1()
    test_h2()
