#!/usr/bin/env python

import numpy as np
from scipy.special import factorial as fac

def hh_model_1d(nao, U, w, lam):
    hcore = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore[i, i+1] = hcore[i+1, i] = -1
    hcore[0, nao-1] = hcore[nao-1, 0] = -1

    eri = np.zeros((nao, nao, nao, nao))
    eri[range(nao), range(nao), range(nao), range(nao)] = U

    h_ep_const = np.sqrt(lam * w)
    h_ep = np.zeros((nao, nao, nao))
    h_ep[range(nao), range(nao), range(nao)] = h_ep_const

    #w_p = np.zeros((nao, nao))
    #w_p[range(nao), range(nao)] = w
    w_p = np.zeros((nao,))
    w_p[range(nao)] = w

    return hcore, eri, h_ep, w_p


def test_fci_hh_model():
    from scipy import linalg as la
    from polar.fci import fci

    np.set_printoptions(3, linewidth=1000)

    U = 2.0
    w_p = 0.5
    lam = 0.5

    nao = 4
    hcore, eri, h_ep, hpp = hh_model_1d(nao, U, w_p, lam)
    g = h_ep[0, 0, 0]

    norb = hcore.shape[-1]
    nelec = norb
    nmode = norb
    nph = 12

    ci_shape = fci.get_ci_shape(norb, nelec, nmode=nmode, nph=nph)
    np.random.seed(1)
    fcivec = np.random.random(ci_shape)

    # 1. ref
    from pyscf.fci import direct_ep
    e_ref, civec = direct_ep.kernel(hcore, U, g, np.diag(hpp), nsite=norb, nelec=nelec,
                                    nphonon=nph, tol=1e-9)
    e_ref -= 2.0

    # 2. shifted
    e, (civec, zs) = fci.kernel(hcore, eri, norb, nelec, nmode, nph, h_ep, hpp,
                                shift_vac=True, ecore=0, tol=1e-9, max_cycle=100000)
    diff = abs(e - e_ref)
    print ("e_fci", e)
    print ("diff", diff)
    assert diff < 1e-8

    rdm1_ee = fci.make_rdm1e(civec, norb, nelec)
    print ("rdm1 elec")
    print (rdm1_ee)

    rdm1_ph_linear = fci.make_rdm1p_linear(civec, norb, nelec, nmode, nph, zs=zs)
    print ("rdm1 ph linear")
    print (rdm1_ph_linear)

    rdm1_ph = fci.make_rdm1p(civec, norb, nelec, nmode, nph, zs=zs)
    print ("rdm1 ph")
    print (rdm1_ph)

    # 3. full hpp
    e_full, (civec, zs) = fci.kernel(hcore, eri, norb, nelec, nmode, nph, h_ep, np.diag(hpp),
                                shift_vac=True, ecore=0, tol=1e-9, max_cycle=100000)
    diff = abs(e_full - e)
    print ("e_fci (full hpp)", e_full)
    print ("diff", diff)
    assert diff < 1e-8

    # 4. unshifted
    e_unshift, civec = fci.kernel(hcore, eri, norb, nelec, nmode, nph, h_ep, hpp,
                                  shift_vac=False, ecore=0, tol=1e-9, max_cycle=100000)

    diff = abs(e - e_unshift)
    print ("e_fci unshift", e_unshift)
    print ("diff shift and unshift", diff)
    assert diff < 1e-5

    rdm1_ee_2 = fci.make_rdm1e(civec, norb, nelec)
    print ("rdm1 elec")
    print (rdm1_ee_2)
    diff = la.norm(rdm1_ee_2 - rdm1_ee)
    print ("diff")
    print (diff)
    assert diff < 1e-4

    rdm1_ph_linear_2 = fci.make_rdm1p_linear(civec, norb, nelec, nmode, nph)
    print ("rdm1 ph linear")
    print (rdm1_ph_linear_2)
    diff = la.norm(rdm1_ph_linear_2 - rdm1_ph_linear)
    print ("diff")
    print (diff)
    assert diff < 1e-4

    rdm1_ph_2 = fci.make_rdm1p(civec, norb, nelec, nmode, nph)
    print ("rdm1 ph")
    print (rdm1_ph_2)
    diff = la.norm(rdm1_ph_2 - rdm1_ph)
    print ("diff")
    print (diff)
    assert diff < 1e-4

def test_hpp_full():
    from scipy import linalg as la
    from polar.fci import fci

    np.set_printoptions(3, linewidth=1000)

    U = 2.0
    w_p = 0.5
    lam = 0.5
    #lam = 0.0

    nao = 4
    hcore, eri, h_ep, hpp = hh_model_1d(nao, U, w_p, lam)
    hpp = np.diag(hpp)
    g = h_ep[0, 0, 0]

    norb = hcore.shape[-1]
    nelec = norb
    nmode = norb
    nph = 12

    for i in range(nmode-1):
        hpp[i, i+1] = hpp[i+1, i] = 0.1
    hpp[0, nmode-1] = 0.1
    hpp[nmode-1, 0] = 0.1

    from pyscf.fci import direct_ep
    e_ref, civec = direct_ep.kernel(hcore, U, g, hpp, nsite=norb, nelec=nelec,
                                    nphonon=nph, tol=1e-9)
    e_ref -= 1.428571428571428
    print ("e_ref", e_ref)

    rdm1_ee = direct_ep.make_rdm1e(civec, norb, nelec)
    print ("rdm1 elec")
    print (rdm1_ee)

    rdm1_ph = direct_ep.make_rdm1p(civec, norb, nelec, nph)
    print ("rdm1 ph")
    print (rdm1_ph)

    e_full, (civec, zs) = fci.kernel(hcore, eri, norb, nelec, nmode, nph, h_ep, hpp,
                                     shift_vac=True, ecore=0, tol=1e-9, max_cycle=100000)
    diff = abs(e_ref - e_full)
    print ("e_fci full", e_full)
    print ("diff diag and full", diff)
    assert diff < 1e-5

    rdm1_ee_2 = fci.make_rdm1e(civec, norb, nelec)
    print ("rdm1 elec")
    print (rdm1_ee_2)
    diff = la.norm(rdm1_ee_2 - rdm1_ee)
    print ("diff")
    print (diff)
    assert diff < 1e-4

    # ZHC NOTE there is a bug in PySCF's implementation, which ignores the zs
    zs = None
    rdm1_ph_2 = fci.make_rdm1p(civec, norb, nelec, nmode, nph, zs=zs)
    print ("rdm1 ph")
    print (rdm1_ph_2)
    diff = la.norm(rdm1_ph_2 - rdm1_ph)
    print ("diff")
    print (diff)
    assert diff < 1e-4

def fc_factor_ref(n, m, l):
    """
    Get the Franck-Condon factors, <n|exp(-l(b-b+))|m>
    Displacement op matrix element from definition.
    """
    lsq = l * l
    factor  = np.exp(-lsq * 0.5)
    factor /= np.sqrt(fac(n)) * np.sqrt(fac(m))

    res = 0.0
    if m >= n:
        for s in range(n + 1):
            res += (-1) ** (m-n+s) * l ** (m - n + 2*s) / fac(s) * fac(m) / fac(m-n+s) * fac(n) / fac(n-s)
    else:
        for s in range(m + 1):
            res += (-1) ** (s) * l ** (n - m + 2*s) / fac(s) * fac(n) / fac(n-m+s) * fac(m) / fac(m-s)

    res *= factor
    return res

def test_fc_factor():
    from polar.fci import fci
    np.set_printoptions(3, linewidth=1000, suppress=True)

    l = 0.51
    for n in range(0, 10):
        for m in range(0, 10):
            val = fci.fc_factor(n, m, l)
            val_ref = fc_factor_ref(n, m, l)
            diff = abs(val - val_ref)
            print (n, m, val, diff)
            assert diff < 1e-13

    l = 0.1
    fc_arr = fci.get_fc_arr(10, l)
    print (fc_arr)

    assert fci.fc_factor(0, 1, l) == -fci.fc_factor(1, 0, l)
    assert fci.fc_factor(1, 3, l) == fci.fc_factor(3, 1, l)
    assert fci.fc_factor(10, 2, l) == fci.fc_factor(2, 10, l)
    assert fci.fc_factor(10, 3, l) == -fci.fc_factor(3, 10, l)

if __name__ == "__main__":
    test_hpp_full()
    test_fci_hh_model()
    test_fc_factor()
