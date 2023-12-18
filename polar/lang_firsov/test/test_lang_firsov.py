#!/usr/bin/env python

def test_lf_ham():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import lang_firsov as lf

    np.random.seed(10086)
    np.set_printoptions(3, linewidth=1000, suppress=True)
    nao = 4
    nmode = nao
    nelec = 1
    w_p = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * w_p)

    hcore = np.random.random((nao, nao)) - 0.5
    hcore = hcore + hcore.conj().T

    params = (np.random.random(nao * 2) - 0.5)
    w_p_arr = np.ones(nmode) * w_p

    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)
    #eri = np.zeros((nao, nao, nao, nao))
    #eri[range(nao), range(nao), range(nao), range(nao)] = 8.0

    mylf = lf.LangFirsov(h0=0.0, h1=hcore, h2=eri, h_ep=g, w_p=w_p_arr,
                         nelec=1, uniform=False, spin=1)
    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep, w_p = mylf.get_lf_ham(params)

    # GLF
    lams_g = np.diag(lams)
    g_g = np.zeros((nmode, nao))
    g_g[range(nmode), range(nao)] = g
    w_p_g = np.zeros((nmode))
    w_p_g[:] = w_p

    params_g = np.hstack((lams_g.ravel(), zs))
    mylf = lf.GLangFirsov(None, h0=0.0, h1=hcore, h2=eri, h_ep=g_g, w_p=w_p_arr,
                          nelec=1, uniform=False, spin=1)
    h0_g, h1_g, h2_g, h_ep_g, w_p = mylf.get_lf_ham(params_g)

    h0_diff = abs(h0_g - h0)
    print ("h0 diff")
    print (h0_diff)
    assert h0_diff < 1e-10

    h_ep_diff = la.norm(h_ep - h_ep_g)
    print ("h_ep diff")
    print (h_ep_diff)
    assert h_ep_diff < 1e-10

    h1_diff = la.norm(h1 - h1_g)
    print ("h1 diff")
    print (h1_diff)
    assert h1_diff < 1e-10

    h2_diff = max_abs(h2 - h2_g)
    print ("h2 diff")
    print (h2_diff)
    assert h2_diff < 1e-10

def test_glf():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf

    from pyscf import gto

    np.random.seed(10086)
    mol = gto.Mole(atom=[['H', [0.0, 0.0, 0.0]],
                         ['H', [0.0, 0.0, 1.0]]],
                   basis='321g')
    mol.build()

    nao = mol.nao_nr()
    nelec = mol.nelectron
    nmode = 1

    w_p_arr = np.ones(nmode) * 0.5
    h_ep = np.zeros((nmode, nao))
    h_ep[range(nmode), range(nao)] = 0.5

    mylf = lf.GLF(mol, h_ep=h_ep, w_p=w_p_arr,
                  nelec=nelec, uniform=False)

    params = np.asarray([0.94886041, 0.62512691, 0.14215147, 0.3223859, 0.55071256])

    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep_new, w_p = mylf.get_lf_ham(params)

    # symmetry of the new Hamiltonian
    perm_symm = la.norm(h2 - h2.transpose(1, 0, 2, 3))
    hermi_symm = la.norm(h2 - h2.transpose(2, 3, 0, 1))
    print ("perm symmetry")
    print (perm_symm)
    print ("hermi symmetry")
    print (hermi_symm)
    assert hermi_symm < 1e-10

    from pyscf import fci
    #cisolver = fci.fci_dhf_slow.FCI()
    cisolver = fci.direct_nosym.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol = 1e-8
    e, fcivec = cisolver.kernel(h1, h2, nao, 2)
    print (e)
    assert abs(e - -3.593181172677882) < 1e-7

    # GGLF
    h_ep = np.zeros((nmode, nao, nao))
    h_ep[:, range(nao), range(nao)] = 0.5
    mylf = lf.GGLF(mol, h_ep=h_ep, w_p=w_p_arr,
                   nelec=nelec, uniform=False)

    lams_g = np.zeros((nmode, nao, nao))
    lams_g[:, range(nao), range(nao)] = lams
    params = mylf.pack_params(lams_g, zs)
    H0, H1, H2, H_ep, w_p = mylf.get_lf_ham(params)

    assert abs(H0 - h0) < 1e-10
    assert la.norm(H1 - h1) < 1e-10
    assert la.norm(H2 - h2) < 1e-10
    assert la.norm(H_ep - h_ep_new) < 1e-10

    # symmetry of the new Hamiltonian
    perm_symm = la.norm(H2 - H2.transpose(1, 0, 2, 3))
    hermi_symm = la.norm(H2 - H2.transpose(2, 3, 0, 1))
    print ("perm symmetry")
    print (perm_symm)
    print ("hermi symmetry")
    print (hermi_symm)
    assert hermi_symm < 1e-10

    from pyscf import fci
    #cisolver = fci.fci_dhf_slow.FCI()
    cisolver = fci.direct_nosym.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol = 1e-8
    e, fcivec = cisolver.kernel(H1, H2, nao, 2)
    print (e)
    assert abs(e - -3.593181172677882) < 1e-7

def test_grad_lf():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import lang_firsov as lf

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)
    nao = 4
    nmode = nao
    nelec = 4
    w_p = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * w_p)

    hcore = np.random.random((nao, nao)) - 0.5
    hcore = hcore + hcore.conj().T

    params = (np.random.random(nao * 2) - 0.5)
    w_p_arr = np.ones(nmode) * w_p

    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)
    #eri = np.zeros((nao, nao, nao, nao))
    #eri[range(nao), range(nao), range(nao), range(nao)] = np.random.random(nao)

    mylf = lf.LangFirsov(h0=0.0, h1=hcore, h2=eri, h_ep=g, w_p=w_p_arr,
                         nelec=nelec, uniform=False)
    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep, w_p = mylf.get_lf_ham(params)

    e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=0)

    grad = mylf.get_grad(params, rdm1=rdm1)
    grad_num = np.empty_like(grad)

    dp = 1e-5
    e_ref = mylf.solve_lf_ham(params=params, dm0=rdm1, scf_max_cycle=0)[0]
    for i in range(len(params)):
        tmp = np.array(params, copy=True)
        tmp[i] += dp
        e = mylf.solve_lf_ham(params=tmp, dm0=rdm1, scf_max_cycle=0)[0]
        grad_num[i] = (e - e_ref) / dp

    print ("grad")
    print (grad)
    print ("grad numerical")
    print (grad_num)

    diff_grad = la.norm(grad - grad_num)
    print ("diff")
    print (diff_grad)
    assert diff_grad < 1e-4

def test_grad_glf():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import lang_firsov as lf

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)
    nao = 4
    nmode = nao
    nelec = 4
    w_p = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * w_p)

    hcore = np.random.random((nao, nao)) - 0.5
    hcore = hcore + hcore.conj().T

    params = (np.random.random(nmode*nao + nmode) - 0.5)
    w_p_arr = np.ones(nmode) * w_p

    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)
    #eri = np.zeros((nao, nao, nao, nao))
    #eri[range(nao), range(nao), range(nao), range(nao)] = np.random.random(nao)
    #eri = None
    g_g = np.zeros((nmode, nao))
    #g_g[range(nmode), range(nao)] = g
    g_g[range(nmode), range(nao)] = np.random.random(len(g_g[range(nmode), range(nao)]))

    mylf = lf.GLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=g_g, w_p=w_p_arr,
                          nelec=nelec, uniform=False)
    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep, w_p = mylf.get_lf_ham(params)

    e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=0)

    grad = mylf.get_grad(params, rdm1=rdm1)
    grad_num = np.empty_like(grad)

    dp = 1e-6
    e_ref = mylf.solve_lf_ham(params=params, dm0=rdm1, scf_max_cycle=0)[0]
    for i in range(len(params)):
        tmp = np.array(params, copy=True)
        tmp[i] += dp
        e = mylf.solve_lf_ham(params=tmp, dm0=rdm1, scf_max_cycle=0)[0]
        grad_num[i] = (e - e_ref) / dp

    print ("grad")
    print (grad)
    print ("grad numerical")
    print (grad_num)

    diff_grad = la.norm(grad - grad_num)
    print ("diff")
    print (diff_grad)
    assert diff_grad < 1e-4

def test_grad_glf_2():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import lang_firsov as lf

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)
    nao = 4
    nmode = nao
    nelec = 4
    w_p = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * w_p)

    hcore = np.random.random((nao, nao)) - 0.5
    hcore = hcore + hcore.conj().T

    params = (np.random.random(nmode*nao + nmode) - 0.5)
    w_p_arr = np.ones(nmode) * w_p

    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)
    #eri = np.zeros((nao, nao, nao, nao))
    #eri[range(nao), range(nao), range(nao), range(nao)] = np.random.random(nao)
    #eri[range(nao), range(nao), range(nao), range(nao)] = 5.0
    #eri = None
    #g_g = np.zeros((nmode, nao))
    #g_g[range(nmode), range(nao)] = np.random.random(len(g_g[range(nmode), range(nao)]))
    g_g = np.random.random((nmode, nao, nao))
    g_g = (g_g + g_g.transpose(0, 2, 1)) * 0.5

    mylf = lf.GLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=g_g, w_p=w_p_arr,
                          nelec=nelec, uniform=False)
    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep, w_p = mylf.get_lf_ham(params)

    e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=0)

    grad = mylf.get_grad(params, rdm1=rdm1)
    grad_num = np.empty_like(grad)

    dp = 1e-6
    e_ref = mylf.solve_lf_ham(params=params, dm0=rdm1, scf_max_cycle=0)[0]

    for i in range(len(params)):
        tmp = np.array(params, copy=True)
        tmp[i] += dp
        e = mylf.solve_lf_ham(params=tmp, dm0=rdm1, scf_max_cycle=0)[0]
        grad_num[i] = (e - e_ref) / dp

    print ("grad")
    print (grad)
    print ("grad numerical")
    print (grad_num)

    diff_grad = la.norm(grad - grad_num)
    print ("diff")
    print (diff_grad)
    assert diff_grad < 1e-4

def test_grad_gglf():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import lang_firsov as lf

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)
    nao = 4
    nmode = nao
    nelec = 4
    w_p = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * w_p)

    hcore = np.random.random((nao, nao)) - 0.5
    hcore = hcore + hcore.conj().T

    params = (np.random.random(nmode*nao*(nao+1)//2 + nmode) - 0.5)
    w_p_arr = np.ones(nmode) * w_p

    eri = np.random.random((nao, nao, nao, nao))
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(2, 3, 0, 1)
    #eri = np.zeros((nao, nao, nao, nao))
    #eri[range(nao), range(nao), range(nao), range(nao)] = np.random.random(nao)
    #eri[range(nao), range(nao), range(nao), range(nao)] = 5.0
    #eri = None
    #g_g = np.zeros((nmode, nao))
    #g_g[range(nmode), range(nao)] = np.random.random(len(g_g[range(nmode), range(nao)]))
    g_g = np.random.random((nmode, nao, nao))
    g_g = (g_g + g_g.transpose(0, 2, 1)) * 0.5

    mylf = lf.GGLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=g_g, w_p=w_p_arr,
                           nelec=nelec, uniform=False)
    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep, w_p = mylf.get_lf_ham(params)

    e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=0)

    return

    grad = mylf.get_grad(params, rdm1=rdm1)
    grad_num = np.empty_like(grad)

    dp = 1e-6
    e_ref = mylf.solve_lf_ham(params=params, dm0=rdm1, scf_max_cycle=0)[0]

    for i in range(len(params)):
        tmp = np.array(params, copy=True)
        tmp[i] += dp
        e = mylf.solve_lf_ham(params=tmp, dm0=rdm1, scf_max_cycle=0)[0]
        grad_num[i] = (e - e_ref) / dp

    print ("grad")
    print (grad)
    print ("grad numerical")
    print (grad_num)

    diff_grad = la.norm(grad - grad_num)
    print ("diff")
    print (diff_grad)
    assert diff_grad < 1e-4

if __name__ == "__main__":
    test_glf()
    test_grad_glf_2()
    test_grad_glf()
    test_grad_lf()
    test_lf_ham()
    test_grad_gglf()
