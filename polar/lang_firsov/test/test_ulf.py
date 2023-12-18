#!/usr/bin/env python

def test_grad_uglf():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import ulf as lf

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

    rdm1[1] = np.eye(4) * 0.5

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

    rdm1 = mylf.make_rdm1()
    rdm1e = mylf.make_rdm1e(lams=lams)

def test_grad_uglf_2():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from libdmet.utils import max_abs
    from polar.lang_firsov import ulf as lf

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

    rdm1[1] = np.eye(nao) - rdm1[1]
    print (rdm1)

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
    test_grad_uglf_2()
    test_grad_uglf()
