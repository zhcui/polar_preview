#!/usr/bin/env python

def test_m_mat():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian

    from polar.lang_firsov import lang_firsov as lf
    from polar.lang_firsov import mp as lfmp

    nao = 4
    nmode = nao
    nelec = 1
    w_p = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * w_p)

    hcore = np.random.random((nao, nao)) - 0.5
    hcore = hcore + hcore.conj().T

    w_p_arr = np.zeros(nmode)
    w_p_arr[:] = w_p

    params = (np.random.random(nao * 2) - 0.5)

    mylf = lf.LangFirsov(h0=0.0, h1=hcore, h2=None, h_ep=g, w_p=w_p_arr,
                         nelec=1, uniform=False)
    lams, zs = mylf.unpack_params(params)
    h0, h1, h2, h_ep, w_p = mylf.get_lf_ham(params)

    mo_energy, mo_coeff = la.eigh(hcore)
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1.0
    nocc = nelec
    nmo = mo_coeff.shape[-1]

    h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep, mo_coeff.conj(), mo_coeff, optimize=True)
    psi1 = (0, 0, list(0 for x in range(nmode)))

    m_mat = lfmp.get_m_mat_hf_ij(hcore, h_ep, w_p_arr, lams, zs, mo_coeff[:, :nocc], 1, 1)
    print ("CHECK ij 11 ")
    m_mat_re = np.zeros_like(m_mat)
    for i in range(nmode):
        for j in range(nmode):
            if i != j:
                ph_str = np.zeros(nmode, dtype=int)
                ph_str[i] = 1
                ph_str[j] = 1
                m_mat_re[i, j] = lfmp.compute_ovlp(psi1, (0, 0, ph_str), hcore, h_ep_mo, w_p_arr,
                                                   lams, zs, mo_coeff, mo_occ, nocc)

    print (m_mat)
    print (m_mat_re)
    print ("norm", la.norm(m_mat - m_mat_re))
    assert la.norm(m_mat - m_mat_re) < 1e-10


    m_mat = lfmp.get_m_mat_hf_ij(hcore, h_ep, w_p_arr, lams, zs, mo_coeff[:, :nocc], 1, 2)
    print ("CHECK ij 12 ")
    m_mat_re = np.zeros_like(m_mat)
    for i in range(nmode):
        for j in range(nmode):
            if i != j:
                ph_str = np.zeros(nmode, dtype=int)
                ph_str[i] = 1
                ph_str[j] = 2
                m_mat_re[i, j] = lfmp.compute_ovlp(psi1, (0, 0, ph_str), hcore, h_ep_mo, w_p_arr,
                                                   lams, zs, mo_coeff, mo_occ, nocc)

    print ("norm", la.norm(m_mat - m_mat_re))
    assert la.norm(m_mat - m_mat_re) < 1e-10

    print ("check HF x i")
    m_mat = lfmp.get_m_mat_hf_i(hcore, h_ep, w_p_arr, lams, zs, mo_coeff[:, :nocc], 2)
    m_mat_re = np.zeros_like(m_mat)
    for i in range(nmode):
        ph_str = np.zeros(nmode)
        ph_str[i] = 2
        m_mat_re[i] = lfmp.compute_ovlp(psi1, (0, 0, ph_str), hcore, h_ep_mo, w_p_arr, lams, zs,
                                        mo_coeff, mo_occ, nocc)

    print (m_mat)
    print (m_mat_re)
    print ("norm", la.norm(m_mat - m_mat_re))
    assert la.norm(m_mat - m_mat_re) < 1e-10

    print ("check ia x i")
    m_mat = lfmp.get_m_mat_ia_i(hcore, h_ep, w_p_arr, lams, zs, mo_coeff[:, :nocc], mo_coeff[:, nocc:], 1)
    m_mat_re = np.zeros_like(m_mat)
    for i in range(nmode):
        for I in range(nocc):
            for iJ, J in enumerate(range(nocc, nmo)):
                ph_str = np.zeros(nmode)
                ph_str[i] = 1
                m_mat_re[i, I, iJ] = lfmp.compute_ovlp(psi1, (I, J, ph_str), hcore, h_ep_mo, w_p_arr, lams, zs, mo_coeff,
                        mo_occ, nocc)

    print ("norm", la.norm(m_mat - m_mat_re))
    assert la.norm(m_mat - m_mat_re) < 1e-10

    print ("check ia x i_1j_1")
    m_mat = lfmp.get_m_mat_ia_ij(hcore, h_ep, w_p_arr, lams, zs, mo_coeff[:, :nocc], mo_coeff[:, nocc:], 1, 1)
    m_mat_re = np.zeros_like(m_mat)
    for i in range(nmode):
        for j in range(nmode):
            if i != j:
                for I in range(nocc):
                    for iJ, J in enumerate(range(nocc, nmo)):
                        ph_str = np.zeros(nmode)
                        ph_str[i] = 1
                        ph_str[j] = 1
                        m_mat_re[i, j, I, iJ] = lfmp.compute_ovlp(psi1, (I, J, ph_str), hcore, h_ep_mo, w_p_arr, lams, zs,
                                mo_coeff, mo_occ, nocc)

    print ("norm", la.norm(m_mat - m_mat_re))
    assert la.norm(m_mat - m_mat_re) < 1e-10

def test_mp2():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf
    from polar.lang_firsov import mp as lfmp

    nao = 4
    nph = 9
    latt_size = nao
    imp_size = nao
    nelec = 1
    w_p = 0.5
    w_p_arr = np.zeros(nao)
    w_p_arr[:] = w_p
    alpha = 2.4
    g = np.sqrt(alpha * w_p)
    U = 0.0
    uniform = False
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1

    latt = lattice.ChainLattice(latt_size, imp_size)
    params = (np.random.random(nao * 2) - 0.5)
    lams, zs = lfmp.unpack_params(params, uniform=uniform, nmode=nao)

    hcore_bare = np.random.random((nao, nao)) - 0.5
    hcore_bare = hcore_bare + hcore_bare.conj().T

    mylf = lf.LangFirsov(h0=0.0, h1=hcore_bare, h2=None, h_ep=g, w_p=w_p_arr,
                         nelec=1, uniform=False)
    e_mp2_re = mylf.solve_lf_ham(params, mp2=True, verbose=True)[0]

    # from definition
    h0, hcore, h2, h_ep, w_p = mylf.get_lf_ham(params)
    w_p_arr = np.ones(nao) * w_p
    mo_energy, mo_coeff = la.eigh(hcore)
    e_fci = np.sum(mo_energy[:nelec]) + h0
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1.0
    rdm1 = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
    conv = True

    nocc = nelec
    nmo = mo_energy.shape[-1]
    nmode = len(h_ep)

    states = lfmp.gen_states(nmo, nocc, nmode, nph)[1:]
    psi1 = [0, 0, tuple(0 for x in range(nmode))]
    e_mp2 = e_fci

    h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep, mo_coeff.conj(), mo_coeff, optimize=True)
    for i, state in enumerate(states):
        de = lfmp.compute_de(state, w_p_arr, mo_energy)
        ovlp = lfmp.compute_ovlp(psi1, state, hcore_bare, h_ep_mo, w_p_arr, lams, zs, mo_coeff, mo_occ, nocc)
        e_mp2 -= ovlp**2 / de

    print (e_mp2)
    print (e_mp2_re)
    print ("diff", abs(e_mp2 - e_mp2_re))
    assert abs(e_mp2 - e_mp2_re) < 1e-12


def test_pt2():
    import numpy as np
    from scipy import linalg as la
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf
    from polar.lang_firsov import mp as lfmp

    nao = 2
    nph = 9
    latt_size = nao
    imp_size = nao
    nelec = 1
    w_p = 0.5
    w_p_arr = np.zeros(nao)
    w_p_arr[:] = w_p
    alpha = 2.4
    g = np.sqrt(alpha * w_p)
    U = 0.0
    uniform = False
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1

    latt = lattice.ChainLattice(latt_size, imp_size)
    params = (np.random.random(nao * 2) - 0.5)
    lams, zs = lfmp.unpack_params(params, uniform=uniform, nmode=nao)

    hcore = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore[i,i+1] = hcore[i+1,i] = -1.0
    hcore[nao-1, 0] = hcore[0, nao-1] = -1.0  # PBC

    mylf = lf.LangFirsov(h0=0.0, h1=hcore, h2=None, h_ep=g, w_p=w_p_arr,
                         nelec=1, uniform=False)
    e_mp2_re = mylf.solve_lf_ham(params, mp2=True, verbose=True)[0]

def test_gen_states():
    from polar.lang_firsov import mp as lfmp

    nph = 4
    nmode = 2
    nocc = 2
    nmo = 5
    states_G = lfmp.gen_states_G(nmo, nocc, nmode, nph)
    states_S = lfmp.gen_states_S(nmo, nocc, nmode, nph)
    states_D = lfmp.gen_states_D(nmo, nocc, nmode, nph)

    for st in states_D:
        print (st)

if __name__ == "__main__":
    test_mp2()
    test_gen_states()
    test_pt2()
    test_m_mat()
