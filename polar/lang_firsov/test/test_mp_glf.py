#!/usr/bin/env python

from mpi4pyscf.tools import mpi
import numpy as np
from scipy import linalg as la

def test_mp2_glf():
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf
    from polar.lang_firsov import mp as lfmp

    np.random.seed(1)

    nao = 4
    nph = 3
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
    e_mp2_re = mylf.solve_lf_ham(params, mp2=True, verbose=True, nph=nph)[0]
    e_hf = mylf.e_hf

    # from definition
    h0, hcore, h2, h_ep_, w_p = mylf.get_lf_ham(params)
    w_p_arr = np.ones(nao) * w_p
    mo_energy, mo_coeff = la.eigh(hcore)
    e_fci = np.sum(mo_energy[:nelec]) + h0
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1.0
    rdm1 = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
    conv = True

    nocc = nelec
    nmo = mo_energy.shape[-1]
    nmode = len(h_ep_)
    h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep_, mo_coeff.conj(), mo_coeff, optimize=True)

    # GLF
    nmode = nao
    lams_g = np.diag(lams)
    g_g = np.zeros((nmode, nao, nao))
    g_g[range(nmode), range(nao), range(nao)] = g
    w_p_g = np.zeros((nmode))
    w_p_g[:] = w_p

    eri = None
    print ("test overlap between states")

    lams_old = lams
    lams = lams_g

    w_p = w_p_arr
    h_ep = g_g

    from pyscf import lib
    from polar.fci.fci import fc_factor

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    f00 = fc_factor(0, 0, diff)
    diff = None
    fac1 = np.prod(f00, axis=0)

    h1 = hcore_bare + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
    v56 = np.einsum('xpq, xq -> pq', h_ep, lams)
    h1 -= v56
    h1 -= v56.conj().T

    v56 = None
    h_ep_exp = h_ep

    h_ep_linear = np.einsum("x, xp -> xp", -w_p, lams)
    h_ep_linear_mo = np.einsum("xp, pm, pn -> xmn", h_ep_linear, mo_coeff,
                               mo_coeff, optimize=True)

    states = lfmp.gen_states(nmo, nocc, nmode, nph)
    for psi1 in states:
        for psi2 in states:
            ovlp_ref = lfmp.compute_ovlp(psi1, psi2, hcore_bare, h_ep_mo, w_p_arr,
                                         lams_old, zs, mo_coeff, mo_occ, nocc)
            ovlp = lfmp.compute_ovlp_g(psi1, psi2, h1, h_ep_exp, h_ep_linear_mo,
                                       fac1=fac1,
                                       w_p=w_p, lams=lams, zs=zs,
                                       mo_coeff=mo_coeff, mo_occ=mo_occ)
            print (abs(ovlp - ovlp_ref))
            assert abs(ovlp - ovlp_ref) < 1e-13

    e_mp2 = 0.0
    for i, state in enumerate(states[1:]):
        de = lfmp.compute_de(state, w_p_arr, mo_energy)
        ovlp = lfmp.compute_ovlp_g(states[0], state, h1, h_ep_exp, h_ep_linear_mo,
                                   fac1=fac1,
                                   w_p=w_p, lams=lams, zs=zs,
                                   mo_coeff=mo_coeff, mo_occ=mo_occ)
        e_mp2 -= ovlp**2 / de
        print (state, de, ovlp)
    e_mp2_ref = e_mp2_re - e_hf

    print ("e_mp2")
    print (e_mp2)
    print ("e_mp2 re")
    print (e_mp2_ref)
    print ("diff")
    print (abs(e_mp2 - e_mp2_ref))
    assert abs(e_mp2 - e_mp2_ref) < 1e-12

def test_reproduce_mp2():
    from pyscf import gto, scf, mp, ao2mo
    from polar.lang_firsov import mp_glf as lfmp

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '321g')

    mf = mol.RHF().run()
    hcore = mf.get_hcore()

    fock = mf.get_fock()

    mf = mf.to_ghf()

    mo_occ = mf.mo_occ
    mo_energy = ew = mf.mo_energy
    mo_coeff = ev = mf.mo_coeff
    nocc = np.sum(mo_occ >= 0.5)
    nso, nmo = mo_coeff.shape
    nao = nso // 2


    nvir = nmo - nocc
    print ("nmo nocc nvir")
    print (nmo, nocc, nvir)

    mymp = mf.MP2().run()

    nmode = 1
    nph = 1

    states_G = lfmp.gen_states_G(nmo, nocc, nmode=2, nph=3)
    states_S = lfmp.gen_states_S(nmo, nocc, nmode, nph)
    states_D = lfmp.gen_states_D(nmo, nocc, nmode, nph)

    assert len(states_G) == 6
    assert len(states_S) == nocc * nvir
    assert len(states_D) == nocc * (nocc - 1) // 2 * nvir * (nvir - 1) // 2

    states = lfmp.gen_states(nmo, nocc, nmode, nph)
    dic_st2idx = dict(zip(states, range(len(states))))

    for key, val in dic_st2idx.items():
        idx_re = lfmp.get_config_idx(key[:-1], nocc, nvir)
        print (val, key)
        assert idx_re == val

    # construct the CISD hamiltonian matrix
    myci = mf.CISD()
    eris = myci.ao2mo()
    nconfig = lfmp.get_nconfig(nocc, nvir)

    ket = np.zeros(nconfig)
    ket[0] = 1.0
    ket = myci.contract(ket, eris)
    diag = myci.make_diagonal(eris)[0]
    ket[0] += diag

    e_mp2 = 0.0
    for i, state in enumerate(states):
        if len(state) == 1:
            continue
        elif len(state) == 3:
            I, A, ph_str = state
            idx = lfmp.get_config_idx((I, A), nocc, nvir)
            M = ket[idx]
            e_mp2 -= M**2 / (ew[A] - ew[I])
        elif len(state) == 5:
            I, J, A, B, ph_str = state
            idx = lfmp.get_config_idx((I, J, A, B), nocc, nvir)
            M = ket[idx]
            e_mp2 -= M**2 / (ew[A] + ew[B] - ew[I] - ew[J])
        else:
            raise ValueError

    print ("e_mp2 from definition")
    print (e_mp2)
    print ("diff", abs(e_mp2 - mymp.e_corr))
    assert abs(e_mp2 - mymp.e_corr) < 1e-10

    eri = ao2mo.restore(1, mf._eri, nao)
    e_mp2 = 0.0
    for i, state in enumerate(states):
        if len(state) == 1:
            continue
        elif len(state) == 3:
            I, A, ph_str = state
            M = lfmp.sc_rule_h1_0X(hcore - fock, mo_coeff, state, nocc=nocc,
                                   ao_repr=True)
            M += lfmp.sc_rule_h2_0X(eri, mo_coeff, state, nocc=nocc,
                                    ao_repr=True)
            e_mp2 -= M**2 / (ew[A] - ew[I])
        elif len(state) == 5:
            I, J, A, B, ph_str = state
            M = lfmp.sc_rule_h1_0X(hcore - fock, mo_coeff, state, nocc=nocc,
                                   ao_repr=True)
            M += lfmp.sc_rule_h2_0X(eri, mo_coeff, state, nocc=nocc,
                                    ao_repr=True)
            e_mp2 -= M**2 / (ew[A] + ew[B] - ew[I] - ew[J])
        else:
            raise ValueError

    print ("e_mp2 from definition 3 (rhf integral ghf mo)")
    print (e_mp2)
    print ("diff", abs(e_mp2 - mymp.e_corr))
    assert abs(e_mp2 - mymp.e_corr) < 1e-10

def test_reproduce_mp2_2():
    from polar.lang_firsov import lang_firsov as lf
    from polar.system import molecule as mole
    from pyscf import gto, scf, ao2mo
    from pyscf import lo
    from libdmet.basis_transform import make_basis

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)

    shift = 0.0
    mol = gto.Mole(atom=[['H', [0, 0, 0.0 - shift]],
                         ['F', [0, 0, 0.918 - shift]]],
                   basis='321G',
                   unit='A',
                   verbose=4)
    mol.build()

    C = lo.orth_ao(mol)
    w = 0.466751
    lc = 0.0

    nao = mol.nao_nr()
    nmode = 1
    nelec = mol.nelectron

    mf = scf.HF(mol)
    mf.kernel()
    mf.analyze(unit='AU')
    rdm1 = mf.make_rdm1()
    print ("HF energy", mf.e_tot)

    gmf = mf.to_ghf()

    mymp = gmf.MP2()
    e_mp2, t2 = mymp.kernel()
    ao_repr = True
    rdm1_ref = mymp.make_rdm1(ao_repr=ao_repr)
    print ("MP2 energy", e_mp2)

    # e-ph
    dip_op = mole.get_dip_op(mol)

    w_p = np.array([w])
    h_c = lc * dip_op[2]

    h_ep = -np.sqrt(w * 0.5) * h_c
    h_ep = h_ep[None]

    H0 = mf.energy_nuc()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = ao2mo.restore(1, mf._eri, nao)

    # transform everything to orthogonal basis
    hcore = make_basis.transform_h1_to_lo_mol(hcore, C)
    ovlp_lo = make_basis.transform_h1_to_lo_mol(ovlp, C)
    eri = ao2mo.kernel(eri, C)
    h_ep = np.einsum("xpq, pm, qn -> xmn", h_ep, C, C, optimize=True)
    h_c = np.einsum('pq, pm, qn -> mn', h_c, C, C, optimize=True)
    hcore += 0.5 * np.einsum('pq, qs -> ps', h_c, h_c)
    eri -= np.einsum('pq, rs -> pqrs', h_c, h_c)

    mo_coeff = C.conj().T @ ovlp @ mf.mo_coeff
    mo_occ = mf.mo_occ

    mylf = lf.GLangFirsov(mol=mol, h0=H0, h1=hcore, h2=eri, ovlp=ovlp_lo,
                          h_ep=h_ep, w_p=w_p,
                          nelec=nelec, uniform=False)

    params = (np.random.random(mylf.nparam_full) - 0.5) #* 0.0

    e = mylf.kernel(params=params,
                    use_num_grad=False, full_opt=True,
                    mo_coeff=mo_coeff, mo_occ=mo_occ, conv_tol=1e-8,
                    max_cycle=200, ntrial=3, mp2=True, nph=3)

    print ("GLF energy")
    print (e)

    mo_occ = mylf._scf.mo_occ
    mo_energy = mylf._scf.mo_energy
    mo_coeff = mylf._scf.mo_coeff

    rdm1 = mylf._scf.make_rdm1()
    vhf = mylf._scf.get_veff(dm=rdm1)
    vhf = vhf[:nao, :nao]

    params_opt = mylf.params_full_opt
    kappa, lams, zs = mylf.unpack_params_full(params_opt)

    nso, nmo = mo_coeff.shape
    nao = nso // 2
    nocc = np.sum(mo_occ > 0.5)

    from polar.lang_firsov import mp_glf as lfmp
    nph = 3
    e_mp2_re, rdm1 = lfmp.get_e_mp2(mylf, nph=nph, make_rdm1=True, ao_repr=ao_repr)
    print ("MP2 from e-ph ham")
    print (e_mp2_re)

    print ("diff MP2")
    print (abs(e_mp2 - e_mp2_re))
    assert abs(e_mp2 - e_mp2_re) < 1e-7

    print ("rdm1")
    if ao_repr:
        C = la.block_diag(C, C)
        rdm1 = np.einsum("pm, mn, qn -> pq", C, rdm1, C)
    print (rdm1)
    print (rdm1_ref)
    diff = la.norm(rdm1 - rdm1_ref)
    print ("rdm1 diff")
    print (diff)
    assert diff < 1e-7

def test_mp2_glf_2():
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf
    from polar.lang_firsov import mp as lfmp
    from polar.lang_firsov import mp_glf as lfmp_g

    np.random.seed(1)
    np.set_printoptions(3, linewidth=1000, suppress=True)

    nao = 4
    nph = 5
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
    e_mp2_re = mylf.solve_lf_ham(params, mp2=True, verbose=True, nph=nph)[0]
    e_hf = mylf.e_hf
    e_mp2_ref = e_mp2_re - e_hf

    # from definition
    print ("from definition 1")
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
        print (state, de, ovlp)

    print (e_mp2)
    print (e_mp2_re)
    print ("diff", abs(e_mp2 - e_mp2_re))
    assert abs(e_mp2 - e_mp2_re) < 1e-12

    # GLF
    print ("GLF + MP2")
    nmode = nao
    lams_g = np.diag(lams)
    g_g = np.zeros((nmode, nao, nao))
    g_g[range(nmode), range(nao), range(nao)] = g
    w_p_g = np.zeros((nmode))
    w_p_g[:] = w_p

    lams = lams_g
    w_p = w_p_g
    h_ep = g_g
    hcore = hcore_bare
    eri = None

    from pyscf import lib
    from polar.fci.fci import fc_factor

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    f00 = fc_factor(0, 0, diff)
    diff = None
    fac1 = np.prod(f00, axis=0)

    states = lfmp_g.gen_states(nmo, nocc, nmode, nph)

    # prepare integrals
    h_coh = w_p * zs

    heff  = hcore_bare + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
    heff -= np.einsum('xp, xpq -> pq', lams, h_ep)
    heff -= np.einsum('xpq, xq -> pq', h_ep, lams)

    h1  = -heff * fac1
    #h1 -= vhf

    h_ep_linear = np.zeros_like(h_ep)
    h_ep_linear[:, range(nao), range(nao)] = np.einsum('x, xp -> xp', -w_p, lams)

    h_ep_exp = h_ep
    h2 = h2_exp_1 = h2_exp_2 = None

    e_mp2 = 0.0
    for i, state in enumerate(states[1:]):
        de = lfmp_g.compute_de(state, w_p_arr, mo_energy)
        ovlp = lfmp_g.compute_ovlp(states[0], state, mo_coeff, mo_occ, lams,
                                   w_p, h_coh, h1, heff, h_ep_linear, h_ep_exp,
                                   h2, h2_exp_1, h2_exp_2)
        print (state, de, ovlp)
        e_mp2 -= ovlp**2 / de

    print ("e_mp2")
    print (e_mp2)
    print ("e_mp2 re")
    print (e_mp2_ref)
    print ("diff")
    print (abs(e_mp2 - e_mp2_ref))
    assert abs(e_mp2 - e_mp2_ref) < 1e-12

def test_mp2_glf_3():
    """
    test mp2 glf for 1e problem.
    from 2 different implementations.
    """
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf
    from polar.lang_firsov import mp as lfmp
    from polar.lang_firsov import mp_glf as lfmp_g

    np.random.seed(1)
    np.set_printoptions(3, linewidth=1000, suppress=True)

    """
    First check LF and GLF (not full) gives the same MP2 energy.
    here g is constant.
    """

    nao = 4
    nmode = nao
    nph = 5
    latt_size = nao
    imp_size = nao
    nelec = 1
    w_p = 0.5
    w_p_arr = np.zeros(nao)
    w_p_arr[:] = w_p
    alpha = 2.4
    g = np.sqrt(alpha * w_p)
    h_ep = g
    U = 0.0
    uniform = False
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1

    latt = lattice.ChainLattice(latt_size, imp_size)
    #params = (np.random.random(nao * 2) - 0.5)

    #hcore_bare = np.random.random((nao, nao)) - 0.5
    #hcore_bare = hcore_bare + hcore_bare.conj().T
    hcore_bare = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore_bare[i, i+1] = -1
        hcore_bare[i+1, i] = -1
    hcore_bare[0, nao-1] = -1
    hcore_bare[nao-1, 0] = -1

    mylf = lf.LangFirsov(mol=None, h0=0.0, h1=hcore_bare, h2=None, h_ep=h_ep, w_p=w_p_arr,
                         nelec=1, uniform=False)
    e_mp2_re = mylf.solve_lf_ham(params=None, mp2=True, verbose=True, nph=nph)[0]
    e_mp2_re_slow = mylf.solve_lf_ham(params=None, mp2='slow', verbose=True, nph=nph)[0]
    #e_hf = mylf.e_hf
    #e_mp2_ref = e_mp2_re - e_hf

    print ("LF and LF slow diff")
    print (abs(e_mp2_re - e_mp2_re_slow))
    assert abs(e_mp2_re - e_mp2_re_slow) < 1e-12
    params = mylf.params
    lams, zs = mylf.unpack_params(params)

    lams_g = np.diag(lams)
    h_ep = np.zeros((nmode, nao, nao))
    h_ep[range(nmode), range(nao), range(nao)] = g

    mylf = lf.GLangFirsov(mol=None, h0=0.0, h1=hcore_bare, h2=None, h_ep=h_ep, w_p=w_p_arr,
                          nelec=1, uniform=False)
    params_g = mylf.pack_params(lams_g, zs)

    e_mp2_re_g1 = mylf.solve_lf_ham(params=params_g, mp2='slow', verbose=True, nph=nph)[0]
    #e_hf = mylf.e_hf
    #e_mp2_ref_g1 = e_mp2_re_g1 - e_hf

    print ("LF and GLF slow diff")
    print (abs(e_mp2_re - e_mp2_re_g1))
    assert abs(e_mp2_re - e_mp2_re_g1) < 1e-12

    """
    Then check GLF and GLF (full) gives the same MP2 energy.
    here g is a tensor.
    """

    mo_coeff = mylf.mo_coeff

    eri = np.zeros((nao, nao, nao, nao))
    eri[range(nao), range(nao), range(nao), range(nao)] = 0.0

    from polar.lang_firsov import ulf as lf
    mylf = lf.GLangFirsov(mol=None, h0=0.0, h1=hcore_bare, h2=eri, h_ep=h_ep, w_p=w_p_arr,
                          nelec=1, uniform=False)

    mo_coeff = np.asarray((mo_coeff, mo_coeff))
    mo_occ = np.zeros((2, nao))
    mo_occ[0, 0] = 1.0

    kappa_a = np.zeros((3,))

    params_full = np.append(kappa_a, params_g)

    mylf.solve_lf_ham_full(params=params_full, mp2=True, mp3=False, mp4=False,
                      nph=nph, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                      scf_max_cycle=50, mo_coeff=mo_coeff, mo_occ=mo_occ, canonicalization=True)

    e_hf = mylf.e_hf
    e_mp2_full = mylf.e_mp2 + e_hf
    print (e_mp2_re)
    print (e_mp2_full)
    print ("GLF and GLF full diff")
    print (abs(e_mp2_re - e_mp2_full))
    assert abs(e_mp2_re - e_mp2_full) < 1e-12

def test_mp2_glf_4():
    """
    test mp2 glf for 1e problem.
    from 2 different implementations.
    h_ep is diagonal. but lams is not.
    """
    from libdmet.system import lattice, hamiltonian
    from polar.lang_firsov import lang_firsov as lf
    #from polar.lang_firsov import mp as lfmp
    #from polar.lang_firsov import mp_glf as lfmp_g

    np.random.seed(1)
    np.set_printoptions(3, linewidth=1000, suppress=True)

    """
    First check LF and GLF (not full) gives the same MP2 energy.
    here g is constant.
    """

    nao = 2
    nmode = nao
    nph = 3
    latt_size = nao
    imp_size = nao
    nelec = 1
    w_p = 1.0
    w_p_arr = np.zeros(nao)
    w_p_arr[:] = w_p
    alpha = 2.4
    g = np.sqrt(alpha * w_p)
    U = 0.0
    uniform = False
    mo_occ = np.zeros(nao)
    mo_occ[:nelec] = 1

    latt = lattice.ChainLattice(latt_size, imp_size)
    #params = (np.random.random(nao * 2) - 0.5)

    #hcore_bare = np.random.random((nao, nao)) - 0.5
    #hcore_bare = hcore_bare + hcore_bare.conj().T
    hcore_bare = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore_bare[i, i+1] = -1
        hcore_bare[i+1, i] = -1
    hcore_bare[0, nao-1] = -1
    hcore_bare[nao-1, 0] = -1

    h_ep = np.zeros((nmode, nao, nao))
    h_ep[range(nmode), range(nao), range(nao)] = g

    #h_ep += 1e-3
    mylf = lf.GLangFirsov(mol=None, h0=0.0, h1=hcore_bare, h2=None, h_ep=h_ep, w_p=w_p_arr,
                          nelec=1, uniform=False)
    mylf.conv_tol = 1e-14
    mylf.kernel()

    params_g = mylf.params_opt

    lams_g, zs = mylf.unpack_params(params_g)
    #lams_g = np.diag(np.diag(lams_g))
    #params_g = mylf.pack_params(lams_g, zs)

    e_mp2_re_g1 = mylf.solve_lf_ham(params=params_g, mp2='slow', verbose=True, nph=nph)[0]
    #e_hf = mylf.e_hf
    #e_mp2_ref_g1 = e_mp2_re_g1 - e_hf

    #print ("LF and GLF slow diff")
    #print (abs(e_mp2_re - e_mp2_re_g1))
    #assert abs(e_mp2_re - e_mp2_re_g1) < 1e-12

    """
    Then check GLF and GLF (full) gives the same MP2 energy.
    here g is a tensor.
    """

    mo_coeff = mylf.mo_coeff

    eri = np.zeros((nao, nao, nao, nao))
    eri[range(nao), range(nao), range(nao), range(nao)] = 0.0

    #from polar.fci import fci
    #e_fci, civec = fci.kernel(hcore_bare, eri, nao, nelec, nmode, nph, h_ep, w_p_arr,
    #                      shift_vac=True, ecore=mylf.get_h0(), tol=1e-9, max_cycle=10000)
    #print ("ED", e_fci)

    from polar.lang_firsov import ulf as lf
    mylf = lf.GLangFirsov(mol=None, h0=0.0, h1=hcore_bare, h2=eri, h_ep=h_ep, w_p=w_p_arr,
                          nelec=1, uniform=False)
    mylf.conv_tol = 1e-14

    mo_coeff = np.asarray((mo_coeff, mo_coeff))
    mo_occ = np.zeros((2, nao))
    mo_occ[0, 0] = 1.0

    kappa_a = np.zeros(((nao-1)*1,))

    params_full = np.append(kappa_a, params_g)

    mylf.solve_lf_ham_full(params=params_full, mp2=True, mp3=False, mp4=True,
                      nph=nph, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                      scf_max_cycle=50, mo_coeff=mo_coeff, mo_occ=mo_occ, canonicalization=True)

    e_hf = mylf.e_hf
    #e_mp2_full = mylf.e_mp2 + e_hf
    #print (e_mp2_full)
    #print ("GLF and GLF full diff")
    #print (abs(e_mp2_re_g1 - e_mp2_full))
    #assert abs(e_mp2_re_g1 - e_mp2_full) < 1e-12

    from polar.lang_firsov import mp_glf
    e_mp2_ref = mp_glf.get_e_mp2_ref(mylf, lams=lams_g, zs=zs, nph=nph)

    print ("ref")
    print (e_mp2_ref)
    print (abs(e_mp2_ref + e_hf - e_mp2_re_g1))

if __name__ == "__main__":
    test_reproduce_mp2_2()
    test_mp2_glf_3()
    test_mp2_glf_4()
    test_mp2_glf()
    test_mp2_glf_2()
    test_reproduce_mp2()
