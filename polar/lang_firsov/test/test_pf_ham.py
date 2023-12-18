#!/usr/bin/env python

import pytest
import numpy as np

@pytest.mark.parametrize(
    "basis", ['631g']
)
@pytest.mark.parametrize(
    "w", [0.466751]
)
def test_pf_ham_H2(basis, w, lc=0.05, pol_axis=2):
    from scipy import linalg as la
    from polar.system import molecule as mole
    from polar.lang_firsov import lang_firsov as lf
    from pyscf import gto, scf, ao2mo
    from pyscf import lo
    from libdmet.basis_transform import make_basis

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)

    shift = 0.0
    mol = gto.Mole(atom=[['H', [0, 0, 0.0 - shift]],
                         ['H', [0, 0, 0.746 - shift]]],
                   #symmetry=True
                   #basis = '631g',
                   #basis='6311++g**',
                   basis=basis,
                   unit='A',
                   verbose=4,
                   )
    mol.build()

    C = lo.orth_ao(mol)
    #w_H2 = 0.466751
    #lc = 0.05

    nao = mol.nao_nr()
    nmode = 1
    nelec = mol.nelectron

    mf = scf.HF(mol)
    mf.kernel()
    mf.analyze(unit='AU')
    rdm1 = mf.make_rdm1()

    print ("HF energy", mf.e_tot)

    mymp = mf.MP2()
    e_mp2, t2 = mymp.kernel()

    print ("MP2 energy", e_mp2)

    mycc = mf.CCSD()

    e_cc, t1, t2 = mycc.kernel()

    print ("CCSD energy", e_cc)

    # e-ph

    dip_op = mole.get_dip_op(mol)
    #dip_val = np.einsum('xpq, qp -> x', dip_op, rdm1)

    w_p = np.array([w])
    h_c = lc * dip_op[pol_axis]

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

    # FCI
    print ("-" * 79)
    nph = 24
    from polar.fci import fci
    e_fci, civec = fci.kernel(hcore, eri, nao, nelec, nmode, nph, h_ep, w_p,
                          shift_vac=True, ecore=H0, tol=1e-9, max_cycle=10000)

    print ("ED", e_fci)

    mo_coeff = C.conj().T @ ovlp @ mf.mo_coeff
    mo_occ = mf.mo_occ

    mylf = lf.GGLangFirsov(mol=mol, h0=H0, h1=hcore, h2=eri, ovlp=ovlp_lo,
                           h_ep=h_ep, w_p=w_p,
                           nelec=nelec, uniform=False)

    params = (np.random.random(mylf.nparam_full) - 0.5) #* 0.0

    e = mylf.kernel(params=params,
                    use_num_grad=True, full_opt=True,
                    mo_coeff=mo_coeff, mo_occ=mo_occ, conv_tol=1e-8,
                    max_cycle=200, ntrial=3)

    rdm1 = mylf.make_rdm1()
    print (e)

@pytest.mark.parametrize(
    "basis", ['sto6g']
)
@pytest.mark.parametrize(
    "w", [0.531916]
)
def test_pf_ham_HF(basis, w, lc=0.05, pol_axis=2):
    from scipy import linalg as la
    from polar.system import molecule as mole
    from polar.lang_firsov import lang_firsov as lf
    from pyscf import gto, scf, ao2mo
    from pyscf import lo
    from libdmet.basis_transform import make_basis

    np.set_printoptions(3, linewidth=1000, suppress=True)
    np.random.seed(1)

    shift = 0.0
    mol = gto.Mole(atom=[['H', [0, 0, 0.0 - shift]],
                         ['F', [0, 0, 0.918 - shift]]],
                   #basis='6311++g**',
                   #basis='sto6g',
                   basis=basis,
                   unit='A',
                   verbose=4,
                   symmetry=True)

    mol.build()

    C = lo.orth_ao(mol)
    #w_H2 = 0.466751
    lc = 0.05
    #lc = 0.0

    nao = mol.nao_nr()
    nmode = 1
    nelec = mol.nelectron

    mf = scf.HF(mol)
    mf.kernel()
    mf.analyze(unit='AU')
    rdm1 = mf.make_rdm1()

    print ("HF energy", mf.e_tot)

    mymp = mf.MP2()
    e_mp2, t2 = mymp.kernel()

    print ("MP2 energy", e_mp2)

    mycc = mf.CCSD()

    e_cc, t1, t2 = mycc.kernel()

    print ("CCSD energy", e_cc)

    # e-ph

    dip_op = mole.get_dip_op(mol)
    #dip_val = np.einsum('xpq, qp -> x', dip_op, rdm1)

    w_p = np.array([w])
    h_c = lc * dip_op[pol_axis]

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
    eri   += np.einsum('pq, rs -> pqrs', h_c, h_c)

    # FCI
    print ("-" * 79)
    nph = 16
    from polar.fci import fci
    e_fci, civec = fci.kernel(hcore, eri, nao, nelec, nmode, nph, h_ep, w_p,
                          shift_vac=True, ecore=H0, tol=1e-9, max_cycle=10000)

    print ("ED", e_fci)

    mo_coeff = C.conj().T @ ovlp @ mf.mo_coeff
    mo_occ = mf.mo_occ

    mylf = lf.GLangFirsov(mol=mol, h0=H0, h1=hcore, h2=eri, ovlp=ovlp_lo,
                          h_ep=h_ep, w_p=w_p,
                          nelec=nelec, uniform=False)

    params = (np.random.random(mylf.nparam_full) - 0.5) #* 0.0

    e = mylf.kernel(params=params,
                    use_num_grad=False, full_opt=True,
                    mo_coeff=mo_coeff, mo_occ=mo_occ, conv_tol=1e-8,
                    max_cycle=200, ntrial=3, mp2=True)

    print ("GLF energy")
    print (e)

    mo_occ = mylf._scf.mo_occ
    mo_energy = mylf._scf.mo_energy
    mo_coeff = mylf._scf.mo_coeff

    rdm1 = mylf._scf.make_rdm1()
    vhf = mylf._scf.get_veff(dm=rdm1)
    vhf = vhf[:nao, :nao]

    params_opt = mylf.params_opt
    lams, zs = mylf.unpack_params(params_opt)
    params_p = mylf.pack_params(lams, zs)
    print (la.norm(mylf.get_lf_ham(params=params_p)[2] - eri))

    nso, nmo = mo_coeff.shape
    nao = nso // 2
    nocc = np.sum(mo_occ > 0.5)

    from polar.lang_firsov import mp_glf as lfmp

    nph = 8
    e_mp2_re = lfmp.get_e_mp2(mylf, nph=nph)
    print (e_mp2_re)

if __name__ == "__main__":
    settings_H2 = [{"basis": '631g', "w": 0.466751, "lc": 0.05, "pol_axis": 2},
                   {"basis": '6311++g**', "w": 0.466751, "lc": 0.05, "pol_axis": 2}]
    settings_HF = [{"basis": 'sto6g', "w": 0.531916, "lc": 0.05, "pol_axis": 2},
                   {"basis": '6311++g**', "w": 0.531916, "lc": 0.05, "pol_axis": 2}]
    test_pf_ham_HF(**settings_HF[0])
    test_pf_ham_H2(**settings_H2[0])
