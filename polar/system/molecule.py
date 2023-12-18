#!/usr/bin/env python

"""
Molecular system with e-ph coupling.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""


from collections.abc import Iterable
import numpy as np


def get_dip_op(mol, include_nuc=True):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)

    if include_nuc:
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nuc_dip = np.einsum('i, ix -> x', charges, coords)
        ovlp = mol.intor_symmetric('int1e_ovlp')
        dip_op = ao_dip - np.einsum('x, pq -> xpq', nuc_dip / float(mol.nelectron), ovlp)
    else:
        dip_op = ao_dip
    return dip_op


def get_quadp_op(mol):
    nao = mol.nao_nr()
    quadp = mol.intor('int1e_rr').reshape(3, 3, nao, nao)
    return quadp


def get_pf_ham(mol, w0, lc, pol_axis, nmode=1, lo_method='meta-lowdin',
               self_energy='dipole'):
    """
    Solve molecule with different methods.

    Args:
        mol: Mole object.
        w0: float, frequency of photon.
        lc: float, e-ph coupling constant, = sqrt(1/V_eff)
        pol_axis: polarization direction, can be a number or a vector.
        nmode: number of modes.
        lo_method: localized orbital method to represent hamiltonian.
        self_energy: string, can be 'dipole' (product of dipole operator)
                     or 'quadrupole' (first product then second quantize).

    """
    from pyscf import gto

    from pyscf.scf import hf
    from pyscf import ao2mo
    from pyscf import lo

    from polar.lang_firsov import lang_firsov as lf
    from polar.basis import trans_1e

    mol.build(dump_input=False)
    nao = mol.nao_nr()
    nelec = mol.nelectron

    # local orbital
    C = lo.orth_ao(mol, method=lo_method)

    # integral in the AO basis
    dip_op = get_dip_op(mol)
    dip_op_lo = np.einsum('pq, pm, qn -> mn', dip_op[pol_axis], C, C, optimize=True)

    if isinstance(pol_axis, Iterable):
        h_c = lc * np.einsum("Apq, A -> pq", dip_op, pol_axis)
    else:
        h_c = lc * dip_op[pol_axis]

    w_p = np.asarray([w0 * (im*2+1) for im in range(nmode)])
    h_ep = np.einsum("x, pq -> xpq", -np.sqrt(w_p * 0.5), h_c)

    h0 = mol.energy_nuc()
    ovlp = hf.get_ovlp(mol)
    hcore = hf.get_hcore(mol)
    eri = mol.intor('int2e', aosym='s8')
    eri = ao2mo.restore(1, eri, nao)
    dm0 = hf.get_init_guess(mol)

    # transform everything to orthogonal basis
    hcore = trans_1e.trans_h1_to_lo_mol(hcore, C)
    ovlp = trans_1e.trans_h1_to_lo_mol(ovlp, C)
    eri = ao2mo.kernel(eri, C)
    h_ep = np.einsum("xpq, pm, qn -> xmn", h_ep, C, C, optimize=True)
    dm0 = trans_1e.trans_rdm1_to_lo_mol(dm0, C, ovlp)

    # self-energy
    if self_energy is not None:
        h_c = np.einsum('pq, pm, qn -> mn', h_c, C, C, optimize=True)
        hcore += 0.5 * np.einsum('pq, qs -> ps', h_c, h_c) * len(w_p)
        eri   += np.einsum('pq, rs -> pqrs', h_c, h_c) * len(w_p)

        if self_energy == 'quadrupole':
            # compute the corection term
            dip_elec = get_dip_op(mol, include_nuc=False)
            qup_elec = get_quadp_op(mol)

            if isinstance(pol_axis, Iterable):
                dip_elec = np.einsum("Apq, A -> pq", dip_elec, pol_axis)
                qup_elec = np.einsum("ABpq, A, B -> pq", qup_elec, pol_axis, pol_axis,
                                     optmize=True)
            else:
                dip_elec = dip_elec[pol_axis]
                qup_elec = qup_elec[pol_axis, pol_axis]

            dip_lo = np.einsum('pq, pm, qn -> mn', dip_elec, C, C, optimize=True) * lc
            qup_lo = np.einsum('pq, pm, qn -> mn', qup_elec, C, C, optimize=True) * lc**2
            dh1 = (qup_lo - np.einsum('pq, qs -> ps', dip_lo, dip_lo)) * (len(w_p) * 0.5)
            hcore += dh1

    # mean-field in LO basis
    mol_fake = gto.M(verbose=4, dump_input=False)
    mol_fake.nelectron = int(mol.nelectron)
    mol_fake.nao_nr = lambda *args: nao
    mol_fake.incore_anyway = True

    return mol_fake, h0, hcore, ovlp, eri, w_p, h_ep, dm0


if __name__ == "__main__":
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
    mol.basis = 'sto-3g'
    mol.build()

    w0 = 0.5
    lc = 0.05
    mol_fake, h0, hcore, ovlp, eri, w_p, h_ep, dm0 = \
            get_pf_ham(mol, w0, lc, pol_axis=2, nmode=1, lo_method='meta-lowdin',
                       self_energy='quadrupole')
