#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Zhi-Hao Cui <zhcui0408@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

"""
FCI for electron-boson coupled system.
"""

import numpy as np
from scipy import linalg as la
from scipy import special
from scipy.special import factorial as fac
from scipy.special import assoc_laguerre as lg

from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci.direct_spin1 import _unpack_nelec

def get_ci_shape(norb, nelec, nmode=None, nph=None):
    """
    Get the shape of CI vector.

    Args:
        norb: number of orbitals.
        nelec: number of electrons.
        nmode: number of ph mode.
        nph: number of ph.

    Returns:
        ci_shape: (na, nb, nph+1, ... nph+1).
    """
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    if nmode is None:
        ci_shape = (na, nb)
    else:
        ci_shape = (na, nb) + (nph+1,) * nmode
    return ci_shape

def contract_1e(f1e, fcivec, norb, nelec, nmode, nph, out=None):
    """
    Contract the 1e integral.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)
    if out is None:
        fcinew = np.zeros(ci_shape)
    else:
        fcinew = out.reshape(ci_shape)

    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += ci0[str0] * (sign * f1e[a, i])
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += ci0[:, str0] * (sign * f1e[a, i])
    return fcinew.reshape(fcivec.shape)

def contract_2e(eri, fcivec, norb, nelec, nmode, nph, opt=None, out=None):
    """
    Compute E_{pq}E_{rs}|CI>
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)
    eri = eri.reshape(norb, norb, norb, norb)

    # ZHC TODO to avoid this large intermidiate
    t1 = np.zeros((norb, norb,) + ci_shape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a, i, :, str1] += sign * ci0[:, str0]

    t1 = np.tensordot(eri, t1, axes=((2, 3), (0, 1)))

    if out is None:
        fcinew = np.zeros(ci_shape)
    else:
        fcinew = out.reshape(ci_shape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a, i, str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * t1[a, i, :, str0]
    return fcinew.reshape(fcivec.shape)

def _unpack_u(u):
    if np.ndim(u) == 0:
        u_aa = u_ab = u_bb = u
    else:
        u_aa, u_ab, u_bb = u
    return u_aa, u_ab, u_bb

def contract_2e_hubbard(u, fcivec, norb, nelec, nmode, nph, opt=None, out=None):
    """
    Special case where ERI is Hubbard-like.
    u can be [U_aa, U_ab, U_bb] or a single value.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    u_aa, u_ab, u_bb = _unpack_u(u)

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)

    fcivec = fcivec.reshape(ci_shape)
    t1a = np.zeros((norb,) + ci_shape)
    t1b = np.zeros((norb,) + ci_shape)

    if out is None:
        fcinew = np.zeros(ci_shape)
    else:
        fcinew = out.reshape(ci_shape)
    for i in range(norb):
        maska = (strsa & (1 << i)) > 0
        maskb = (strsb & (1 << i)) > 0
        t1a[i, maska] += fcivec[maska]
        t1b[i][:, maskb] += fcivec[:, maskb]

    # ZHC TODO there should a way to simplify this.
    for addr, s in enumerate(strsa):
        for i in range(norb):
            if s & (1 << i):
                # u * n_alpha^+ n_alpha
                fcinew[addr] += t1a[i, addr] * u_aa
                # u * n_alpha^+ n_beta
                fcinew[addr] += t1b[i, addr] * u_ab
    for addr, s in enumerate(strsb):
        for i in range(norb):
            if s & (1 << i):
                # u * n_beta^+ n_beta
                fcinew[:, addr] += t1b[i, :, addr] * u_bb
                # u * n_beta^+ n_alpha
                fcinew[:, addr] += t1a[i, :, addr] * u_ab
    return fcinew

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    """
    Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    """
    if not isinstance(nelec, (int, np.integer)):
        nelec = sum(nelec)
    if eri.size == norb ** 4:
        h2e = np.array(eri, copy=True).reshape(norb, norb, norb, norb)
    else:
        h2e = ao2mo.restore(1, eri, norb)
    f1e = h1e - np.einsum('jiik->jk', h2e) * 0.5
    f1e *= (1.0 / (nelec+1e-100))
    for k in range(norb):
        h2e[k, k, :, :] += f1e
        h2e[:, :, k, k] += f1e
    if fac != 1.0:
        h2e *= fac
    return h2e

def get_slice(nmode, mode_id, ph_id, idxab=None):
    """
    Get the slice for CI vector.

    Returns:
        (idx_a, idx_b, ..., ph_id     , ...).
                            mode_id
        if idxab is not given, idx_a, idx_b will take all indices.
    """
    slices = [slice(None, None, None)] * (2+nmode)  # +2 for electron indiceshpp * t1
    slices[2 + mode_id] = ph_id
    if idxab is not None:
        ia, ib = idxab
        if ia is not None:
            slices[0] = ia
        if ib is not None:
            slices[1] = ib
    return tuple(slices)

def get_slice_cre(nmode, mode_id, ph_id, idxab=None):
    return get_slice(nmode, mode_id, ph_id+1, idxab)

def get_slice_des(nmode, mode_id, ph_id, idxab=None):
    return get_slice(nmode, mode_id, ph_id-1, idxab)

# Contract to one phonon creation operator
def cre_phonon(fcivec, nsite, nelec, nmode, nph, mode_id, out=None):
    ci_shape = get_ci_shape(nsite, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)
    if out is None:
        fcinew = np.zeros(ci_shape, dtype=ci0.dtype)
    else:
        fcinew = out.reshape(ci_shape)

    phonon_cre = np.sqrt(np.arange(1,nph+1))
    for ip in range(nph):
        slices1 = get_slice_cre(nmode, mode_id, ip)
        slices0 = get_slice    (nmode, mode_id, ip)
        fcinew[slices1] += phonon_cre[ip] * ci0[slices0]
    return fcinew.reshape(fcivec.shape)

# Contract to one phonon annihilation operator
def des_phonon(fcivec, nsite, nelec, nmode, nph, mode_id, out=None):
    ci_shape = get_ci_shape(nsite, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)
    if out is None:
        fcinew = np.zeros(ci_shape, dtype=ci0.dtype)
    else:
        fcinew = out.reshape(ci_shape)

    phonon_cre = np.sqrt(np.arange(1,nph+1))
    for ip in range(nph):
        slices1 = get_slice_cre(nmode, mode_id, ip)
        slices0 = get_slice    (nmode, mode_id, ip)
        fcinew[slices0] += phonon_cre[ip] * ci0[slices1]
    return fcinew.reshape(fcivec.shape)

def contract_pp(hpp, fcivec, norb, nelec, nmode, nph, zs=None, out=None):
    """
    Contract the pp part.
    """
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)
    if out is None:
        fcinew = np.zeros(ci_shape, dtype=ci0.dtype)
    else:
        fcinew = out.reshape(ci_shape)
    ph_cre = np.sqrt(np.arange(1, nph+1))

    # ZHC NOTE diagonal term:
    # w_x b^+_x b_x
    for mode_id in range(nmode):
        for i in range(1, nph+1):
            slices0 = get_slice(nmode, mode_id, i)
            fcinew[slices0] += ci0[slices0] * (hpp[mode_id] * i)

    # boson vacuum shift term, zs
    if zs is not None:
        for mode_id in range(nmode):
            for i in range(nph):
                slices1 = get_slice_cre(nmode, mode_id, i)
                slices0 = get_slice(nmode, mode_id, i)
                tmp = zs[mode_id] * (ph_cre[i] * hpp[mode_id])
                fcinew[slices1] += tmp * ci0[slices0]
                fcinew[slices0] += tmp * ci0[slices1]
    return fcinew.reshape(fcivec.shape)

def contract_pp_full(hpp, fcivec, norb, nelec, nmode, nph, zs=None, out=None):
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)
    if out is None:
        fcinew = np.zeros(ci_shape, dtype=ci0.dtype)
    else:
        fcinew = out.reshape(ci_shape)

    ph_cre = np.sqrt(np.arange(1,nph+1))
    t1 = np.zeros((nmode,) + ci_shape)
    for mode_id in range(nmode):
        for i in range(nph):
            slices1 = get_slice_cre(nmode, mode_id, i)
            slices0 = get_slice(nmode, mode_id, i)
            t1[(mode_id,) + slices0] += ci0[slices1] * ph_cre[i] # annihilation

    t1 = lib.dot(hpp, t1.reshape(nmode, -1)).reshape(t1.shape)

    if zs is not None:
        wz = np.dot(hpp, zs)

    for mode_id in range(nmode):
        for i in range(nph):
            slices1 = get_slice_cre(nmode, mode_id, i)
            slices0 = get_slice(nmode, mode_id, i)
            fcinew[slices1] += t1[(mode_id,) + slices0] * ph_cre[i] # creation
            if zs is not None:
                tmp = wz[mode_id] * ph_cre[i]
                fcinew[slices1] += tmp * ci0[slices0]
                fcinew[slices0] += tmp * ci0[slices1]

    return fcinew.reshape(fcivec.shape)

def contract_ep(g, fcivec, norb, nelec, nmode, nph, out=None):
    """
    Contract the electron-ph part.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)

    ci0 = fcivec.reshape(ci_shape)
    if out is None:
        fcinew = np.zeros(ci_shape, dtype=ci0.dtype)
    else:
        fcinew = out.reshape(ci_shape)
    ph_cre = np.sqrt(np.arange(1, nph+1))

    for mode_id in range(nmode):
        for ip in range(nph):
            for str0, tab in enumerate(link_indexa):
                for a, i, str1, sign in tab:
                    # b^+
                    slices1 = get_slice_cre(nmode, mode_id, ip, idxab=(str1, None))
                    slices0 = get_slice(nmode, mode_id, ip, idxab=(str0, None))
                    fcinew[slices1] += (g[mode_id, a, i] * ph_cre[ip] * sign) * ci0[slices0]
                    # b
                    slices0 = get_slice(nmode, mode_id, ip, idxab=(str1, None))
                    slices1 = get_slice_cre(nmode, mode_id, ip, idxab=(str0, None))
                    fcinew[slices0] += (g[mode_id, a, i] * ph_cre[ip] * sign) * ci0[slices1]
            for str0, tab in enumerate(link_indexb):
                for a, i, str1, sign in tab:
                    # b^+
                    slices1 = get_slice_cre(nmode, mode_id, ip, idxab=(None, str1))
                    slices0 = get_slice(nmode, mode_id, ip, idxab=(None, str0))
                    fcinew[slices1] += (g[mode_id, a, i] * ph_cre[ip] * sign) * ci0[slices0]
                    # b
                    slices0 = get_slice(nmode, mode_id, ip, idxab=(None, str1))
                    slices1 = get_slice_cre(nmode, mode_id, ip, idxab=(None, str0))
                    fcinew[slices0] += (g[mode_id, a, i] * ph_cre[ip] * sign) * ci0[slices1]

    return fcinew.reshape(fcivec.shape)

def make_hdiag(h1e, eri, hpp, norb, nelec, nmode, nph, opt=None):
    neleca, nelecb = _unpack_nelec(nelec)

    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    hdiag = np.zeros(ci_shape)

    try:
        occslista = cistring.gen_occslst(range(norb), neleca)
        occslistb = cistring.gen_occslst(range(norb), nelecb)
    except AttributeError:
        occslista = cistring._gen_occslst(range(norb), neleca)
        occslistb = cistring._gen_occslst(range(norb), nelecb)
    eri = ao2mo.restore(1, eri, norb)
    diagj = np.einsum('iijj->ij', eri)
    diagk = np.einsum('ijji->ij', eri)
    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag[ia, ib] = e1 + e2*.5

    for mode_id in range(nmode):
        for i in range(0, nph+1):
            slices0 = get_slice(nmode, mode_id, i)
            # ZHC NOTE to avoid local minima
            if hpp.ndim == 1:
                hdiag[slices0] += max(i * hpp[mode_id], 0.5)
            else:
                hdiag[slices0] += max(i * hpp[mode_id, mode_id], 0.5)

    return hdiag.ravel()

def make_hdiag_hubbard(h1e, eri, hpp, norb, nelec, nmode, nph, opt=None):
    neleca, nelecb = _unpack_nelec(nelec)
    nelec_tot = neleca + nelecb
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    occslista = [tab[:neleca,0] for tab in link_indexa]
    occslistb = [tab[:nelecb,0] for tab in link_indexb]

    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    hdiag = np.zeros(ci_shape)

    u_aa, u_ab, u_bb = _unpack_u(eri)

    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = u_ab * nelec_tot
            hdiag[ia,ib] = e1 + e2

    for mode_id in range(nmode):
        for i in range(0, nph+1):
            slices0 = get_slice(nmode, mode_id, i)
            # ZHC NOTE to avoid local minima
            if hpp.ndim == 1:
                hdiag[slices0] += max(i * hpp[mode_id], 0.5)
            else:
                hdiag[slices0] += max(i * hpp[mode_id, mode_id], 0.5)

    return hdiag.ravel()

def kernel(h1e, eri, norb, nelec, nmode, nph, g, hpp, shift_vac=True,
           hubbard=False, ecore=0, tol=1e-10, lindep=1e-14, max_cycle=200,
           max_space=16, nroots=1, verbose=5, ci0=None):
    """
    Main FCI kernel.
    """
    # ZHC NOTE first solve the electronic part
    if not hubbard:
        from pyscf.fci import direct_spin1
        e, c = direct_spin1.kernel(h1e, eri, norb, nelec, ecore=ecore,
                                   verbose=verbose,
                                   tol=tol, lindep=lindep, max_cycle=max_cycle,
                                   max_space=max_space, nroots=nroots)
        print ("FCI electronic", e)

    assert h1e.shape[-1] == norb
    if isinstance(g, np.ndarray):
        assert g.shape == (nmode, norb, norb)

    if shift_vac:
        if nroots > 1:
            dm0 = direct_spin1.make_rdm1(c[0], norb, nelec)
        else:
            dm0 = direct_spin1.make_rdm1(c, norb, nelec)
        if hpp.ndim == 1:
            zs = -np.einsum('xpq, qp -> x', g, dm0, optimize=True) / hpp
            ecore = ecore + np.einsum("x, x ->", hpp, zs**2)
        else:
            tmp = -np.einsum('xpq, qp -> x', g, dm0)
            zs = la.solve(hpp, tmp)
            ecore = ecore + np.einsum("xy, x, y ->", hpp, zs, zs, optimize=True)

        h1e = h1e + np.einsum('xpq, x -> pq', g, zs * 2, optimize=True)
    else:
        zs = None
    print ("ecore: ", ecore)

    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    if ci0 is None:
        ci0 = np.zeros(ci_shape)
        ci0.__setitem__((0, 0) + (0,)*nmode, 1)
        #ci0 = ci0.reshape(c.shape[0], c.shape[1], -1).transpose(2, 0, 1)
        #ci0[:] = c
        #ci0 = ci0.transpose(1, 2, 0).reshape(ci_shape)
        ci0 += (np.random.random(ci_shape) - 0.5) * 1e-5
        if nroots > 1:
            ci0 = [ci0]
            for r in range(nroots-1):
                ci0.append(ci0[0] + (np.random.random(ci_shape) - 0.5) * 1e-4)

    # ZHC NOTE reshape the CI vectors
    if nroots > 1:
        if isinstance(ci0, np.ndarray) and ci0.shape == ci_shape:
            ci0 = ci0.ravel()
        else:
            ci0_col = []
            for r in range(nroots):
                ci0_col.append(ci0[r].ravel())
            ci0 = ci0_col
    else:
        assert ci0.size == np.prod(ci_shape)
        ci0 = ci0.ravel()

    if hpp.ndim == 1:
        contract_hpp = contract_pp
    else:
        contract_hpp = contract_pp_full

    if hubbard:
        print ("hubbard routine")
        u_aa, u_ab, u_bb = _unpack_u(eri)
        f1e = np.array(h1e, copy=True)
        f1e[range(norb), range(norb)] -= u_ab * 0.5

        def hop(c):
            hc  = contract_2e_hubbard(np.asarray(eri) * 0.5, c, norb, nelec, nmode, nph)
            contract_1e(f1e, c, norb, nelec, nmode, nph, out=hc)
            contract_ep(g, c, norb, nelec, nmode, nph, out=hc)
            contract_hpp(hpp, c, norb, nelec, nmode, nph, zs=zs, out=hc)
            return hc.reshape(-1)

        hdiag = make_hdiag_hubbard(h1e, eri, hpp, norb, nelec, nmode, nph)
    else:
        h2e = absorb_h1e(h1e, eri, norb, nelec, 0.5)
        def hop(c):
            hc  = contract_2e(h2e, c, norb, nelec, nmode, nph)
            contract_ep(g, c, norb, nelec, nmode, nph, out=hc)
            contract_hpp(hpp, c, norb, nelec, nmode, nph, zs=zs, out=hc)
            return hc.reshape(-1)

        #f1e = h1e - np.einsum('jiik->jk', eri) * 0.5
        #def hop(c):
        #    hc  = contract_2e(eri * 0.5, c, norb, nelec, nmode, nph)
        #    contract_1e(f1e, c, norb, nelec, nmode, nph, out=hc)
        #    contract_ep(g, c, norb, nelec, nmode, nph, out=hc)
        #    contract_hpp(hpp, c, norb, nelec, nmode, nph, zs=zs, out=hc)
        #    return hc.reshape(-1)

        hdiag = make_hdiag(h1e, eri, hpp, norb, nelec, nmode, nph)
    precond = lib.make_diag_precond(hdiag, level_shift=1e-3)


    e, c = lib.davidson(hop, ci0, precond, verbose=verbose,
                        tol=tol, lindep=lindep, max_cycle=max_cycle,
                        max_space=max_space, nroots=nroots)
    if shift_vac:
        return e+ecore, (c, zs)
    else:
        return e+ecore, c

def make_rdm1e(fcivec, norb, nelec):
    '''1-electron density matrix dm_pq = <|p^+ q|>'''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    rdm1 = np.zeros((norb,norb))
    ci0 = fcivec.reshape(na,-1)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * np.dot(ci0[str1],ci0[str0])

    ci0 = fcivec.reshape(na,nb,-1)
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * np.einsum('ax,ax->', ci0[:,str1],ci0[:,str0])
    return rdm1

def make_rdm12e(fcivec, norb, nelec):
    '''1-electron and 2-electron density matrices
    dm_pq = <|p^+ q|>
    dm_{pqrs} = <|p^+ r^+ q s|>  (note 2pdm is ordered in chemist notation)
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = fcivec.reshape(na,nb,-1)
    rdm1 = np.zeros((norb,norb))
    rdm2 = np.zeros((norb,norb,norb,norb))
    for str0 in range(na):
        t1 = np.zeros((norb,norb,nb)+ci0.shape[2:])
        for a, i, str1, sign in link_indexa[str0]:
            t1[i,a,:] += sign * ci0[str1,:]

        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[i,a,k] += sign * ci0[str0,str1]

        rdm1 += np.einsum('mp,ijmp->ij', ci0[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += np.einsum('ijmp,klmp->jikl', t1, t1)
        tmp = lib.dot(t1.reshape(norb**2,-1), t1.reshape(norb**2,-1).T)
        rdm2 += tmp.reshape((norb,)*4).transpose(1,0,2,3)
    rdm1, rdm2 = rdm.reorder_rdm(rdm1, rdm2, True)
    return rdm1, rdm2

def make_rdm1p_linear(fcivec, norb, nelec, nmode, nph, zs=None):
    """
    1-phonon density matrix dm_x = <|x^+|> (linear part).

    Returns:
        rdm1: (nmode,).
    """
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)

    t1 = np.zeros((nmode,) + ci_shape)
    phonon_cre = np.sqrt(np.arange(1, nph+1))
    for mode_id in range(nmode):
        for i in range(nph):
            slices1 = get_slice_cre(nmode, mode_id, i)
            slices0 = get_slice    (nmode, mode_id, i)
            t1[(mode_id,) + slices0] += ci0[slices1] * phonon_cre[i]

    rdm1 = np.dot(t1.reshape(nmode, -1), ci0.reshape(-1))
    if zs is not None:
        rdm1 += zs
    return rdm1

def make_rdm1p(fcivec, norb, nelec, nmode, nph, zs=None):
    """
    1-phonon density matrix dm_xy = <|y^+ x|>

    Returns:
        rdm1: (nmode, nmode).
    """
    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    ci0 = fcivec.reshape(ci_shape)

    t1 = np.zeros((nmode,) + ci_shape)
    phonon_cre = np.sqrt(np.arange(1, nph+1))
    for mode_id in range(nmode):
        for i in range(nph):
            slices1 = get_slice_cre(nmode, mode_id, i)
            slices0 = get_slice    (nmode, mode_id, i)
            t1[(mode_id,) + slices0] += ci0[slices1] * phonon_cre[i]

    rdm1 = lib.dot(t1.reshape(nmode, -1), t1.reshape(nmode, -1).T)
    if zs is not None:
        rdm1 += np.einsum("y, x -> xy", zs, zs)
    return rdm1

def fc_factor(n, m, l):
    """
    Get the Franck-Condon factors, <n|exp(-l(b-b+))|m>
    https://physics.stackexchange.com/questions/553225/representation-of-the-displacement-operator-in-number-basis
    """
    lsq = l * l
    res  = np.exp(lsq * (-0.5))
    if n >= m:
        res *= l ** (n-m)
        res *= np.sqrt(fac(m) / fac(n))
        res *= lg(lsq, m, n-m)
    else:
        res *= l ** (m-n)
        res *= (np.sqrt(fac(n) / fac(m)) * ((-1)**(m-n)))
        res *= lg(lsq, n, m-n)
    return res

def get_fc_arr(nph, lam):
    fc_arr = np.empty((nph+1, nph+1))
    for n in range(nph+1):
        for m in range(nph+1):
            fc_arr[n, m] = fc_factor(n, m, lam)
    return fc_arr

if __name__ == '__main__':
    norb = 2
    nelec = 2
    nph = 10
    nmode = norb

    t = np.zeros((norb,norb))
    idx = np.arange(norb-1)
    t[idx+1,idx] = t[idx,idx+1] = -1
    u = 1.5
    eri = np.zeros((norb, norb, norb, norb))
    eri[range(norb), range(norb), range(norb), range(norb)] = u
    g = 0.5
    #g = 0.0
    gmat = np.zeros((norb, norb, norb))
    gmat[0, 0, 0] = g
    gmat[1, 1, 1] = g

    hpp = np.ones(norb) * 3
    #hpp[:] = 0.0
    print('nelec = ', nelec)
    print('nph = ', nph)
    print('t =\n', t)
    print('u =', u)
    print('g =', g)
    print('hpp =\n', hpp)

    es = []
    nelecs = [(ia,ib) for ia in range(norb+1) for ib in range(ia+1)]
    for nelec in nelecs:
        e,c = kernel(t, eri, norb, nelec, norb, nph, gmat, hpp, tol=1e-10,
                verbose=5, nroots=1, shift_vac=False)
        print('nelec =', nelec, 'E =', e)
        es.append(e)
    es = np.hstack(es)
    idx = np.argsort(es)
    print(es[idx])

    print('\nGround state is')
    nelec = nelecs[idx[0]]
    #nelec = [2, 2]
    e,c = kernel(t, eri, norb, nelec, norb, nph, gmat, hpp, tol=1e-10,
            verbose=0, nroots=1, shift_vac=False)
    print('nelec =', nelec, 'E =', e)
    dm1 = make_rdm1e(c, norb, nelec)
    print('electron DM')
    print(dm1)

    dm1a, dm2 = make_rdm12e(c, norb, nelec)
    print('check 1e DM', np.allclose(dm1, dm1a))
    print('check 2e DM', np.allclose(dm1, np.einsum('ijkk->ij', dm2)/(sum(nelec)-1.)))
    print('check 2e DM', np.allclose(dm1, np.einsum('kkij->ij', dm2)/(sum(nelec)-1.)))

    print('phonon DM')
    dm1 = make_rdm1p(c, norb, nelec, nmode, nph)
    print(dm1)

    dm1a = np.empty_like(dm1)
    for i in range(norb):
        for j in range(norb):
            c1 = des_phonon(c, norb, nelec, nmode, nph, j)
            c1 = cre_phonon(c1, norb, nelec, nmode, nph, i)
            dm1a[i,j] = np.dot(c.ravel(), c1.ravel())
    print('check phonon DM', np.allclose(dm1, dm1a))

    ci_shape = get_ci_shape(norb, nelec, nmode, nph)
    eri = np.zeros((norb,norb,norb,norb))
    for i in range(norb):
        eri[i,i,i,i] = u
    np.random.seed(3)
    ci0 = np.random.random(ci_shape)
    ci1 = contract_2e(eri, ci0, norb, nelec, nmode, nph)
    ci2 = contract_2e_hubbard(u, ci0, norb, nelec, nmode, nph)
    print('Check contract_2e', abs(ci1-ci2).sum())
