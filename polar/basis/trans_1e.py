#!/usr/bin/env python

"""
Transform 1e quantities.

Authors:
    Zhi-Hao Cui
    Tianyu Zhu
    Shunyue Yuan
"""


import numpy as np
import scipy.linalg as la

from polar.utils.misc import (mdot, kdot, get_spin_dim, add_spin_dim)


# *****************************************************************************
# Transform functions AO -> LO and LO -> AO
# for h1 and rdm1
# *****************************************************************************

def trans_h1_to_lo(h_ao_ao, C_ao_lo):
    r"""
    Transform h1 to lo basis, with kpts.
    h^{LO} = C^{\dagger} h^{AO} C
    """
    h_ao_ao = np.asarray(h_ao_ao)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nlo = C_ao_lo.shape[-1]
    res_type = np.result_type(h_ao_ao.dtype, C_ao_lo.dtype)

    # treat the special case where h is 0 or [0, 0]
    if h_ao_ao.ndim == 0: # scalar
        return np.ones((nkpts, nlo, nlo), dtype=res_type) * h_ao_ao
    elif h_ao_ao.ndim == 1: # [0, 0]
        spin = len(h_ao_ao)
        h_lo_lo = np.ones((spin, nkpts, nlo, nlo), dtype=res_type)
        for s in range(spin):
            h_lo_lo[s] *= h_ao_ao[s]
        return h_lo_lo

    if C_ao_lo.ndim == 3 and h_ao_ao.ndim == 3:
        h_lo_lo  = np.zeros((nkpts, nlo, nlo), dtype=res_type)
        for k in range(nkpts):
            h_lo_lo[k] = mdot(C_ao_lo[k].conj().T, h_ao_ao[k], C_ao_lo[k])
    else:
        spin = get_spin_dim((h_ao_ao, C_ao_lo))
        h_ao_ao = add_spin_dim(h_ao_ao, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert h_ao_ao.ndim == C_ao_lo.ndim
        h_lo_lo  = np.zeros((spin, nkpts, nlo, nlo), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                h_lo_lo[s, k] = mdot(C_ao_lo[s, k].conj().T, h_ao_ao[s, k], C_ao_lo[s, k])
    return h_lo_lo

trans_h1_to_mo = trans_h1_to_lo


def trans_h1_to_ao(h_lo_lo, C_ao_lo, S_ao_ao):
    r"""
    Transform h1 to ao basis, with kpts.
    h^{LO} = C^{-1 \dagger} h^{AO} C^{-1}
    C^{-1} = C^{\dagger} S
    """
    h_lo_lo = np.asarray(h_lo_lo)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nao = C_ao_lo.shape[-2]
    res_type = np.result_type(h_lo_lo.dtype, C_ao_lo.dtype, S_ao_ao.dtype)

    # S_ao_ao is assumed to be spin unrelated
    if C_ao_lo.ndim == 3 and h_lo_lo.ndim == 3:
        h_ao_ao  = np.zeros((nkpts, nao, nao), dtype=res_type)
        for k in range(nkpts):
            C_inv = C_ao_lo[k].conj().T.dot(S_ao_ao[k])
            h_ao_ao[k] = mdot(C_inv.conj().T, h_lo_lo[k], C_inv)
    else:
        spin = get_spin_dim((h_lo_lo, C_ao_lo))
        h_lo_lo = add_spin_dim(h_lo_lo, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert h_lo_lo.ndim == C_ao_lo.ndim
        h_ao_ao  = np.zeros((spin, nkpts, nao, nao), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                C_inv = C_ao_lo[s, k].conj().T.dot(S_ao_ao[k])
                h_ao_ao[s, k] = mdot(C_inv.conj().T, h_lo_lo[s, k], C_inv)
    return h_ao_ao


def trans_rdm1_to_lo(dm_ao_ao, C_ao_lo, S_ao_ao):
    r"""
    Transform rdm1 to lo basis, with kpts.
    \gamma^{LO} = C^{-1} \gamma^{AO} (C^{-1})^{\dagger}
    C^{-1} = C^{\dagger} S
    """
    dm_ao_ao = np.asarray(dm_ao_ao)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nlo = C_ao_lo.shape[-1]
    res_type = np.result_type(dm_ao_ao.dtype, C_ao_lo.dtype, S_ao_ao.dtype)

    # S_ao_ao is assumed to be spin unrelated
    if C_ao_lo.ndim == 3 and dm_ao_ao.ndim == 3:
        dm_lo_lo  = np.zeros((nkpts, nlo, nlo), dtype=res_type)
        for k in range(nkpts):
            C_inv = C_ao_lo[k].conj().T.dot(S_ao_ao[k])
            dm_lo_lo[k] = mdot(C_inv, dm_ao_ao[k], C_inv.conj().T)
    else:
        spin = get_spin_dim((dm_ao_ao, C_ao_lo))
        dm_ao_ao = add_spin_dim(dm_ao_ao, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert dm_ao_ao.ndim == C_ao_lo.ndim
        dm_lo_lo  = np.zeros((spin, nkpts, nlo, nlo), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                C_inv = C_ao_lo[s, k].conj().T.dot(S_ao_ao[k])
                dm_lo_lo[s, k] = mdot(C_inv, dm_ao_ao[s, k], C_inv.conj().T)
    return dm_lo_lo

trans_rdm1_to_mo = trans_rdm1_to_lo


def trans_rdm1_to_ao(dm_lo_lo, C_ao_lo):
    r"""
    Transform rdm1 to ao basis, with kpts.
    \gamma^{AO} = C \gamma^{LO} C^{\dagger}
    """
    dm_lo_lo = np.asarray(dm_lo_lo)
    C_ao_lo = np.asarray(C_ao_lo)
    nkpts  = C_ao_lo.shape[-3]
    nao = C_ao_lo.shape[-2]
    res_type = np.result_type(dm_lo_lo, C_ao_lo)

    if C_ao_lo.ndim == 3 and dm_lo_lo.ndim == 3:
        dm_ao_ao  = np.zeros((nkpts, nao, nao), dtype=res_type)
        for k in range(nkpts):
            dm_ao_ao[k] = mdot(C_ao_lo[k], dm_lo_lo[k], C_ao_lo[k].conj().T)
    else:
        spin = get_spin_dim((dm_lo_lo, C_ao_lo))
        dm_lo_lo = add_spin_dim(dm_lo_lo, spin)
        C_ao_lo = add_spin_dim(C_ao_lo, spin)
        assert dm_lo_lo.ndim == C_ao_lo.ndim
        dm_ao_ao  = np.zeros((spin, nkpts, nao, nao), dtype=res_type)
        for s in range(spin):
            for k in range(nkpts):
                dm_ao_ao[s, k] = mdot(C_ao_lo[s, k], dm_lo_lo[s, k], C_ao_lo[s, k].conj().T)
    return dm_ao_ao


# *****************************************************************************
# Transform functions for molecular calculations
# *****************************************************************************

def trans_h1_to_ao_mol(h_mo_mo, C_ao_mo, S_ao_ao):
    r"""
    Transform h1 to ao basis.
    """
    h_mo_mo = np.asarray(h_mo_mo)
    C_ao_mo = np.asarray(C_ao_mo)

    nao = C_ao_mo.shape[-2]
    # spin should be encoded in C_ao_mo,
    # h_mo_mo may be spin unrelated
    if C_ao_mo.ndim < h_mo_mo.ndim:
        C_ao_mo = add_spin_dim(C_ao_mo, h_mo_mo.shape[0], non_spin_dim=2)
    if C_ao_mo.ndim == 2:
        C_inv = C_ao_mo.conj().T.dot(S_ao_ao)
        h_ao_ao = mdot(C_inv.conj().T, h_mo_mo, C_inv)
    else:
        spin = C_ao_mo.shape[0]
        h_mo_mo = add_spin_dim(h_mo_mo, spin, non_spin_dim=2)
        assert h_mo_mo.ndim == C_ao_mo.ndim
        h_ao_ao  = np.zeros((spin, nao, nao), dtype=C_ao_mo.dtype)
        for s in range(spin):
            C_inv = C_ao_mo[s].conj().T.dot(S_ao_ao)
            h_ao_ao[s] = mdot(C_inv.conj().T, h_mo_mo[s], C_inv)
    return h_ao_ao


def trans_rdm1_to_ao_mol(dm_mo_mo, C_ao_mo):
    r"""
    Transform rdm1 to ao basis. [For molecular calculations, no kpts]
    \gamma^{AO} = C \gamma^{MO} C^{\dagger}
    """
    dm_mo_mo = np.asarray(dm_mo_mo)
    C_ao_mo = np.asarray(C_ao_mo)

    nao = C_ao_mo.shape[-2]
    # spin should be encoded in C_ao_mo,
    # dm_mo_mo may be spin unrelated
    if C_ao_mo.ndim < dm_mo_mo.ndim:
        C_ao_mo = add_spin_dim(C_ao_mo, dm_mo_mo.shape[0], non_spin_dim=2)
    if C_ao_mo.ndim == 2:
        dm_ao_ao = mdot(C_ao_mo, dm_mo_mo, C_ao_mo.conj().T)
    else:
        spin = C_ao_mo.shape[0]
        dm_mo_mo = add_spin_dim(dm_mo_mo, spin, non_spin_dim=2)
        assert dm_mo_mo.ndim == C_ao_mo.ndim
        dm_ao_ao  = np.zeros((spin, nao, nao), dtype=C_ao_mo.dtype)
        for s in range(spin):
            dm_ao_ao[s] = mdot(C_ao_mo[s], dm_mo_mo[s], C_ao_mo[s].conj().T)
    return dm_ao_ao


def trans_h1_to_mo_mol(h_ao_ao, C_ao_mo):
    h_ao_ao = np.asarray(h_ao_ao)
    C_ao_mo = np.asarray(C_ao_mo)
    return trans_rdm1_to_ao_mol(h_ao_ao, np.swapaxes(C_ao_mo.conj(), -1, -2))

trans_h1_to_lo_mol = trans_h1_to_mo_mol


def trans_rdm1_to_mo_mol(rdm1_ao_ao, C_ao_mo, ovlp):
    rdm1_ao_ao = np.asarray(rdm1_ao_ao)
    C_ao_mo = np.asarray(C_ao_mo)
    nao, nmo = C_ao_mo.shape[-2:]
    ovlp = np.asarray(ovlp)
    # spin should be encoded in C_ao_mo,
    # rdm1_ao_ao may be spin unrelated
    if C_ao_mo.ndim < rdm1_ao_ao.ndim:
        C_ao_mo = add_spin_dim(C_ao_mo, rdm1_ao_ao.shape[0], non_spin_dim=2)
    if C_ao_mo.ndim == 2:
        C_inv = C_ao_mo.conj().T.dot(ovlp)
        rdm1_mo_mo = mdot(C_inv, rdm1_ao_ao, C_inv.conj().T)
    else:
        spin = C_ao_mo.shape[0]
        rdm1_ao_ao = add_spin_dim(rdm1_ao_ao, spin, non_spin_dim=2)
        assert(rdm1_ao_ao.ndim == C_ao_mo.ndim)
        rdm1_mo_mo  = np.zeros((spin, nmo, nmo), dtype=C_ao_mo.dtype)
        for s in range(spin):
            if ovlp.ndim == 3:
                C_inv = C_ao_mo[s].conj().T.dot(ovlp[s])
            else:
                C_inv = C_ao_mo[s].conj().T.dot(ovlp)

            rdm1_mo_mo[s] = mdot(C_inv, rdm1_ao_ao[s], C_inv.conj().T)
    return rdm1_mo_mo

trans_rdm1_to_lo_mol = trans_rdm1_to_mo_mol


def trans_rdm2_to_ao_mol(rdm2_mo, C_ao_mo, aabbab=True):
    r"""
    Transform rdm2 to ao basis. [For molecular calculations, no kpts]
    \gamma^{AO} = C C \rdm2^{MO} C^{\dagger} C^{\dagger}
    NOTE assume aaaa, bbbb, aabb order
    """
    rdm2_mo = np.asarray(rdm2_mo)
    C_ao_mo = np.asarray(C_ao_mo)

    # spin should be encoded in C_ao_mo,
    # rdm2_mo may be spin unrelated
    if C_ao_mo.ndim == 2 and rdm2_mo.ndim == 5:
        C_ao_mo = add_spin_dim(C_ao_mo, 2, non_spin_dim=2)

    if C_ao_mo.ndim == 2:
        rdm2_ao = _trans_rdm2_to_ao_mol(rdm2_mo, C_ao_mo)
    else:
        spin = rdm2_mo.shape[0]
        nao = C_ao_mo.shape[-2]
        rdm2_ao = np.zeros((spin, nao, nao, nao, nao), dtype=rdm2_mo.dtype)
        # ZHC NOTE assume aaaa, aabb, bbbb order
        if spin == 1:
            rdm2_ao[0] = _trans_rdm2_to_ao_mol(rdm2_mo[0], C_ao_mo[0])
        elif spin == 3:
            if aabbab:
                # aaaa
                rdm2_ao[0] = _trans_rdm2_to_ao_mol(rdm2_mo[0], C_ao_mo[0])
                # bbbb
                rdm2_ao[1] = _trans_rdm2_to_ao_mol(rdm2_mo[1], C_ao_mo[1])
                # aabb
                rdm2_ao[2] = _trans_rdm2_to_ao_mol(rdm2_mo[2], C_ao_mo[0], C_ao_mo[1])
            else:
                # aaaa
                rdm2_ao[0] = _trans_rdm2_to_ao_mol(rdm2_mo[0], C_ao_mo[0])
                # aabb
                rdm2_ao[1] = _trans_rdm2_to_ao_mol(rdm2_mo[1], C_ao_mo[0], C_ao_mo[1])
                # bbbb
                rdm2_ao[2] = _trans_rdm2_to_ao_mol(rdm2_mo[2], C_ao_mo[1])
        else:
            raise ValueError
    return rdm2_ao


def _trans_rdm2_to_ao_mol(rdm2_mo, C_a, C_b=None):
    if C_b is None:
        C_b = C_a
    assert C_a.shape == C_b.shape
    #nao, nmo = C_a.shape[-2:]
    ## (M1M2|M3M4) -> (A1M2|M3M4)
    #rdm2_ao = np.dot(C_a, rdm2_mo.reshape(nmo,-1))
    ## (A1M2|M3M4) -> (A1M2|M3B4)
    #rdm2_ao = np.dot(rdm2_ao.reshape(-1,nmo), C_b.conj().T)
    ## (A1M2|M3B4) -> (M3B4|A1M2)
    #rdm2_ao = rdm2_ao.reshape((nao,nmo,nmo,nao)).transpose(2,3,0,1)
    ## (M3B4|A1M2) -> (B3B4|A1M2)
    #rdm2_ao = np.dot(C_b, rdm2_ao.reshape(nmo,-1))
    ## (B3B4|A1M2) -> (B3B4|A1A2)
    #rdm2_ao = np.dot(rdm2_ao.reshape(-1,nmo), C_a.conj().T)
    ## (B3B4|A1A2) -> (A1A2|B3B4)
    #rdm2_ao = rdm2_ao.reshape([nao]*4).transpose((2,3,0,1))
    rdm2_ao = np.einsum("ijkl, pi, qj, rk, sl -> pqrs", rdm2_mo,
                        C_a.conj(), C_a, C_b.conj(), C_b, optimize=True)

    return rdm2_ao


# *****************************************************************************
# basis rotation related
# *****************************************************************************

def tile_u_matrix(u_val, u_virt=None, u_core=None):
    r"""
    Tile the u matrix from different subspaces.
    u has shape (nkpts, nmo, nlo)
    return C_mo_lo.

    Args:
        u_val: valence
        u_virt: virtual
        u_core: core

    Returns:
        u_tiled: C_mo_lo.
    """
    nkpts = u_val.shape[-3]
    if u_virt is None:
        u_virt = np.zeros((nkpts, 0, 0), dtype=u_val.dtype)
    if u_core is None:
        u_core = np.zeros((nkpts, 0, 0), dtype=u_val.dtype)
    nval  = u_val.shape[-1] # num of LO
    nvirt = u_virt.shape[-1]
    ncore = u_core.shape[-1]
    nlo = nmo = nval + nvirt + ncore
    if u_val.ndim == 3:
        u_tiled  = np.zeros((nkpts, nmo, nlo), dtype=u_val.dtype)
        for k in range(nkpts):
            u_tiled[k] = la.block_diag(u_core[k], u_val[k], u_virt[k])
    else:
        spin = u_val.shape[0]
        u_core = add_spin_dim(u_core, spin)
        u_virt = add_spin_dim(u_virt, spin)
        u_tiled  = np.zeros((spin, nkpts, nmo, nlo), dtype=u_val.dtype)
        for s in range(spin):
            for k in range(nkpts):
                u_tiled[s, k] = la.block_diag(u_core[s, k], u_val[s, k],
                                              u_virt[s, k])
    return u_tiled


def tile_C_ao_iao(C_val, C_virt=None, C_core=None):
    r"""
    Tile the C matrix (IAO) from different subspaces.
    C_{(s), (k), AO, LO}

    Args:
        C_val: coefficent of valence orb
        C_virt: coefficent of virtual orb
        C_core: coefficent of core orb

    Returns:
        C_tiled: tiled coeffcient.
    """
    C_val = np.asarray(C_val)
    nao = C_val.shape[-2]
    if C_val.ndim == 2:
        spin = 0
        nkpts = 0
        if C_core is None:
            C_core = np.zeros((nao, 0), dtype=C_val.dtype)
        if C_virt is None:
            C_virt = np.zeros((nao, 0), dtype=C_val.dtype)
    elif C_val.ndim == 3:
        spin = 0
        nkpts  = C_val.shape[-3]
        if C_core is None:
            C_core = np.zeros((nkpts, nao, 0), dtype=C_val.dtype)
        if C_virt is None:
            C_virt = np.zeros((nkpts, nao, 0), dtype=C_val.dtype)
    else:
        spin = C_val.shape[-4]
        nkpts  = C_val.shape[-3]
        if C_core is None:
            C_core = np.zeros((spin, nkpts, nao, 0), dtype=C_val.dtype)
        if C_virt is None:
            C_virt = np.zeros((spin, nkpts, nao, 0), dtype=C_val.dtype)

    nval  = C_val.shape[-1]
    nvirt = C_virt.shape[-1]
    ncore = C_core.shape[-1]
    nlo = nval + nvirt + ncore
    if C_val.ndim == 2:
        C_tiled = np.hstack((C_core, C_val, C_virt))
    elif C_val.ndim == 3:
        C_tiled  = np.zeros((nkpts, nao, nlo), dtype=C_val.dtype)
        for k in range(nkpts):
            C_tiled[k] = np.hstack((C_core[k], C_val[k], C_virt[k]))
    else:
        spin = C_val.shape[0]
        C_tiled  = np.zeros((spin, nkpts, nao, nlo), dtype=C_val.dtype)
        for s in range(spin):
            for k in range(nkpts):
                C_tiled[s, k] = np.hstack((C_core[s, k], C_val[s, k], C_virt[s, k]))
    return C_tiled


def multiply_basis(C_ao_lo, C_lo_eo):
    """
    Get a new basis for C_ao_eo = C_ao_lo * C_lo_eo.
    Final shape would be (spin, nkpts, nao, neo) if either has spin
    (nkpts, nao, neo) otherwise.

    Args:
        C_ao_lo: ((spin,), nkpts, nao, nlo)
        C_lo_eo: ((spin,), nkpts, nlo, neo)

    Returns:
        C_ao_eo: ((spin,), nkpts, nao, neo)
    """
    C_ao_lo = np.asarray(C_ao_lo)
    C_lo_eo = np.asarray(C_lo_eo)
    nkpts, nlo, neo = C_lo_eo.shape[-3:]
    nao = C_ao_lo.shape[-2]

    if C_ao_lo.ndim == 3 and C_lo_eo.ndim == 3:
        C_ao_eo = kdot(C_ao_lo, C_lo_eo)
    else:
        if C_ao_lo.ndim == 3 and C_lo_eo.ndim == 4:
            spin = C_lo_eo.shape[0]
            C_ao_lo = add_spin_dim(C_ao_lo, spin)
        elif C_ao_lo.ndim == 4 and C_lo_eo.ndim == 3:
            spin = C_ao_lo.shape[0]
            C_lo_eo = add_spin_dim(C_lo_eo, spin)
        elif C_ao_lo.ndim == 4 and C_lo_eo.ndim == 4:
            spin = max(C_ao_lo.shape[0], C_lo_eo.shape[0])
            C_ao_lo = add_spin_dim(C_ao_lo, spin)
            C_lo_eo = add_spin_dim(C_lo_eo, spin)
        else:
            raise ValueError("invalid shape for multiply_basis: "
                             "C_ao_lo shape %s, C_lo_eo shape: %s"
                             %(C_ao_lo.shape, C_lo_eo.shape))
        C_ao_eo = np.zeros((spin, nkpts, nao, neo),
                           dtype=np.result_type(C_ao_lo.dtype, C_lo_eo.dtype))
        for s in range(spin):
            C_ao_eo[s] = kdot(C_ao_lo[s], C_lo_eo[s])
    return C_ao_eo


def trans_mo(mo_coeff, u):
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim == 2:
        res = np.dot(mo_coeff, u)
    else:
        spin, nao, nmo = mo_coeff.shape
        res = np.zeros((spin, nao, nmo), dtype=mo_coeff.dtype)
        for s in range(spin):
            res[s] = np.dot(mo_coeff[s], u[s])
    return res


def get_mo_ovlp(mo1, mo2, ovlp):
    """
    Get MO overlap, C_1.conj().T ovlp C_2.

    Args:
        mo1: (nao, nmo1), can with spin and kpts dimension.
        mo2: (nao, nmo2), can with spin and kpts dimension.
        ovlp: can be (nao, nao) or (nkpts, nao, nao).

    Returns:
        res: (nmo1, nmo2), can with spin and kpts dimension.
    """
    ovlp = np.asarray(ovlp)
    mo1 = np.asarray(mo1)
    mo2 = np.asarray(mo2)
    if ovlp.ndim == 3: # with kpts
        nkpts = ovlp.shape[-3]
        nmo1, nmo2 = mo1.shape[-1], mo2.shape[-1]
        if mo1.ndim == 3:
            res = np.zeros((nkpts, nmo1, nmo2), dtype=np.result_type(mo1, mo2))
            for k in range(nkpts):
                res[k] = mdot(mo1[k].conj().T, ovlp[k], mo2[k])
        else:
            assert mo1.shape[0] == mo2.shape[0]
            spin = mo1.shape[0]
            res = np.zeros((spin, nkpts, nmo1, nmo2),
                           dtype=np.result_type(mo1, mo2))
            for s in range(spin):
                for k in range(nkpts):
                    res[s, k] = mdot(mo1[s, k].conj().T, ovlp[k], mo2[s, k])
    else: # without kpts
        if mo1.ndim == 2:
            res = mdot(mo1.conj().T, ovlp, mo2)
        else:
            assert mo1.shape[0] == mo2.shape[0]
            spin, nao, nmo1 = mo1.shape
            nmo2 = mo2.shape[-1]
            res = np.zeros((spin, nmo1, nmo2), dtype=np.result_type(mo1, mo2))
            for s in range(spin):
                res[s] = mdot(mo1[s].conj().T, ovlp, mo2[s])
    return res

get_mo_ovlp_k = get_mo_ovlp


def find_closest_mo(mo_coeff, mo_coeff_ref, ovlp=None, return_rotmat=False):
    """
    Given mo_coeff and a reference mo_coeff_ref,
    find the U matrix so that |mo_coeff.dot(U) - mo_coeff_ref|_F is minimal.
    i.e. so-called orthogonal Procrustes problem

    Args:
        mo_coeff: MOs need to be rotated
        mo_coeff_ref: target reference MOs
        ovlp: overlap matrix for AOs
        return_rotmat: return rotation matrix

    Returns:
        closest MO (and rotation matrix if return_rotmat == True).
    """
    mo_coeff = np.asarray(mo_coeff)
    mo_coeff_ref = np.asarray(mo_coeff_ref)
    mo_shape = mo_coeff.shape
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None]
    if mo_coeff_ref.ndim == 2:
        mo_coeff_ref = mo_coeff_ref[None]
    spin, nao, nmo = mo_coeff.shape
    if ovlp is None:
        ovlp = np.eye(nao)

    rotmat = np.zeros((spin, nmo, nmo), dtype=np.result_type(mo_coeff, ovlp))
    mo_coeff_closest = np.zeros_like(mo_coeff)
    for s in range(spin):
        ovlp_mo = mdot(mo_coeff[s].conj().T, ovlp, mo_coeff_ref[s])
        u, sigma, vt = la.svd(ovlp_mo)
        rotmat[s] = np.dot(u, vt)
        mo_coeff_closest[s] = np.dot(mo_coeff[s], rotmat[s])

    mo_coeff_closest = mo_coeff_closest.reshape(mo_shape)
    if len(mo_shape) == 2:
        rotmat = rotmat[0]
    if return_rotmat:
        return mo_coeff_closest, rotmat
    else:
        return mo_coeff_closest


if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000, suppress=False)
