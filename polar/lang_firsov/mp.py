#!/usr/bin/env python

"""
MP2, MP3, MP4.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from mpi4pyscf.tools import mpi
comm = mpi.comm
rank = mpi.rank

from functools import partial
import numpy as np

from pyscf import lib
from polar.fci.fci import fc_factor

einsum = partial(np.einsum, optimize=True)

def _task_location(n, task=rank):
    neach, extras = divmod(n, mpi.pool.size)
    section_sizes = ([0] + extras * [neach+1] + (mpi.pool.size-extras) * [neach])
    div_points = np.cumsum(section_sizes)
    loc0 = div_points[task]
    loc1 = div_points[task + 1]
    return loc0, loc1

def unpack_params(params, uniform=None, nmode=None):
    if uniform:
        l, z = params
        lams = np.zeros(nmode)
        lams[:] = l
        zs = np.zeros(nmode)
        zs[:] = z
    else:
        nparam = len(params)
        if nmode is not None:
            zs = params[-nmode:]
            lams = params[:-nmode].reshape(nmode, -1)
            if lams.shape[-1] == 1:
                lams = lams[:, 0]
        else:
            nmode = len(params) // 2
            lams = params[:nmode]
            zs = params[nmode:]
    return lams, zs

def get_m_mat_hf(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph_i, nph_j, lf='lf'):
    if nph_i != 0 and nph_j != 0:
        return get_m_mat_hf_ij(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph_i, nph_j, lf=lf)
    else:
        nph = max(nph_i, nph_j)
        return get_m_mat_hf_i(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph, lf=lf)

def get_m_mat_hf_i(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph, lf='lf'):
    if nph == 1: # special treatment of 1ph state.
        m_mat = w_p * zs
        m_mat += np.einsum("xpq, pI, qI -> x", h_ep, mo_coeff_o.conj(), mo_coeff_o,
                           optimize=True)
    else:
        m_mat = np.zeros_like(zs)
    if lf:
        f00   = fc_factor(0, 0, lams)
        f0n   = fc_factor(0, nph,  lams)
        f0n_m = fc_factor(0, nph, -lams)
        hcore = np.array(hcore, copy=True)
        nao = hcore.shape[-1]
        hcore[range(nao), range(nao)] = 0.0
        m_mat += np.einsum("iq, i, q, iI, qI -> i", hcore, f0n, f00,
                           mo_coeff_o.conj(), mo_coeff_o, optimize=True)
        m_mat += np.einsum("pi, i, p, pI, iI -> i", hcore, f0n_m, f00,
                           mo_coeff_o.conj(), mo_coeff_o, optimize=True)
    return m_mat

def get_m_mat_hf_ij(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph_i, nph_j, lf='lf'):
    if lf:
        f0i   = fc_factor(0, nph_i, lams)
        f0i_m = fc_factor(0, nph_i, -lams)
        f0j   = fc_factor(0, nph_j, lams)
        f0j_m = fc_factor(0, nph_j, -lams)
        hcore = np.array(hcore, copy=True)
        nao = hcore.shape[-1]
        hcore[range(nao), range(nao)] = 0.0
        m_mat  = np.einsum("ij, i, j, iI, jI -> ij", hcore, f0i, f0j_m,
                           mo_coeff_o.conj(), mo_coeff_o, optimize=True)
        m_mat += np.einsum("ji, j, i, jI, iI -> ij", hcore, f0j, f0i_m,
                           mo_coeff_o.conj(), mo_coeff_o, optimize=True)
    else:
        nmode = len(h_ep)
        m_mat = np.zeros((nmode, nmode), dtype=h_ep.dtype)
    return m_mat

def get_m_mat_ia(hcore, h_ep, w_p, lams, zs, mo_coeff_o, mo_coeff_v, nph_i, nph_j):
    if nph_i != 0 and nph_j != 0:
        return get_m_mat_ia_ij(hcore, h_ep, w_p, lams, zs,
                               mo_coeff_o, mo_coeff_v, nph_i, nph_j)
    else:
        nph = max(nph_i, nph_j)
        return get_m_mat_ia_i(hcore, h_ep, w_p, lams, zs,
                              mo_coeff_o, mo_coeff_v, nph)

def get_m_mat_ia_i(hcore, h_ep, w_p, lams, zs, mo_coeff_o, mo_coeff_v, nph):
    f00   = fc_factor(0, 0, lams)
    f0n   = fc_factor(0, nph,  lams)
    f0n_m = fc_factor(0, nph, -lams)
    hcore = np.array(hcore, copy=True)
    nao = hcore.shape[-1]
    hcore[range(nao), range(nao)] = 0.0
    m_mat  = np.einsum("iq, i, q, iI, qA -> iIA", hcore, f0n, f00,
                       mo_coeff_o.conj(), mo_coeff_v, optimize=True)
    m_mat += np.einsum("pi, p, i, pI, iA -> iIA", hcore, f00, f0n_m,
                       mo_coeff_o.conj(), mo_coeff_v, optimize=True)
    if nph == 1: # special treatment of 1ph state.
        m_mat += np.einsum("xpq, pI, qA -> xIA", h_ep, mo_coeff_o.conj(), mo_coeff_v,
                           optimize=True)
    return m_mat

def get_m_mat_ia_ij(hcore, h_ep, w_p, lams, zs, mo_coeff_o, mo_coeff_v, nph_i, nph_j):
    f0i   = fc_factor(0, nph_i, lams)
    f0i_m = fc_factor(0, nph_i, -lams)
    f0j   = fc_factor(0, nph_j, lams)
    f0j_m = fc_factor(0, nph_j, -lams)
    hcore = np.array(hcore, copy=True)
    nao = hcore.shape[-1]
    hcore[range(nao), range(nao)] = 0.0
    m_mat  = np.einsum("ij, i, j, iI, jA -> ijIA", hcore, f0i, f0j_m,
                       mo_coeff_o.conj(), mo_coeff_v, optimize=True)
    m_mat += np.einsum("ji, j, i, jI, iA -> ijIA", hcore, f0j, f0i_m,
                       mo_coeff_o.conj(), mo_coeff_v, optimize=True)
    return m_mat

def get_e_mp2_hf(hcore, h_ep, w_p, lams, zs, mo_coeff, mo_occ, nph_i, nph_j):
    """
    Get the MP2 energy for state |HF> x |..., nph_i, ..., nph_j, ...>

    Args:
        hcore: hcore, (nao, nao).
        h_ep : electron-phonon coupling matrix (nmode, nao, nao).
        w_p  : frequency (nmode,).
        lams : lambda parameter for Lang-Firsov transform, (nmode,).
        zs   : coherent shift, (nmode,).
        mo_coeff_o: MO coeff, (nao, nmo)
        mo_occ: MO occupation, (nmo)
        nph_i: the number of phonon at mode i.
        nph_j: the number of phonon at mode j.

    Returns:
        e_mp2: MP2 energy.
    """
    assert (nph_i >= 0) and (nph_j >= 0) and (nph_i + nph_j > 0)
    nocc = (np.asarray(mo_occ > 0.5)).sum()
    mo_coeff_o = mo_coeff[:, :nocc]

    if nph_i != 0 and nph_j != 0:
        m_mat = get_m_mat_hf(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph_i, nph_j)
        m_mat **= 2
        w_mat = lib.direct_sum('i + j -> ij', w_p * nph_i, w_p * nph_j)
        # avoid double counting
        nmode = m_mat.shape[-1]
        m_mat[range(nmode), range(nmode)] = 0.0
        if nph_i == nph_j:
            idx = np.tril_indices(nmode)
            m_mat = m_mat[idx]
            w_mat = w_mat[idx]

        e_mp2 = -(m_mat / w_mat).sum()
    else:
        nph = max(nph_i, nph_j)
        m_mat = get_m_mat_hf(hcore, h_ep, w_p, lams, zs, mo_coeff_o, nph_i, nph_j)
        m_mat **= 2
        e_mp2 = -(m_mat / (w_p * nph)).sum()
    return e_mp2

def get_e_mp2_single(hcore, h_ep, w_p, lams, zs, mo_coeff, mo_occ, mo_energy, nph_i, nph_j):
    """
    Get the MP2 energy for state |ia> x |..., nph_i, ..., nph_j, ...>

    Args:
        hcore: hcore, (nao, nao).
        h_ep : electron-phonon coupling matrix (nmode, nao, nao).
        w_p  : frequency (nmode,).
        lams : lambda parameter for Lang-Firsov transform, (nmode,).
        zs   : coherent shift, (nmode,).
        nph_i: the number of phonon at mode i.
        nph_j: the number of phonon at mode j.
        mo_coeff_o: MO coeff, (nao, nmo)

    Returns:
        e_mp2: MP2 energy.
    """
    assert (nph_i >= 0) and (nph_j >= 0) and (nph_i + nph_j > 0)
    nocc = (np.asarray(mo_occ > 0.5)).sum()
    mo_coeff_o = mo_coeff[:, :nocc]
    mo_coeff_v = mo_coeff[:, nocc:]
    ew_o = mo_energy[:nocc]
    ew_v = mo_energy[nocc:]
    nmode = len(w_p)

    if nph_i != 0 and nph_j != 0:
        m_mat = get_m_mat_ia(hcore, h_ep, w_p, lams, zs, mo_coeff_o, mo_coeff_v, nph_i, nph_j)
        m_mat **= 2
        w_mat = lib.direct_sum('i + j -> ij', w_p * nph_i, w_p * nph_j)
        e_mp2 = 0.0
        e_ia = lib.direct_sum('A - I -> IA', ew_v, ew_o)

        for i in range(nmode):
            for j in range(nmode):
                if i != j:
                    # avoid double counting
                    if nph_i == nph_j:
                        if i > j:
                            e_mp2 -= (m_mat[i, j] / (w_mat[i, j] + e_ia)).sum()
                    else:
                        e_mp2 -= (m_mat[i, j] / (w_mat[i, j] + e_ia)).sum()
    else:
        nph = max(nph_i, nph_j)
        m_mat = get_m_mat_ia(hcore, h_ep, w_p, lams, zs, mo_coeff_o, mo_coeff_v, nph_i, nph_j)
        m_mat **= 2
        e_mp2 = 0.0
        e_ia = lib.direct_sum('A - I -> IA', ew_v, ew_o)
        for i in range(nmode):
            e_mp2 -= (m_mat[i] / (w_p[i] * nph + e_ia)).sum()
    return e_mp2

def get_e_mp2(hcore, h_ep, w_p_arr, lams, zs, mo_coeff, mo_occ, mo_energy, nph):
    """
    MP2 energy.
    """
    e_mp2 = 0.0
    for iph in range(1, nph):
        print ("nph: %d" % iph)
        for nph_i in range(nph):
            nph_j = iph - nph_i
            if nph_j < nph_i:
                continue
            e_mp2_hf = get_e_mp2_hf    (hcore, h_ep, w_p_arr, lams, zs,
                                        mo_coeff=mo_coeff, mo_occ=mo_occ, nph_i=nph_i, nph_j=nph_j)
            e_mp2_ia = get_e_mp2_single(hcore, h_ep, w_p_arr, lams, zs,
                                        mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy, nph_i=nph_i, nph_j=nph_j)
            print ("e_hf %5d%5d : %15.8f" %(nph_i, nph_j, e_mp2_hf))
            print ("e_ia %5d%5d : %15.8f" %(nph_i, nph_j, e_mp2_ia))
            e_mp2 += e_mp2_hf
            e_mp2 += e_mp2_ia
    return e_mp2

def compute_de(psi1, w_p, mo_energy):
    """
    Compute the energy of psi1.
    """
    I, A, str1 = psi1
    res = np.sum(w_p * np.asarray(str1)) + mo_energy[A] - mo_energy[I]
    return res

def compute_ovlp(psi1, psi2, hcore, h_ep_mo, w_p, lams, zs, mo_coeff, mo_occ, nocc,
                 lf='lf'):
    """
    Compute the matrix element <psi1|V|psi2>.

    V = \sum_{xpq} (h_ep)^x_pq a^+_p a_q (b_x + b^+_x) + \sum_{x} w_x z_x (b_x + b^+_x)
        + LF terms - <0|LF|0>
    psi1: I, A, str1
    psi2: J, B, str2
    lf: string, lf or glf; if None, will skip the exp terms.

    Returns:
        res: the overlap.
    """
    I, A, str1 = psi1
    J, B, str2 = psi2

    if I != J and A != B:
        return 0.0

    res = 0.0
    diff_str = np.asarray(str2) - np.asarray(str1)

    if I == J and A == B: # same det
        occ_idx = (mo_occ >= 0.5)
        occ_idx[I] = False
        occ_idx[A] = True
        val_h_ep = np.sum(h_ep_mo[:, occ_idx, occ_idx], axis=1)
        #if not diff_str.any(): # two ph strings are the same
        #    # ZHC NOTE w_x b+_x b_x term
        #    res += np.einsum("x, x ->", w_p, str1)
        #    # ZHC NOTE w_x z^2_x
        #    res += np.einsum("x, x ->", w_p, zs**2)
    elif I == J and A != B:
        val_h_ep = h_ep_mo[:, A, B]
    elif I != J and A == B:
        val_h_ep = h_ep_mo[:, I, J]


    for x, (w, z) in enumerate(zip(w_p, zs)):
        diff_x = abs(diff_str[x])
        if diff_x == 1:
            tmp = diff_str.copy()
            tmp[x] = 0
            if tmp.any() == False:
                val_ph = np.sqrt(max(str1[x], str2[x]))
                # coherent term
                if I == J and A == B:
                    res += w * z * val_ph
                # e-ph coupling term
                res += val_h_ep[x] * val_ph

    # LF terms
    if lf:
        for x1, l1 in enumerate(lams):
            ph1 = fc_factor(str1[x1], str2[x1], l1)
            for x2, l2 in enumerate(lams):
                if x1 != x2:
                    tmp_diff = diff_str.copy()
                    tmp_diff[x1] = 0
                    tmp_diff[x2] = 0
                    if tmp_diff.any() == False:
                        ph2 = fc_factor(str1[x2], str2[x2], -l2)
                        tmp = hcore[x1, x2] * ph1 * ph2
                        if str1[x1] == str2[x1] and str1[x2] == str2[x2]:
                            tmp -= hcore[x1, x2] * np.exp(-0.5 * (l1**2 + l2**2))
                        if I == J and A == B: # same det
                            res += tmp * np.einsum("I, I ->", mo_coeff[x1, occ_idx], mo_coeff[x2, occ_idx])
                        elif I == J and A != B:
                            res += tmp * mo_coeff[x1, A] * mo_coeff[x2, B]
                        elif I != J and A == B:
                            res += tmp * mo_coeff[x1, I] * mo_coeff[x2, J]
    return res

def sc_rule_h1(h1, mo_coeff, I, J, A, B, occ_idx=None, ao_repr=True):
    if ao_repr:
        if I == J and A == B:
            mo_o = mo_coeff[:, occ_idx]
            res = einsum("po, pq, qo -> ", mo_o, h1, mo_o)
        elif I == J and A != B:
            res = einsum("p, pq, q ->", mo_coeff[:, A], h1, mo_coeff[:, B])
        elif I != J and A == B:
            res = einsum("p, pq, q ->", mo_coeff[:, I], h1, mo_coeff[:, J])
    else:
        if I == J and A == B:
            res = np.sum(h1[occ_idx, occ_idx])
        elif I == J and A != B:
            res = h1[A, B]
        elif I != J and A == B:
            res = h1[I, J]
    return res

def compute_ovlp_g(psi1, psi2, h1, h_ep_exp, h_ep_linear_mo, fac1,
                   w_p, lams, zs, mo_coeff, mo_occ, lf='glf'):
    """
    Compute the matrix element <psi1|V|psi2>.

    V = \sum_{xpq} (h_ep)^x_pq a^+_p a_q (b_x + b^+_x) + \sum_{x} w_x z_x (b_x + b^+_x)
        + LF terms - <0|LF|0>
    psi1: I, A, str1
    psi2: J, B, str2
    lf: string, lf or glf; if None, will skip the exp terms.

    h1: h_pq a+_p a_q [e^{l^x_q - l^x_p}(b_x - b+_x) - fac1],
        h = hcore + 2 g_pq z_x - g^x_pq lam_q - g^x_pq lam_p  (term 1, 4, 5, 6)
    h_ep_exp: h_pq (b + b^+) exp(l_pq (b - b+))
              h = g^x_pq (term 2 and 3), 1/2 factor is not needed
              since <n|m+1> + <n+1|m> = <n|m-1> + <n-1|m>
    h_ep_linear_mo: h_pq (b + b^+)
                    h = - w_x l^x_p a+_p a_p

    Returns:
        res: the overlap.
    """
    I, A, str1 = psi1
    J, B, str2 = psi2

    nmode, nao = lams.shape

    if I != J and A != B:
        return 0.0

    if I == J and A == B: # same det
        occ_idx = (mo_occ >= 0.5)
        occ_idx[I] = False
        occ_idx[A] = True
        val_h_ep = np.sum(h_ep_linear_mo[:, occ_idx, occ_idx], axis=1)
    elif I == J and A != B:
        val_h_ep = h_ep_linear_mo[:, A, B]
        occ_idx = None
    elif I != J and A == B:
        val_h_ep = h_ep_linear_mo[:, I, J]
        occ_idx = None

    diff_str = np.asarray(str2) - np.asarray(str1)
    res = 0.0

    for x, (w, z) in enumerate(zip(w_p, zs)):
        diff_x = abs(diff_str[x])
        if diff_x == 1:
            tmp = diff_str.copy()
            tmp[x] = 0
            if tmp.any() == False:
                val_ph = np.sqrt(max(str1[x], str2[x]))
                # coherent term
                if I == J and A == B:
                    res += w * z * val_ph
                # e-ph coupling term, h_ep linear
                res += val_h_ep[x] * val_ph

    # LF terms
    if lf:
        # lams shape (nmode, nao)
        # normal exp term 1, 4, 5, 6, h1
        lams_diff = lib.direct_sum('yp - yq -> ypq', lams, lams)

        prod = 1.0
        for x in range(nmode):
            prod *= fc_factor(str1[x], str2[x], lams_diff[x])

        if str1 == str2:
            prod -= fac1
        prod *= h1
        res += sc_rule_h1(prod, mo_coeff, I, J, A, B, occ_idx=occ_idx)

        # exp and linear term
        # term 2 and 3
        for x in range(nmode):
            prod = 1.0
            for y in range(nmode):
                left = str1[y]
                if y == x:
                    right = str2[y] - 1
                    if right < 0:
                        prod = None
                        break
                    else:
                        tmp = fc_factor(left, right, lams_diff[y])
                        tmp *= np.sqrt(right + 1)
                else:
                    right = str2[y]
                    tmp = fc_factor(left, right, lams_diff[y])
                prod *= tmp

            if prod is not None:
                prod *= h_ep_exp[x]
                res += sc_rule_h1(prod, mo_coeff, I, J, A, B, occ_idx=occ_idx)

            prod = 1.0
            for y in range(nmode):
                right = str2[y]
                if y == x:
                    left = str1[y] - 1
                    if left < 0:
                        prod = None
                        break
                    else:
                        tmp = fc_factor(left, right, lams_diff[y])
                        tmp *= np.sqrt(left + 1)
                else:
                    left = str1[y]
                    tmp = fc_factor(left, right, lams_diff[y])
                prod *= tmp
            if prod is not None:
                prod *= h_ep_exp[x]
                res += sc_rule_h1(prod, mo_coeff, I, J, A, B, occ_idx=occ_idx)

    return res

def gen_states(nmo, nocc, nmode, nph):
    """
    Generate ground and excited states of e-ph system.
    """
    states = []
    for iph in range(0, nph):
        for nph_i in range(nph):
            nph_j = iph - nph_i
            if nph_j < nph_i:
                continue
            for i in range(nmode):
                for j in range(nmode):
                    str1 = [0 for x in range(nmode)]
                    str1[i] = nph_i
                    str1[j] = nph_j
                    str1 = tuple(str1)
                    for I in range(nocc):
                        for A in ([I] + list(range(nocc, nmo))):
                            states.append((I, A, str1))
    states = list(dict.fromkeys(states))
    return states

def gen_states_G(nmo, nocc, nmode, nph):
    """
    Generate ground state.
    """
    states = []
    for iph in range(0, nph):
        for nph_i in range(nph):
            nph_j = iph - nph_i
            if nph_j < nph_i:
                continue
            for i in range(nmode):
                for j in range(nmode):
                    str1 = [0 for x in range(nmode)]
                    str1[i] = nph_i
                    str1[j] = nph_j
                    str1 = tuple(str1)
                    states.append((str1,))
    states = list(dict.fromkeys(states))
    return states

def gen_states_S(nmo, nocc, nmode, nph):
    """
    Generate single-excited states.
    """
    states = []
    for iph in range(0, nph):
        for nph_i in range(nph):
            nph_j = iph - nph_i
            if nph_j < nph_i:
                continue
            for i in range(nmode):
                for j in range(nmode):
                    str1 = [0 for x in range(nmode)]
                    str1[i] = nph_i
                    str1[j] = nph_j
                    str1 = tuple(str1)
                    for I in range(nocc):
                        for A in (range(nocc, nmo)):
                            states.append((I, A, str1))
    states = list(dict.fromkeys(states))
    return states

def gen_states_D(nmo, nocc, nmode, nph):
    """
    Generate double-excited states.
    """
    states = []
    for iph in range(0, nph):
        for nph_i in range(nph):
            nph_j = iph - nph_i
            if nph_j < nph_i:
                continue
            for i in range(nmode):
                for j in range(nmode):
                    str1 = [0 for x in range(nmode)]
                    str1[i] = nph_i
                    str1[j] = nph_j
                    str1 = tuple(str1)
                    for I in range(nocc):
                        for A in (range(nocc, nmo)):
                            for J in range(I+1, nocc):
                                for B in (range(A+1, nmo)):
                                    states.append((I, A, J, B, str1))
    states = list(dict.fromkeys(states))
    return states

@mpi.parallel_call
def get_e_mp2_slow(mol, hcore, h_ep, w_p_arr, lams, zs, mo_coeff, mo_occ, mo_energy, nph,
                   lf='lf', h_ep_bare=None):
    """
    MP2 energy, slow version.
    """
    nmo = mo_energy.shape[-1]
    nocc = np.sum(mo_occ >= 0.5)
    nmode = len(h_ep)

    states = gen_states(nmo, nocc, nmode, nph)
    nstates = len(states)
    if rank == 0:
        print ("nstates: ", nstates)
    ntasks = mpi.pool.size
    mlocs = [_task_location(nstates, task_id) for task_id in range(ntasks)]
    m_start, m_end = mlocs[rank]

    ovlp = np.zeros(nstates)
    de = np.zeros(nstates)

    # prepare integrals according to LF or GLF.
    if lf == 'lf':
        h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep, mo_coeff.conj(), mo_coeff, optimize=True)
        for m in range(m_start, m_end):
            de[m] = compute_de(states[m], w_p_arr, mo_energy)
            ovlp[m] = compute_ovlp(states[0], states[m], hcore, h_ep_mo, w_p_arr,
                                   lams, zs, mo_coeff, mo_occ, nocc)
    elif lf == 'glf':
        h_ep = h_ep_bare
        diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
        f00 = fc_factor(0, 0, diff)
        diff = None
        fac1 = np.prod(f00, axis=0)

        h1 = hcore + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
        v56 = np.einsum('xpq, xq -> pq', h_ep, lams)
        h1 -= v56
        h1 -= v56.conj().T
        v56 = None
        h_ep_exp = h_ep

        h_ep_linear = np.einsum("x, xp -> xp", -w_p_arr, lams)
        h_ep_linear_mo = np.einsum("xp, pm, pn -> xmn", h_ep_linear, mo_coeff,
                                   mo_coeff, optimize=True)

        for m in range(m_start, m_end):
            de[m] = compute_de(states[m], w_p_arr, mo_energy)
            ovlp[m] = compute_ovlp_g(states[0], states[m], h1, h_ep_exp, h_ep_linear_mo,
                                     fac1=fac1,
                                     w_p=w_p_arr, lams=lams, zs=zs,
                                     mo_coeff=mo_coeff, mo_occ=mo_occ)
    else:
        raise NotImplementedError
    ovlp = mpi.allreduce_inplace(ovlp)
    de = mpi.allreduce_inplace(de)

    e_mp1 = ovlp[0]
    e_mp2 = 0.0
    for m in range(max(1, m_start), m_end):
        de_m = de[m]
        ovlp_0m = ovlp[m]
        e_mp2 -= ovlp_0m ** 2 / de_m

    e_mp2 = comm.allreduce(e_mp2)
    return e_mp1, e_mp2

@mpi.parallel_call
def get_e_mp3(mol, hcore, h_ep, w_p_arr, lams, zs, mo_coeff, mo_occ, mo_energy, nph,
              lf='lf', h_ep_bare=None):
    """
    MP3 energy.
    """
    nmo = mo_energy.shape[-1]
    nocc = np.sum(mo_occ >= 0.5)
    nmode = len(h_ep)

    states = gen_states(nmo, nocc, nmode, nph)
    nstates = len(states)
    ntasks = mpi.pool.size
    mlocs = [_task_location(nstates, task_id) for task_id in range(ntasks)]
    m_start, m_end = mlocs[rank]

    ovlp = np.zeros((nstates, nstates))
    de = np.zeros(nstates)

    # prepare integrals according to LF or GLF.
    if lf == 'lf':
        h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep, mo_coeff.conj(), mo_coeff, optimize=True)
        for m in range(m_start, m_end):
            de[m] = compute_de(states[m], w_p_arr, mo_energy)
            for k in range(nstates):
                ovlp[m, k] = compute_ovlp(states[m], states[k], hcore, h_ep_mo, w_p_arr,
                                          lams, zs, mo_coeff, mo_occ, nocc)
    elif lf == 'glf':
        h_ep = h_ep_bare
        diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
        f00 = fc_factor(0, 0, diff)
        diff = None
        fac1 = np.prod(f00, axis=0)

        h1 = hcore + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
        v56 = np.einsum('xpq, xq -> pq', h_ep, lams)
        h1 -= v56
        h1 -= v56.conj().T
        v56 = None
        h_ep_exp = h_ep

        h_ep_linear = np.einsum("x, xp -> xp", -w_p_arr, lams)
        h_ep_linear_mo = np.einsum("xp, pm, pn -> xmn", h_ep_linear, mo_coeff,
                                   mo_coeff, optimize=True)
        for m in range(m_start, m_end):
            de[m] = compute_de(states[m], w_p_arr, mo_energy)
            for k in range(nstates):
                ovlp[m, k] = compute_ovlp_g(states[m], states[k], h1, h_ep_exp, h_ep_linear_mo,
                                            fac1=fac1,
                                            w_p=w_p_arr, lams=lams, zs=zs,
                                            mo_coeff=mo_coeff, mo_occ=mo_occ)
    else:
        raise NotImplementedError
    ovlp = mpi.allreduce_inplace(ovlp)
    de = mpi.allreduce_inplace(de)

    e_mp1 = ovlp[0, 0]
    e_mp2 = 0.0
    e_mp3 = 0.0
    for m in range(max(1, m_start), m_end):
        de_m = de[m]
        ovlp_0m = ovlp[0, m]
        e_mp2 -= ovlp_0m ** 2 / de_m
        for k in range(1, nstates):
            de_k = de[k]
            ovlp_mk = ovlp[m, k]
            ovlp_k0 = ovlp[k, 0]
            e_mp3 += (ovlp_0m * ovlp_mk * ovlp_k0) / (de_m * de_k)

    e_mp2 = comm.allreduce(e_mp2)
    e_mp3 = comm.allreduce(e_mp3)
    # ZHC TODO add the <0|V|0> term.
    return e_mp1, e_mp2, e_mp3

@mpi.parallel_call
def get_e_mp4(mol, hcore, h_ep, w_p_arr, lams, zs, mo_coeff, mo_occ, mo_energy,
              nph, lf='lf', h_ep_bare=None):
    """
    MP4 energy.
    """
    nmo = mo_energy.shape[-1]
    nocc = np.sum(mo_occ >= 0.5)
    nmode = len(h_ep)

    h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep, mo_coeff.conj(), mo_coeff, optimize=True)

    states = gen_states(nmo, nocc, nmode, nph)
    nstates = len(states)

    ntasks = mpi.pool.size
    mlocs = [_task_location(len(states), task_id) for task_id in range(ntasks)]
    m_start, m_end = mlocs[rank]

    ovlp = np.zeros((nstates, nstates))
    de = np.zeros(nstates)

    # prepare integrals according to LF or GLF.
    if lf == 'lf':
        h_ep_mo = np.einsum("xpq, pm, qn -> xmn", h_ep, mo_coeff.conj(), mo_coeff, optimize=True)
        for m in range(m_start, m_end):
            de[m] = compute_de(states[m], w_p_arr, mo_energy)
            for k in range(nstates):
                ovlp[m, k] = compute_ovlp(states[m], states[k], hcore, h_ep_mo, w_p_arr,
                                          lams, zs, mo_coeff, mo_occ, nocc)
    elif lf == 'glf':
        h_ep = h_ep_bare
        diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
        f00 = fc_factor(0, 0, diff)
        diff = None
        fac1 = np.prod(f00, axis=0)

        h1 = hcore + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
        v56 = np.einsum('xpq, xq -> pq', h_ep, lams)
        h1 -= v56
        h1 -= v56.conj().T
        v56 = None
        h_ep_exp = h_ep

        h_ep_linear = np.einsum("x, xp -> xp", -w_p_arr, lams)
        h_ep_linear_mo = np.einsum("xp, pm, pn -> xmn", h_ep_linear, mo_coeff,
                                   mo_coeff, optimize=True)
        for m in range(m_start, m_end):
            de[m] = compute_de(states[m], w_p_arr, mo_energy)
            for k in range(nstates):
                ovlp[m, k] = compute_ovlp_g(states[m], states[k], h1, h_ep_exp, h_ep_linear_mo,
                                            fac1=fac1,
                                            w_p=w_p_arr, lams=lams, zs=zs,
                                            mo_coeff=mo_coeff, mo_occ=mo_occ)
    else:
        raise NotImplementedError
    ovlp = mpi.allreduce_inplace(ovlp)
    de = mpi.allreduce_inplace(de)

    e_mp1 = ovlp[0, 0]
    e_mp2 = 0.0
    e_mp3 = 0.0
    e_mp4 = 0.0
    for m in range(max(1, m_start), m_end):
        de_m = de[m]
        ovlp_0m = ovlp[0, m]
        ovlp_m0 = ovlp[m, 0]
        e_mp2 -= ovlp_0m ** 2 / de_m
        for k in range(1, nstates):
            de_k = de[k]
            ovlp_mk = ovlp[m, k]
            for n in range(1, nstates):
                de_n = de[n]
                ovlp_kn = ovlp[k, n]
                ovlp_n0 = ovlp[n, 0]
                e_mp4 -= (ovlp_0m * ovlp_mk * ovlp_kn * ovlp_n0) / (de_m * de_k * de_n)

            ovlp_0k = ovlp[0, k]
            ovlp_k0 = ovlp[k, 0]
            e_mp3 += (ovlp_0m * ovlp_mk * ovlp_k0) / (de_m * de_k)
            e_mp4 += (ovlp_0m * ovlp_m0 * ovlp_0k * ovlp_k0) / (de_m * de_k**2)

    e_mp2 = comm.allreduce(e_mp2)
    e_mp3 = comm.allreduce(e_mp3)
    e_mp4 = comm.allreduce(e_mp4)
    # ZHC TODO add the <0|V|0> term.
    return e_mp1, e_mp2, e_mp3, e_mp4

def lfpt2(t, U, g, w_p, nph=20):
    from scipy.special import factorial as fac
    prefac = -2.0 * t**2 * np.exp(-2.0 * g**2 / w_p**2)
    e_pt2 = 0.0
    for m in range(nph):
        for n in range(nph):
            e_pt2 += (g / w_p)**(2 * (m+n)) / fac(m) / fac(n) * \
                    (1 + (-1)**(m+n)) / ((m+n) * w_p - U - 2 * g**2/w_p)
    e_pt2 *= prefac
    return e_pt2

if __name__ == "__main__":
    pass
