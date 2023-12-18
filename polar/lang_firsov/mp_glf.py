#!/usr/bin/env python

"""
MP2 for generalized Lang-Firsov.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from mpi4pyscf.tools import mpi
comm = mpi.comm
rank = mpi.rank

from functools import partial
import numpy as np
from scipy import linalg as la

from pyscf import lib, scf, ao2mo
from pyscf.ci import gcisd
from pyscf.cc import gccsd_rdm
from polar.fci.fci import fc_factor

einsum = partial(np.einsum, optimize=True)

def _task_location(n, task=rank):
    neach, extras = divmod(n, mpi.pool.size)
    section_sizes = ([0] + extras * [neach+1] + (mpi.pool.size-extras) * [neach])
    div_points = np.cumsum(section_sizes)
    loc0 = div_points[task]
    loc1 = div_points[task + 1]
    return loc0, loc1

def gen_states(nmo, nocc, nmode, nph, ph_only=False):
    """
    Generate ground and excited states of e-ph system.
    """
    if ph_only:
        states  = gen_states_G(nmo, nocc, nmode, nph, ph_only=True)
    else:
        states  = gen_states_G(nmo, nocc, nmode, nph, ph_only=False)
        states += gen_states_S(nmo, nocc, nmode, nph)
        states += gen_states_D(nmo, nocc, nmode, nph)
    return states

def gen_states_G(nmo, nocc, nmode, nph, ph_only=False):
    """
    Generate ground state.
    """
    states = []
    for iph in range(nph):
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
                    if ph_only:
                        states.append(str1)
                    else:
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
                        for A in range(nocc, nmo):
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
                        for J in range(I):
                            for A in range(nocc, nmo):
                                for B in range(nocc, A):
                                    states.append((I, J, A, B, str1))
    states = list(dict.fromkeys(states))
    return states

def get_config_idx(psi, nocc, nvir):
    if len(psi) == 0:
        idx = 0
    elif len(psi) == 2:
        I, A = psi
        idx = I * nvir + (A - nocc) + 1
    elif len(psi) == 4:
        I, J, A, B = psi
        idx = (I * (I - 1) // 2 + J) * nvir * (nvir - 1) // 2 + \
              (A - nocc) * (A - nocc - 1) // 2 + (B - nocc) + nocc * nvir + 1
    return idx

def get_nconfig(nocc, nvir):
    nconfig = 1 + nocc * nvir + nocc * (nocc - 1) // 2 * nvir * (nvir -1) // 2
    return nconfig

def compute_de(psi, w_p, mo_energy):
    """
    Compute the energy of psi.
    """
    if len(psi) == 1:
        str1 = psi[-1]
        res = np.sum(w_p * np.asarray(str1))
    elif len(psi) == 3:
        I, A, str1 = psi
        res = np.sum(w_p * np.asarray(str1)) + mo_energy[A] - mo_energy[I]
    elif len(psi) == 5:
        I, J, A, B, str1 = psi
        res = np.sum(w_p * np.asarray(str1)) + mo_energy[A] - mo_energy[I]\
              + mo_energy[B] - mo_energy[J]
    else:
        raise ValueError("Unknown length %s"%len(psi))
    return res

def sc_rule_h1_0X(h1, mo_coeff, psi, nocc=None, ao_repr=True):
    """
    SC rule for h1 with <0| and |0, S, D>.
    """
    if ao_repr:
        nso, nmo = mo_coeff.shape
        nao = h1.shape[-1]
        if nso == nao * 2: # GHF MO with RHF integrals
            if len(psi) == 1:
                mo_o = mo_coeff[:nao, :nocc]
                res  = einsum("po, pq, qo -> ", mo_o, h1, mo_o)
                mo_o = mo_coeff[nao:, :nocc]
                res += einsum("po, pq, qo -> ", mo_o, h1, mo_o)
            elif len(psi) == 3:
                I, A, ph_str = psi
                res  = einsum("p, pq, q ->", mo_coeff[:nao, I], h1, mo_coeff[:nao, A])
                res += einsum("p, pq, q ->", mo_coeff[nao:, I], h1, mo_coeff[nao:, A])
            else:
                res = 0.0
        elif nso == nao:
            if len(psi) == 1:
                mo_o = mo_coeff[:, :nocc]
                res = einsum("po, pq, qo -> ", mo_o, h1, mo_o)
            elif len(psi) == 3:
                I, A, ph_str = psi
                res = einsum("p, pq, q ->", mo_coeff[:, I], h1, mo_coeff[:, A])
            else:
                res = 0.0
        else:
            raise ValueError("Unknown mo_coeff shape %s and h1 shape %s"
                             %(mo_coeff.shape, h1.shape))
    else:
        raise NotImplementedError
    return res

def trans_1(h2, mo_a, mo_b, nocc):
    # 1/2 (iijj - ijji)
    mo_o_a = mo_a[:, :nocc]
    mo_o_b = mo_b[:, :nocc]
    res  = einsum("pqrs, pi, qi, rj, sj ->", h2, mo_o_a, mo_o_a,
                  mo_o_b, mo_o_b)
    res -= einsum("pqrs, pi, qj, rj, si ->", h2, mo_o_a, mo_o_a,
                  mo_o_b, mo_o_b)
    res *= 0.5
    return res

def trans_2(h2, mo_a, mo_b, I, A, nocc):
    # iajj - ijja
    #res  = einsum("pqrs, p, q, rj, sj ->", h2,
    #              mo_a[:, I], mo_a[:, A], mo_b[:, :nocc], mo_b[:, :nocc])
    #res -= einsum("pqrs, p, qj, rj, s ->", h2,
    #              mo_a[:, I], mo_a[:, :nocc], mo_b[:, :nocc], mo_b[:, A])
    tmp  = einsum("pqrs, p -> qrs", h2, mo_a[:, I])
    res  = einsum("qrs, q, rj, sj ->", tmp,
                  mo_a[:, A], mo_b[:, :nocc], mo_b[:, :nocc])
    res -= einsum("qrs, qj, rj, s ->", tmp,
                  mo_a[:, :nocc], mo_b[:, :nocc], mo_b[:, A])
    return res

def trans_3(h2, mo_a, mo_b, I, J, A, B):
    # iajb - ibja
    #res  = einsum("pqrs, p, q, r, s ->", h2,
    #              mo_a[:, I], mo_a[:, A], mo_b[:, J], mo_b[:, B])
    #res -= einsum("pqrs, p, q, r, s ->", h2,
    #              mo_a[:, I], mo_a[:, B], mo_b[:, J], mo_b[:, A])
    tmp  = einsum("pqrs, p, r -> qs", h2, mo_a[:, I], mo_b[:, J])
    res  = einsum("qs, q, s ->", tmp, mo_a[:, A], mo_b[:, B])
    res -= einsum("qs, q, s ->", tmp, mo_a[:, B], mo_b[:, A])
    return res

def sc_rule_h2_0X(h2, mo_coeff, psi, nocc=None, ao_repr=True):
    """
    h2 is a bare h2.
    """
    if ao_repr:
        nso, nmo = mo_coeff.shape
        nao = h2.shape[-1]
        if nso == nao * 2: # GHF MO with RHF integrals
            mo_a = mo_coeff[:nao]
            mo_b = mo_coeff[nao:]
            if len(psi) == 1:
                res  = trans_1(h2, mo_a, mo_a, nocc)
                res += trans_1(h2, mo_b, mo_b, nocc)
                res += trans_1(h2, mo_a, mo_b, nocc)
                res += trans_1(h2, mo_b, mo_a, nocc)
            elif len(psi) == 3:
                I, A, ph_str = psi
                res  = trans_2(h2, mo_a, mo_a, I, A, nocc)
                res += trans_2(h2, mo_b, mo_b, I, A, nocc)
                res += trans_2(h2, mo_a, mo_b, I, A, nocc)
                res += trans_2(h2, mo_b, mo_a, I, A, nocc)
            else:
                I, J, A, B, ph_str = psi
                res  = trans_3(h2, mo_a, mo_a, I, J, A, B)
                res += trans_3(h2, mo_b, mo_b, I, J, A, B)
                res += trans_3(h2, mo_a, mo_b, I, J, A, B)
                res += trans_3(h2, mo_b, mo_a, I, J, A, B)
        elif nso == nao:
            mo_a = mo_coeff
            if len(psi) == 1:
                res  = trans_1(h2, mo_a, mo_a, nocc)
            elif len(psi) == 3:
                I, A, ph_str = psi
                res  = trans_2(h2, mo_a, mo_a, I, A, nocc)
            else:
                I, J, A, B, ph_str = psi
                res  = trans_3(h2, mo_a, mo_a, I, J, A, B)
        else:
            raise ValueError("Unknown mo_coeff shape %s and h1 shape %s"
                             %(mo_coeff.shape, h2.shape))
    else:
        raise NotImplementedError
        if len(psi) == 1: # <0|h2|0>
            res = 0.5 * np.einsum("ijij ->", h2[:nocc, :nocc, :nocc, :nocc])
        elif len(psi) == 3: # <0|h2|S>
            I, A, ph_str = psi
            res = np.trace(h2[I, :nocc, A, :nocc])
        else: # <0|h2|D>
            I, J, A, B, ph_str = psi
            res = h2[I, J, A, B]
    return res

def compute_ovlp(psi1, psi2, mo_coeff, mo_occ, lams,
                 w_p, h_coh, h1, h1_exp, h_ep, h_ep_exp,
                 h2, h2_exp_1, h2_exp_2):
    """
    Compute the matrix element <0|V|psi2>.

    Args:
        psi1: <0,S,D| x <ph|, where S, D are in spin orbitals
        psi2: |0,S,D> x |ph>, where S, D are in spin orbitals
        mo_coeff: (nso, nmo)
        mo_occ:   (nmo,)
        lams:     (nmode,)

        w_p:      (nmode,), w_x b+_x b_x
        h_coh:    (nmode,) h_coh_x (b_x + b+_x)
        h1:       (nao, nao), h1_pq a+_p a_q
        h1_exp:   (nao, nao), h1_exp_pq e^{lpq(b-b+)}
        h_ep:     (nmode, nao, nao) h_ep_xpq a+_p a_q (b_x + b+_x)
        h_ep_exp: (nmode, nao, nao) h_ep_exp_xpq a+_p a_q e^{lpq(b-b+)} (b_x + b+_x) [symm]
        h2:       (nao, nao, nao, nao), 1/2 h2 a+_p a+_r a_s a_q
        h2_exp_1: (nao, nao, nao), 1/2 h2_exp_1 e^{lpq(b-b+)} a+_p a+_r a_s a_q [symm]
        h2_exp_2: (nao, nao, nao, nao), 1/2 h2_exp_1 e^{lpqrs(b-b+)} a+_p a+_r a_s a_q

    Returns:
        res: the overlap.
    """
    # ZHC NOTE current we only support ground state <HF| x <0| as bra.
    assert len(psi1) == 1

    str1 = psi1[-1]
    str2 = psi2[-1]

    nmode, nao = lams.shape
    nocc = np.sum(mo_occ > 0.5)
    nso = len(mo_occ)

    diff_str = np.asarray(str2) - np.asarray(str1)
    is_same_str = not diff_str.any()

    # 1. w_p term
    #if psi1 == psi2:
    #    res1 = np.sum(w_p * np.asarray(str2))
    #else:
    #    res1 = 0.0
    res1 = 0.0

    # 3. h_coh term, 4. h_ep term
    res3 = 0.0

    for x, hx in enumerate(h_coh):
        diff_x = abs(diff_str[x])
        if diff_x == 1: # diff by 1 phonon number at mode x
            tmp = diff_str.copy()
            tmp[x] = 0
            if not tmp.any(): # other modes are the same
                val_ph = np.sqrt(max(str1[x], str2[x]))
                # e-ph coupling term, h_ep
                val_h_ep = sc_rule_h1_0X(h_ep[x], mo_coeff, psi2,
                                         nocc=nocc, ao_repr=True)
                res3 += val_h_ep * val_ph

                # coherent term
                if psi1[:-1] == psi2[:-1]:
                    res3 += hx * val_ph
            break

    # 6. h1 exp term
    lams_diff = lib.direct_sum('yp - yq -> ypq', lams, lams)

    prod1 = 1.0
    for x in range(nmode):
        prod1 *= fc_factor(str1[x], str2[x], lams_diff[x])

    h1_eff = prod1 * h1_exp

    # 2. h1 term
    if is_same_str:
        #res2 = sc_rule_h1_0X(h1, mo_coeff, psi2, nocc=nocc, ao_repr=True)
        h1_eff += h1

    res5 = sc_rule_h1_0X(h1_eff, mo_coeff, psi2, nocc=nocc, ao_repr=True)

    # 7. h2_exp_1, 8. h2_exp_2 term
    if h2 is not None:
        prod2 = 1.0
        for x in range(nmode):
            lams_diff2 = lib.direct_sum('p - q + r - s -> pqrs',
                                        lams[x], lams[x], lams[x], lams[x])
            prod2 *= fc_factor(str1[x], str2[x], lams_diff2)
            lams_diff2 = None

        h2_eff = prod2
        h2_eff *= h2_exp_2
        prod2 = None

        v_pqr = einsum('pqr, pq -> pqr', h2_exp_1, prod1)

        for r in range(nao):
            h2_eff[:, :, r, r] += v_pqr[:, :, r]
            h2_eff[r, r, :, :] += v_pqr[:, :, r]
        v_pqr = None

        # 5. h2 term
        if is_same_str:
            #res4 = sc_rule_h2_0X(h2, mo_coeff, psi2, nocc=nocc, ao_repr=True)
            h2_eff += h2

        res6 = sc_rule_h2_0X(h2_eff, mo_coeff, psi2, nocc=nocc, ao_repr=True)
        h2_eff = None
    else:
        res6 = 0.0

    # 9. h_ep_exp term
    res7 = 0.0
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
            res7 += sc_rule_h1_0X(prod, mo_coeff, psi2, nocc=nocc, ao_repr=True)

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
            res7 += sc_rule_h1_0X(prod, mo_coeff, psi2, nocc=nocc, ao_repr=True)

    res = res1 + res3 + res5 + res6 + res7
    return res

def get_e_mp2_ref(mylf, lams=None, zs=None, nph=8):
    """
    LF-MP2.
    """
    nmode = mylf.nmode
    nao = mylf.nao
    params_opt = mylf.params_opt
    if lams is None or zs is None:
        lams, zs = mylf.unpack_params(params_opt)

    hcore = mylf.get_hcore()
    eri = mylf.get_h2()
    w_p = mylf.get_w_p()
    h_ep = mylf.get_h_ep()

    mf = mylf._scf
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff

    nso, nmo = mo_coeff.shape
    assert nso == nao * 2
    nocc = np.sum(mo_occ > 0.5)

    rdm1 = mf.make_rdm1()
    vhf = mf.get_veff(dm=rdm1)
    vhf = vhf[:nao, :nao]

    # prepare integrals
    h_coh = w_p * zs

    heff  = hcore + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
    heff -= np.einsum('xp, xpq -> pq', lams, h_ep)
    heff -= np.einsum('xpq, xq -> pq', h_ep, lams)

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    f00 = fc_factor(0, 0, diff)
    diff = None
    fac1 = np.prod(f00, axis=0)

    h1  = -heff * fac1
    h1 -= vhf

    h_ep_linear = np.zeros_like(h_ep)
    h_ep_linear[:, range(nao), range(nao)] = np.einsum('x, xp -> xp', -w_p, lams)

    h_ep_exp = h_ep

    h2 = np.zeros_like(eri)
    v_pq = np.einsum('x, xp, xq -> pq', w_p * 2.0, lams, lams, optimize=True)
    for p in range(nao):
        h2[p, p, range(nao), range(nao)] += v_pq[p]

    h2_exp_1 = np.einsum('xpq, xr -> pqr', h_ep, lams * (-2.0))
    h2_exp_2 = eri

    states = gen_states(nmo, nocc, nmode, nph=nph)
    e_mp2 = loop_mp2_mpi_ref(mylf.mol, states, mo_energy, mo_occ, mo_coeff, lams, w_p, h_coh,
                             h1, heff, h_ep_linear, h_ep_exp,
                             h2, h2_exp_1, h2_exp_2)
    return e_mp2

@mpi.parallel_call(skip_args=[12, 13, 14], skip_kwargs=['h2', 'h2_exp_1', 'h2_exp_2'])
def loop_mp2_mpi_ref(mol, states, mo_energy, mo_occ, mo_coeff, lams, w_p, h_coh,
                     h1, heff, h_ep_linear, h_ep_exp,
                     h2, h2_exp_1, h2_exp_2):
    h2 = mpi.bcast(h2)
    h2_exp_1 = mpi.bcast(h2_exp_1)
    h2_exp_2 = mpi.bcast(h2_exp_2)

    state0 = states[0]
    states = states[1:]

    nstates = len(states)
    ntasks = mpi.pool.size
    mlocs = [_task_location(nstates, task_id) for task_id in range(ntasks)]
    m_start, m_end = mlocs[rank]

    e_mp2 = 0.0
    for m in range(m_start, m_end):
        de = max(compute_de(states[m], w_p, mo_energy), 1e-12)
        ovlp = compute_ovlp(state0, states[m], mo_coeff, mo_occ, lams,
                            w_p, h_coh, h1, heff, h_ep_linear, h_ep_exp,
                            h2, h2_exp_1, h2_exp_2)
        e_mp2 -= ovlp**2 / de
    e_mp2 = comm.allreduce(e_mp2)
    return e_mp2

def compute_heff(str1, str2, lams,
                 w_p, h_coh, h1, h1_exp, h_ep, h_ep_exp,
                 h2, h2_exp_1, h2_exp_2, spin_orb=True):
    """
    Compute the matrix element <str1|V|str2>.

    Args:
        str1: <ph|
        str2: |ph>
        lams:     (nmode,)

        w_p:      (nmode,), w_x b+_x b_x
        h_coh:    (nmode,) h_coh_x (b_x + b+_x)
        h1:       (nso, nso), h1_pq a+_p a_q
        h1_exp:   (nao, nao), h1_exp_pq e^{lpq(b-b+)}
        h_ep:     (nmode, nao, nao) h_ep_xpq a+_p a_q (b_x + b+_x)
        h_ep_exp: (nmode, nao, nao) h_ep_exp_xpq a+_p a_q e^{lpq(b-b+)} (b_x + b+_x) [symm]
        h2:       (nao, nao, nao, nao), 1/2 h2 a+_p a+_r a_s a_q
        h2_exp_1: (nao, nao, nao), 1/2 h2_exp_1 e^{lpq(b-b+)} a+_p a+_r a_s a_q [symm]
        h2_exp_2: (nao, nao, nao, nao), 1/2 h2_exp_1 e^{lpqrs(b-b+)} a+_p a+_r a_s a_q
        spin_orb: if True, will return H1 with shape (nso, nso)

    Returns:
        H0: constant.
        H1: (nso, nso) if spin_orb == True. else (nao, nao)
        H2: (nao, nao, nao, nao).
    """
    nmode, nao = lams.shape

    diff_str = np.asarray(str2) - np.asarray(str1)
    is_same_str = not diff_str.any()

    H0 = 0.0
    if is_same_str:
        #1. w_p term
        H0 += np.sum(w_p * np.asarray(str2))

    # 3. h_coh term, 4. h_ep term
    H1 = 0.0
    for x, hx in enumerate(h_coh):
        diff_x = abs(diff_str[x])
        if diff_x == 1: # diff by 1 phonon number at mode x
            tmp = diff_str.copy()
            tmp[x] = 0
            if not tmp.any(): # other modes are the same
                val_ph = np.sqrt(max(str1[x], str2[x]))
                # e-ph coupling term, h_ep
                H1 += h_ep[x] * val_ph
                # coherent term
                H0 += hx * val_ph
            #break

    # 6. h1 exp term
    lams_diff = lib.direct_sum('yp - yq -> ypq', lams, lams)

    prod1 = 1.0
    for x in range(nmode):
        prod1 *= fc_factor(str1[x], str2[x], lams_diff[x])


    H1 += prod1 * h1_exp

    # 7. h2_exp_1, 8. h2_exp_2 term
    if h2 is not None:
        prod2 = 1.0
        for x in range(nmode):
            lams_diff2 = lib.direct_sum('p - q + r - s -> pqrs',
                                        lams[x], lams[x], lams[x], lams[x])
            prod2 *= fc_factor(str1[x], str2[x], lams_diff2)
            lams_diff2 = None

        h2_eff = prod2
        h2_eff *= h2_exp_2
        prod2 = None

        v_pqr = einsum('pqr, pq -> pqr', h2_exp_1, prod1)

        for r in range(nao):
            h2_eff[:, :, r, r] += v_pqr[:, :, r]
            h2_eff[r, r, :, :] += v_pqr[:, :, r]
        v_pqr = None

        # 5. h2 term
        if is_same_str:
            h2_eff += h2
        H2 = h2_eff
    else:
        H2 = None

    # 9. h_ep_exp term
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
            H1 += prod

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
            H1 += prod

    # 2. h1 term
    # ZHC NOTE expand the H1 dimension
    if spin_orb:
        H1 = la.block_diag(H1, H1)
    if is_same_str:
        H1 += h1

    return H0, H1, H2

def get_e_mp2(mylf, lams=None, zs=None, nph=8, make_rdm1=False, ao_repr=False):
    """
    LF-MP2.
    """
    nmode = mylf.nmode
    nao = mylf.nao
    params_opt = mylf.params_opt
    if lams is None or zs is None:
        lams, zs = mylf.unpack_params(params_opt)

    hcore = mylf.get_hcore()
    eri = mylf.get_h2()
    w_p = mylf.get_w_p()
    h_ep = mylf.get_h_ep()

    mf = mylf._scf
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff

    nso, nmo = mo_coeff.shape
    assert nso == nao * 2
    nocc = np.sum(mo_occ > 0.5)

    rdm1 = mf.make_rdm1()
    vhf = mf.get_veff(dm=rdm1)
    #vhf = vhf[:nao, :nao]

    # prepare integrals
    h_coh = w_p * zs

    heff  = hcore + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
    heff -= np.einsum('xp, xpq -> pq', lams, h_ep)
    heff -= np.einsum('xpq, xq -> pq', h_ep, lams)

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    f00 = fc_factor(0, 0, diff)
    diff = None
    fac1 = np.prod(f00, axis=0)

    h1  = -heff * fac1
    # ZHC NOTE here we make h1 to have shape (nso, nso)
    h1 = la.block_diag(h1, h1)
    h1 -= vhf

    h_ep_linear = np.zeros_like(h_ep)
    h_ep_linear[:, range(nao), range(nao)] = np.einsum('x, xp -> xp', -w_p, lams)

    h_ep_exp = h_ep

    h2 = np.zeros_like(eri)
    v_pq = np.einsum('x, xp, xq -> pq', w_p * 2.0, lams, lams, optimize=True)
    for p in range(nao):
        h2[p, p, range(nao), range(nao)] += v_pq[p]

    h2_exp_1 = np.einsum('xpq, xr -> pqr', h_ep, lams * (-2.0))
    h2_exp_2 = eri

    states = gen_states(nmo, nocc, nmode, nph=nph, ph_only=True)
    nstates = len(states)
    print ("nstates: ", nstates)

    e_mp2, rdm1 = loop_mp2_mpi(mylf.mol, states, mo_energy, mo_occ, mo_coeff, lams, w_p, h_coh,
                               h1, heff, h_ep_linear, h_ep_exp,
                               h2, h2_exp_1, h2_exp_2, make_rdm1=make_rdm1, ao_repr=ao_repr)
    if make_rdm1:
        return e_mp2, rdm1
    else:
        return e_mp2

@mpi.parallel_call(skip_args=[12, 13, 14], skip_kwargs=['h2', 'h2_exp_1', 'h2_exp_2'])
def loop_mp2_mpi(mol, states, mo_energy, mo_occ, mo_coeff, lams, w_p, h_coh,
                 h1, heff, h_ep_linear, h_ep_exp,
                 h2, h2_exp_1, h2_exp_2, make_rdm1=False, ao_repr=False):
    h2 = mpi.bcast(h2)
    h2_exp_1 = mpi.bcast(h2_exp_1)
    h2_exp_2 = mpi.bcast(h2_exp_2)

    nocc = np.sum(mo_occ > 0.5)
    nmo = len(mo_occ)
    nvir = nmo - nocc
    e_ia = lib.direct_sum('a - i -> ia', mo_energy[nocc:], mo_energy[:nocc])
    e_ijab = lib.direct_sum('a + b - i - j -> ijab',
                            mo_energy[nocc:], mo_energy[nocc:],
                            mo_energy[:nocc], mo_energy[:nocc])
    evec = gcisd.amplitudes_to_cisdvec(0.0, e_ia, e_ijab)
    e_ia = e_ijab = None

    state0 = states[0]

    nstates = len(states)
    ntasks = mpi.pool.size
    mlocs = [_task_location(nstates, task_id) for task_id in range(ntasks)]
    m_start, m_end = mlocs[rank]

    # ZHC NOTE
    mol.build()
    mf = scf.GHF(mol)
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ

    e_mp2 = 0.0

    if make_rdm1:
        doo = 0.0
        dvv = 0.0
        dvo = 0.0
        dov = 0.0

    for m in range(m_start, m_end):
        de_ph = max(np.sum(w_p * np.asarray(states[m])), 1e-12)
        H0, H1, H2 = compute_heff(state0, states[m], lams,
                                  0.0, h_coh, h1, heff, h_ep_linear, h_ep_exp,
                                  h2, h2_exp_1, h2_exp_2)

        mf.get_hcore = lambda *args: H1
        mf._eri = H2

        myci = gcisd.GCISD(mf)
        myci.verbose = 0
        nocc = myci.nocc
        eris = myci.ao2mo()

        mo_coeff_occ = mo_coeff[:, :nocc]
        e_hf  = np.einsum("pi, pq, qi ->", mo_coeff_occ.conj(), H1, mo_coeff_occ, optimize=True)
        e_hf += 0.5 * np.einsum("ijij ->", eris.oooo)
        e_hf += H0
        H0 = H1 = H2 = mf._eri = None

        nvec = myci.vector_size()
        vec = np.zeros(nvec)
        vec[0] = 1.0

        hc = myci.contract(vec, eris)
        hc[0] += e_hf
        if make_rdm1:
            civec = hc / (evec + de_ph)
            t0, t1, t2 = myci.cisdvec_to_amplitudes(civec)
            l1 = t1
            l2 = t2

            doo -= einsum('ie, je -> ij', l1, t1)
            doo -= einsum('imef, jmef -> ij', l2, t2) * 0.5

            dvv += einsum('ma,mb->ab', t1, l1)
            dvv += einsum('mnea,mneb->ab', t2, l2) * 0.5

            xt1  = einsum('mnef,inef->mi', l2, t2) * 0.5
            xt2  = einsum('mnfa,mnfe->ae', t2, l2) * 0.5
            xt2 += einsum('ma,me->ae', t1, l1)
            dvo += einsum('imae,me->ai', t2, l1)
            dvo -= einsum('mi,ma->ai', xt1, t1)
            dvo -= einsum('ie,ae->ai', t1, xt2)
            dvo += t1.T

            dov += l1

            if m != 0:
                dvo += t0.conj() * t1.T
                dov += t0 * l1

        hc **= 2
        if m == 0: # special treatment of ground state
            e_mp2 -= (hc[1:] / (evec[1:] + de_ph)).sum()
        else:
            e_mp2 -= (hc / (evec + de_ph)).sum()

    e_mp2 = comm.allreduce(e_mp2)

    if make_rdm1:
        doo = mpi.allreduce(doo)
        dvv = mpi.allreduce(dvv)
        dov = mpi.allreduce(dov)
        dvo = mpi.allreduce(dvo)
        d1 = doo, dov, dov.T, dvv
        rdm1 = gccsd_rdm._make_rdm1(myci, d1, with_frozen=True, ao_repr=ao_repr)
    else:
        rdm1 = None

    return e_mp2, rdm1

def get_e_mp4(mylf, lams=None, zs=None, nph=8):
    """
    LF-MP4.
    """
    nmode = mylf.nmode
    nao = mylf.nao
    params_opt = mylf.params_opt
    if lams is None or zs is None:
        lams, zs = mylf.unpack_params(params_opt)

    hcore = mylf.get_hcore()
    eri = mylf.get_h2()
    w_p = mylf.get_w_p()
    h_ep = mylf.get_h_ep()

    mf = mylf._scf
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff

    nso, nmo = mo_coeff.shape
    assert nso == nao * 2
    nocc = np.sum(mo_occ > 0.5)

    rdm1 = mf.make_rdm1()
    vhf = mf.get_veff(dm=rdm1)
    #vhf = vhf[:nao, :nao]

    # prepare integrals
    h_coh = w_p * zs

    heff  = hcore + np.einsum('x, xpq -> pq', zs * 2.0, h_ep)
    heff -= np.einsum('xp, xpq -> pq', lams, h_ep)
    heff -= np.einsum('xpq, xq -> pq', h_ep, lams)

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    f00 = fc_factor(0, 0, diff)
    diff = None
    fac1 = np.prod(f00, axis=0)

    h1  = -heff * fac1
    # ZHC NOTE here we make h1 to have shape (nso, nso)
    h1 = la.block_diag(h1, h1)
    h1 -= vhf

    h_ep_linear = np.zeros_like(h_ep)
    h_ep_linear[:, range(nao), range(nao)] = np.einsum('x, xp -> xp', -w_p, lams)

    h_ep_exp = h_ep

    h2 = np.zeros_like(eri)
    v_pq = np.einsum('x, xp, xq -> pq', w_p * 2.0, lams, lams, optimize=True)
    for p in range(nao):
        h2[p, p, range(nao), range(nao)] += v_pq[p]

    h2_exp_1 = np.einsum('xpq, xr -> pqr', h_ep, lams * (-2.0))
    h2_exp_2 = eri

    states = gen_states(nmo, nocc, nmode, nph=nph, ph_only=True)
    nstates = len(states)
    print ("nstates: ", nstates)
    e_mp1, e_mp2, e_mp3, e_mp4 = loop_mp4_mpi(mylf.mol, states, mo_energy, mo_occ, mo_coeff, lams, w_p, h_coh,
                                              h1, heff, h_ep_linear, h_ep_exp,
                                              h2, h2_exp_1, h2_exp_2)
    return e_mp1, e_mp2, e_mp3, e_mp4

@mpi.parallel_call(skip_args=[12, 13, 14], skip_kwargs=['h2', 'h2_exp_1', 'h2_exp_2'])
def loop_mp4_mpi(mol, states, mo_energy, mo_occ, mo_coeff, lams, w_p, h_coh,
                 h1, heff, h_ep_linear, h_ep_exp,
                 h2, h2_exp_1, h2_exp_2):
    h2 = mpi.bcast(h2)
    h2_exp_1 = mpi.bcast(h2_exp_1)
    h2_exp_2 = mpi.bcast(h2_exp_2)

    nocc = np.sum(mo_occ > 0.5)
    nmo = len(mo_occ)
    nvir = nmo - nocc
    e_ia = lib.direct_sum('a - i -> ia', mo_energy[nocc:], mo_energy[:nocc])
    e_ijab = lib.direct_sum('a + b - i - j -> ijab',
                            mo_energy[nocc:], mo_energy[nocc:],
                            mo_energy[:nocc], mo_energy[:nocc])
    evec = gcisd.amplitudes_to_cisdvec(0.0, e_ia, e_ijab)
    e_ia = e_ijab = None

    state0 = states[0]

    nstates = len(states)
    ntasks = mpi.pool.size
    Xlocs = [_task_location(nstates, task_id) for task_id in range(ntasks)]
    X_start, X_end = Xlocs[rank]

    noo = nocc * (nocc-1) // 2
    nvv = nvir * (nvir-1) // 2
    nvec =  1 + nocc*nvir + noo*nvv

    e_mp1 = 0.0
    e_mp2 = 0.0
    e_mp3 = 0.0
    e_mp4 = 0.0

    hmat = np.zeros((nstates, nvec, nstates, nvec))
    emat = np.zeros((nstates, nvec))

    if rank == 0:
        print ("prepare hmat")

    for X in range(X_start, X_end):
        de_ph = max(np.sum(w_p * np.asarray(states[X])), 1e-12)
        emat[X] = evec + de_ph
        for Y in range(nstates):
            H0, H1, H2 = compute_heff(states[X], states[Y], lams,
                                      0.0, h_coh, h1, heff, h_ep_linear, h_ep_exp,
                                      h2, h2_exp_1, h2_exp_2)
            hmat[X, :, Y] = sc_rule_full(H1, H2, mo_coeff, nocc, e_nuc=H0)
            H0 = H1 = H2 = None

    hmat = hmat.reshape(nstates*nvec, nstates*nvec)
    hmat = mpi.allreduce_inplace(hmat)
    emat = mpi.allreduce_inplace(emat).ravel()

    if rank == 0:
        print ("sum over states")

    mlocs = [_task_location(nstates*nvec, task_id) for task_id in range(ntasks)]
    m_start, m_end = mlocs[rank]

    e_mp1 = hmat[0, 0]
    assert abs(e_mp1) < 1e-10
    e_mp2 = 0.0
    e_mp3 = 0.0
    e_mp4 = 0.0
    for m in range(max(1, m_start), m_end):
        de_m = emat[m]
        ovlp_0m = hmat[0, m]
        ovlp_m0 = hmat[m, 0]
        e_mp2 -= ovlp_0m ** 2 / de_m
        for k in range(1, nstates*nvec):
            de_k = emat[k]
            ovlp_mk = hmat[m, k]
            for n in range(1, nstates*nvec):
                de_n = emat[n]
                ovlp_kn = hmat[k, n]
                ovlp_n0 = hmat[n, 0]
                e_mp4 -= (ovlp_0m * ovlp_mk * ovlp_kn * ovlp_n0) / (de_m * de_k * de_n)

            ovlp_0k = hmat[0, k]
            ovlp_k0 = hmat[k, 0]
            e_mp3 += (ovlp_0m * ovlp_mk * ovlp_k0) / (de_m * de_k)
            e_mp4 += (ovlp_0m * ovlp_m0 * ovlp_0k * ovlp_k0) / (de_m * de_k**2)

    e_mp2 = comm.allreduce(e_mp2)
    e_mp3 = comm.allreduce(e_mp3)
    e_mp4 = comm.allreduce(e_mp4)
    # ZHC TODO add the <0|V|0> term.
    return e_mp1, e_mp2, e_mp3, e_mp4

def sc_rule(h1, h2, mo_coeff, nocc, e_nuc=None):
    """
    SC rule for h1 with <0| and |0, S, D>.
    h1 shape (nso, nso)
    h2 shape (nao, nao, nao, nao)
    """
    assert h2.ndim == 4
    nso, nmo = mo_coeff.shape
    nao = h2.shape[-1]
    nvir = nmo - nocc
    assert h1.shape == (nso, nso)

    if nso == nao * 2: # GHF MO with RHF integrals
        mo_a = mo_coeff[:nao]
        mo_b = mo_coeff[nao:]
        #h1_mo  = np.dot(mo_a.conj().T, np.dot(h1, mo_a))
        #h1_mo += np.dot(mo_b.conj().T, np.dot(h1, mo_b))
        h1_mo = np.dot(mo_coeff.conj().T, np.dot(h1, mo_coeff))

        h_0    = h1_mo[:nocc, :nocc].trace()
        h_ov   = h1_mo[:nocc, nocc:]

        if h2 is not None:
            h2_mo  = ao2mo.kernel(h2, mo_a, compact=False).reshape(nmo, nmo, nmo, nmo)
            h2_mo += ao2mo.kernel(h2, mo_b, compact=False).reshape(nmo, nmo, nmo, nmo)
            tmp  = ao2mo.general(h2, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape(nmo, nmo, nmo, nmo)
            h2_mo += tmp
            h2_mo += tmp.transpose(2, 3, 0, 1)
            tmp = None
            h2_mo = h2_mo.transpose(0, 2, 1, 3) - h2_mo.transpose(0, 2, 3, 1)
            h_0   += 0.5 * np.einsum("ijij ->", h2_mo[:nocc, :nocc, :nocc, :nocc])
            h_ov  += np.einsum("ijaj -> ia", h2_mo[:nocc, :nocc, nocc:, :nocc])
            idx_o = np.tril_indices(nocc, k=-1)
            idx_v = np.tril_indices(nvir, k=-1)
            h_oovv = h2_mo[nocc:, nocc:, :nocc, :nocc][idx_v].transpose(1, 2, 0)[idx_o]
            if e_nuc is not None:
                h_0 += e_nuc
        else:
            h_oovv = None
    elif nso == nao:
        raise NotImplementedError
    else:
        raise ValueError("Unknown mo_coeff shape %s and h1 shape %s"
                         %(mo_coeff.shape, h1.shape))
    return h_0, h_ov, h_oovv

def sc_rule_full(h1, h2, mo_coeff, nocc, e_nuc=0.0):
    """
    SC rule for h1 with <0| and |0, S, D>.
    h1 shape (nso, nso)
    h2 shape (nao, nao, nao, nao)

    Returns:
        h_ci (nvec, nvec)
    """
    assert h2.ndim == 4
    nso, nmo = mo_coeff.shape
    nao = h2.shape[-1]
    nvir = nmo - nocc
    assert h1.shape == (nso, nso)
    assert nso == nao * 2

    # ZHC NOTE we currently only support nocc == 1
    if nocc > 1:
        raise NotImplementedError("nocc %d > 1 is not supported yet."%nocc)
    nvir = nmo - nocc
    noo = nocc * (nocc-1) // 2
    nvv = nvir * (nvir-1) // 2
    nvec =  1 + nocc*nvir + noo*nvv

    # transform h1 and h2 to mo and anti-symmetrize
    mo_a = mo_coeff[:nao]
    mo_b = mo_coeff[nao:]
    h1_mo = np.dot(mo_coeff.conj().T, np.dot(h1, mo_coeff))

    h2_mo  = ao2mo.kernel(h2, mo_a, compact=False).reshape(nmo, nmo, nmo, nmo)
    h2_mo += ao2mo.kernel(h2, mo_b, compact=False).reshape(nmo, nmo, nmo, nmo)
    tmp  = ao2mo.general(h2, (mo_a, mo_a, mo_b, mo_b), compact=False).reshape(nmo, nmo, nmo, nmo)
    h2_mo += tmp
    h2_mo += tmp.transpose(2, 3, 0, 1)
    tmp = None
    h2_mo = h2_mo.transpose(0, 2, 1, 3) - h2_mo.transpose(0, 2, 3, 1)

    # first allocate the hamiltonian matrix
    h_ci = np.zeros((nvec, nvec))

    occ_idx_ = np.zeros(nmo, dtype=bool)
    occ_idx_[:nocc] = True

    # 1. h_00
    h_00  = h1_mo[:nocc, :nocc].trace()
    h_00 += 0.5 * np.einsum("ijij ->", h2_mo[:nocc, :nocc, :nocc, :nocc])
    h_00 += e_nuc
    h_ci[0, 0] = h_00

    # 2. h_0S
    h_0S  = np.einsum("ijaj -> ia", h2_mo[:nocc, :nocc, nocc:, :nocc])
    h_0S += h1_mo[:nocc, nocc:]
    h_ci[0, 1:(1+nocc*nvir)] = h_0S.ravel()
    h_0S = None

    # 3. h_S0
    h_S0  = np.einsum("ajij -> ia", h2_mo[nocc:, :nocc, :nocc, :nocc])
    h_S0 += h1_mo[nocc:, :nocc].T
    h_ci[1:(1+nocc*nvir), 0] = h_S0.ravel()
    h_S0 = None

    # 4. h_SS
    for i in range(nocc):
        for a in range(nvir):
            ia = nocc + a
            il = 1 + i * nvir + a
            occ_idx = np.array(occ_idx_, copy=True)
            occ_idx[i] = False
            occ_idx[ia] = True
            for j in range(nocc):
                for b in range(nvir):
                    ib = nocc + b
                    ir = 1 + j * nvir + b
                    if i == j:
                        if a == b: # same det
                            h  = h1_mo[occ_idx, occ_idx].sum()
                            #h += 0.5 * np.einsum("ijij ->", h2_mo[np.ix_(occ_idx, occ_idx, occ_idx, occ_idx)])
                            h += 0.5 * (h2_mo[occ_idx, :, occ_idx][:, occ_idx, occ_idx].sum())
                            h += e_nuc
                            h_ci[il, ir] = h
                        else: # a !=b
                            #h  = np.einsum("j ->", h2_mo[ia, occ_idx, ib, occ_idx])
                            h  = h2_mo[ia, occ_idx, ib, occ_idx].sum()
                            h += h1_mo[ia, ib]
                            h_ci[il, ir] = h
                    else: # i != j
                        if a == b:
                            #h  = np.einsum("j ->", h2_mo[j, occ_idx, i, occ_idx])
                            h  = h2_mo[j, occ_idx, i, occ_idx].sum()
                            h += h1_mo[j, i]
                            h_ci[il, ir] = h * ((-1)**(i - j - 1))
                        else: # a != b
                            h  = h2_mo[j, ia, i, ib]
                            h_ci[il, ir] = h * ((-1)**(i - j - 1))

    # ZHC TODO implement 0D, SD, DD.
    return h_ci

if __name__ == "__main__":
    pass
