#!/usr/bin/env python

"""
Gradient of LF.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from functools import reduce, partial
import numpy as np
from scipy import linalg as la

from pyscf import gto, scf, ao2mo, lib
from pyscf.scf import hf

einsum = partial(np.einsum, optimize=True)

def get_grad_lf(mylf, params=None, rdm1=None, mo_coeff=None, mo_occ=None,
                scf_max_cycle=50, fci=False, beta=np.inf):
    """
    Analytical gradient w.r.t. params for LF.
    """
    # ZHC TODO support complex case

    if params is None:
        params = mylf.params
    lams, zs = mylf.unpack_params(params)
    nao = mylf.nao

    if rdm1 is None:
        if mo_coeff is None or mo_occ is None:
            e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=scf_max_cycle,
                                            fci=fci, beta=beta)
        else:
            rdm1 = mylf.make_rdm1(mo_coeff, mo_occ)

    g = mylf.h_ep
    w_p = mylf.w_p
    fac = np.exp(-0.5 * lams**2)

    rdm1_diag = rdm1[range(nao), range(nao)]
    # 1-body

    grad_l = (w_p * lams - g - w_p * zs) * rdm1_diag
    h1 = np.array(mylf.get_h1(), copy=True)
    h1[range(nao), range(nao)] = 0.0
    grad_l += einsum('kj, k, k, j, jk -> k', h1, fac, -lams, fac, rdm1)

    # 2-body
    h2 = mylf.get_h2()
    if h2 is not None:
        grad_l += (-g + w_p * lams) * (rdm1_diag * rdm1_diag) * 0.5
        fac2 = np.exp(-2.0 * lams**2)
        for i in range(nao):
            for j in range(nao):
                for k in range(nao):
                    for l in range(nao):
                        hr = h2[i, j, k, l] * (rdm1[j, i] * rdm1[l, k] -
                                               0.5 * rdm1[l, i] * rdm1[j, k]) * 0.25
                        if i == j and i == l and i == k:
                            # i = j = k = l
                            pass
                        elif i == j and i == k:
                            # i = j = k != l
                            tmp = hr * fac[l] * fac[k]
                            grad_l[l] -= tmp * lams[l]
                            grad_l[k] -= tmp * lams[k]
                        elif i == j and i == l:
                            # i = j = l != k
                            tmp = hr * fac[l] * fac[k]
                            grad_l[l] -= tmp * lams[l]
                            grad_l[k] -= tmp * lams[k]
                        elif i == l and i == k:
                            # i = l = k != j
                            tmp = hr * fac[i] * fac[j]
                            grad_l[i] -= tmp * lams[i]
                            grad_l[j] -= tmp * lams[j]
                        elif j == l and j == k:
                            # i != j = k = l
                            tmp = hr * fac[i] * fac[j]
                            grad_l[i] -= tmp * lams[i]
                            grad_l[j] -= tmp * lams[j]

                        elif i == j:
                            if k != l:
                                tmp = hr * fac[k] * fac[l]
                                grad_l[k] -= tmp * lams[k]
                                grad_l[l] -= tmp * lams[l]
                        elif i == k:
                            if j == l:
                                tmp = hr * fac2[i] * fac2[j]
                                grad_l[i] -= tmp * lams[i] * 4.0
                                grad_l[j] -= tmp * lams[j] * 4.0
                            else:
                                tmp = hr * fac2[i] * fac[j] * fac[l]
                                grad_l[i] -= tmp * lams[i] * 4.0
                                grad_l[j] -= tmp * lams[j]
                                grad_l[l] -= tmp * lams[l]
                        elif i == l:
                            if j != k:
                                tmp = hr * fac[j] * fac[k]
                                grad_l[j] -= tmp * lams[j]
                                grad_l[k] -= tmp * lams[k]
                        elif j == k:
                            if i != l:
                                tmp = hr * fac[i] * fac[l]
                                grad_l[i] -= tmp * lams[i]
                                grad_l[l] -= tmp * lams[l]
                        elif j == l:
                            if i == k:
                                tmp = hr * fac2[i] * fac2[j]
                                grad_l[i] -= tmp * lams[i] * 4.0
                                grad_l[j] -= tmp * lams[j] * 4.0
                            else:
                                tmp = hr * fac[i] * fac[k] * fac2[j]
                                grad_l[i] -= tmp * lams[i]
                                grad_l[k] -= tmp * lams[k]
                                grad_l[j] -= tmp * lams[j] * 4.0
                        elif k == l:
                            if i != j:
                                tmp = hr * fac[i] * fac[j]
                                grad_l[i] -= tmp * lams[i]
                                grad_l[j] -= tmp * lams[j]

                        else:
                            tmp = hr * fac[i] * fac[j] * fac[k] * fac[l]
                            grad_l[i] -= tmp * lams[i]
                            grad_l[j] -= tmp * lams[j]
                            grad_l[k] -= tmp * lams[k]
                            grad_l[l] -= tmp * lams[l]

    grad_z = ((g - w_p * lams) * rdm1_diag + (w_p * zs))
    grad = mylf.pack_params(grad_l, grad_z) * 2.0
    return grad

def get_grad_lf_full(mylf, params=None, rdm1=None, mo_coeff=None, mo_occ=None):
    """
    Analytical gradient w.r.t. params for LF.
    """
    kappa, lams, zs = mylf.unpack_params_full(params)
    params_p = mylf.pack_params(lams, zs)

    H0, H1, H2, H_ep, w_p = mylf.get_lf_ham(params=params_p)
    ovlp = mylf.get_ovlp()
    nao = mylf.nao
    h1 = mylf.get_h1()
    nelec = mylf.nelec

    if H2 is not None:
        mf = hf.RHF(mylf.mol)
        mf.energy_nuc = lambda *args: H0
        mf.get_hcore = lambda *args: H1
        mf.get_ovlp = lambda *args: ovlp
        # ZHC FIXME NOTE the transformed H2 may not have the 4-fold symmetry,
        # it is only 2-fold. pqrs = rspq
        #mf._eri = ao2mo.restore(4, H2, nao)
        mf._eri = H2
        mf.direct_scf = False

        dr = hf.unpack_uniq_var(kappa, mo_occ)
        mo_coeff = np.dot(mo_coeff, la.expm(dr))
        rdm1 = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=rdm1)

        grad_k = mf.get_grad(mo_coeff, mo_occ, fock) * 2
        grad_p = mylf.get_grad(params=params_p, mo_coeff=mo_coeff, mo_occ=mo_occ)
        grad = np.hstack((grad_k, grad_p))
    else:
        raise NotImplementedError
    return grad

def get_grad_glf(mylf, params=None, rdm1=None, mo_coeff=None, mo_occ=None,
                 scf_max_cycle=50, fci=False, beta=np.inf):
    """
    Analytical gradient w.r.t. params for GLF.
    """
    h_ep = mylf.get_h_ep()
    if h_ep.ndim == 3:
        return get_grad_glf_2(mylf, params=params, rdm1=rdm1, mo_coeff=mo_coeff, mo_occ=mo_occ,
                              scf_max_cycle=scf_max_cycle, fci=fci, beta=beta)

    if params is None:
        params = mylf.params
    lams, zs = mylf.unpack_params(params)
    nmode = mylf.nmode
    nao = mylf.nao

    if rdm1 is None:
        if mo_coeff is None or mo_occ is None:
            e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=scf_max_cycle,
                                            fci=fci, beta=beta)
        else:
            rdm1 = mylf.make_rdm1(mo_coeff, mo_occ)

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    diff **= 2
    fac = diff.sum(axis=0)
    diff = None
    fac *= (-0.5)
    fac = np.exp(fac)

    h1 = mylf.get_h1()
    h2 = mylf.get_h2()
    w_p = mylf.get_w_p()

    G = h_ep - einsum('x, xp -> xp', w_p, lams)
    rdm1_diag = rdm1[range(nao), range(nao)]
    grad_z = np.einsum('xp, p -> x', G, rdm1_diag) * 2.0
    grad_z += (w_p * zs) * 2.0

    # 1-body linear
    grad_l = np.einsum('y, ym -> ym', w_p, lams)
    grad_l -= h_ep
    grad_l = np.einsum('ym, m -> ym', grad_l, rdm1_diag)
    grad_l -= einsum('y, y, m -> ym', w_p, zs, rdm1_diag)
    grad_l *= 2.0

    # 1-body exp
    grad_l -= einsum('pm, pm, ym, mp -> ym', h1, fac, lams, rdm1) * 2.0
    grad_l += einsum('pm, pm, yp, mp -> ym', h1, fac, lams, rdm1) * 2.0
    #grad_l += einsum('mq, mq, yq, qm -> ym', h1, fac, lams, rdm1)
    #grad_l -= einsum('mq, mq, ym, qm -> ym', h1, fac, lams, rdm1)

    if h2 is not None:
        # 2-body linear
        #rdm2 pqrs = qp, sr -> pqrs - 0.5 * ps, qr
        # mmqq = mm, qq - (mq, mq)
        rdm2_diag = np.einsum('mm, qq -> mq', rdm1, rdm1) - 0.5 * np.einsum('mq, mq -> mq', rdm1, rdm1)
        grad_l += einsum('mq, y, yq -> ym', rdm2_diag, w_p, lams)
        grad_l += einsum('pm, y, yp -> ym', rdm2_diag, w_p, lams)
        grad_l -= einsum('pm, yp -> ym', rdm2_diag, h_ep) * 2.0

        # 2-body exp
        fac = 0.0
        for x in range(nmode):
            diff = lib.direct_sum('q - p + s - r -> pqrs', lams[x], lams[x], lams[x], lams[x])
            diff **= 2
            fac += diff
            diff = None
        fac *= (-0.5)
        fac = np.exp(fac)
        H2 = h2 * fac

        # ZHC TODO modify to vj and vk
        tmp = -einsum("pmrs, ym, mp, sr -> ym", H2, lams, rdm1, rdm1)
        tmp += einsum("pmrs, ym, ps, mr -> ym", H2, lams, rdm1, rdm1) * 0.5

        tmp += einsum("pmrs, yp, mp, sr -> ym", H2, lams, rdm1, rdm1)
        tmp -= einsum("pmrs, yp, ps, mr -> ym", H2, lams, rdm1, rdm1) * 0.5

        tmp -= einsum("pmrs, ys, mp, sr -> ym", H2, lams, rdm1, rdm1)
        tmp += einsum("pmrs, ys, ps, mr -> ym", H2, lams, rdm1, rdm1) * 0.5

        tmp += einsum("pmrs, yr, mp, sr -> ym", H2, lams, rdm1, rdm1)
        tmp -= einsum("pmrs, yr, ps, mr -> ym", H2, lams, rdm1, rdm1) * 0.5

        grad_l += tmp * 2

    grad = mylf.pack_params(grad_l, grad_z)
    return grad

def get_grad_glf_2(mylf, params=None, rdm1=None, mo_coeff=None, mo_occ=None,
                   scf_max_cycle=50, fci=False, beta=np.inf):
    """
    Analytical gradient w.r.t. params for GLF.
    """
    if params is None:
        params = mylf.params
    lams, zs = mylf.unpack_params(params)
    nmode = mylf.nmode
    nao = mylf.nao

    if rdm1 is None:
        if mo_coeff is None or mo_occ is None:
            e_tot, rdm1 = mylf.solve_lf_ham(params=params, scf_max_cycle=scf_max_cycle,
                                            fci=fci, beta=beta)
        else:
            rdm1 = mylf.make_rdm1(mo_coeff, mo_occ)

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    tmp = diff ** 2
    fac = tmp.sum(axis=0)
    tmp = None
    fac *= (-0.5)
    fac = np.exp(fac)
    fac1 = fac

    h1 = mylf.get_h1()
    h2 = mylf.get_h2()
    h_ep = mylf.get_h_ep()
    w_p = mylf.get_w_p()

    G = h_ep * fac1

    # grad z term 0 4 11
    grad_z  = np.einsum('xpq, qp -> x', G, rdm1)
    grad_z -= np.einsum('x, xp -> x', w_p, lams * rdm1[range(nao), range(nao)])
    grad_z += (w_p * zs)
    grad_z *= 2.0

    # grad lamba
    # term 1 and 4
    # q = m
    #grad_l  = einsum('yp, pm, mp -> ym', lams, fac1, rdm1)
    #grad_l -= einsum('ym, pm, mp -> ym', lams, fac1, rdm1)
    # p = m
    #grad_l -= einsum('ym, mq, qm -> ym', lams, h1*fac1, rdm1)
    #grad_l += einsum('yq, mq, qm -> ym', lams, h1*fac1, rdm1)
    h_tmp = einsum('x, xpq -> pq', zs * 2.0, h_ep)
    h_tmp += h1

    # term 5 and 6 exp grad
    tmp = einsum('xpq, xp -> pq', h_ep, lams)
    h_tmp -= tmp
    h_tmp -= tmp.conj().T

    tmp = np.einsum('pm, mp -> pm', h_tmp * fac1, rdm1) * 2.0
    grad_l  = np.dot(lams, tmp)
    grad_l -= np.einsum('ym, pm -> ym', lams, tmp)

    # term 5 and 6 linear grad
    grad_l -= einsum('ypm, mp -> ym', G, rdm1) * 2.0

    # term 7 and 11
    rdm1_diag = rdm1[range(nao), range(nao)]
    tmp = lib.direct_sum('ym - y -> ym', lams, zs)
    grad_l += einsum('y, ym, m -> ym', w_p*2.0, tmp, rdm1_diag)

    if h2 is not None:
        # term 10
        grad_l += einsum("y, yq, q, m -> ym", w_p * 2.0, lams, rdm1_diag, rdm1_diag)
        grad_l -= einsum("y, yq, mq, mq -> ym", w_p, lams, rdm1, rdm1)

        # term 11
        fac = 0.0
        for x in range(nmode):
            diff = lib.direct_sum('q - p + s - r -> pqrs', lams[x], lams[x], lams[x], lams[x])
            diff **= 2
            fac += diff
            diff = None
        fac *= (-0.5)
        h_tmp = np.exp(fac)
        fac = None
        h_tmp *= h2

        # q
        # the total factor is 2 (from 0.5 x 4)
        rdm1_2 = rdm1 * 2.0
        vj = lib.einsum('pqrs, sr -> pq', h_tmp, rdm1_2)
        vk = lib.einsum('pqrs, ps -> qr', h_tmp, rdm1)

        #grad_l -= einsum('pmrs, ym, mp, sr -> ym', h_tmp, lams, rdm1, rdm1_2)
        grad_l -= einsum('pm, ym, mp -> ym', vj, lams, rdm1)
        #grad_l += einsum('pmrs, ym, ps, mr -> ym', h_tmp, lams, rdm1, rdm1)
        grad_l += einsum('mr, ym, mr -> ym', vk, lams, rdm1)

        #grad_l += einsum('pmrs, yp, mp, sr -> ym', h_tmp, lams, rdm1, rdm1_2)
        grad_l += einsum('pm, yp, mp -> ym', vj, lams, rdm1)
        grad_l -= einsum('pmrs, yp, ps, mr -> ym', h_tmp, lams, rdm1, rdm1)

        grad_l -= einsum('pmrs, ys, mp, sr -> ym', h_tmp, lams, rdm1, rdm1_2)
        grad_l += einsum('pmrs, ys, ps, mr -> ym', h_tmp, lams, rdm1, rdm1)

        grad_l += einsum('pmrs, yr, mp, sr -> ym', h_tmp, lams, rdm1, rdm1_2)
        #grad_l -= einsum('pmrs, yr, ps, mr -> ym', h_tmp, lams, rdm1, rdm1)
        grad_l -= einsum('mr, yr, mr -> ym', vk, lams, rdm1)

        h_tmp = None
        # term 8, 9
        # exp grad
        v_pqr = lib.einsum('xpq, xr -> pqr', G, lams * (-2.0))

        # q
        rdm1_diag_2 = rdm1_diag * 2.0
        grad_l -= einsum('pmr, ym, mp, r -> ym', v_pqr, lams, rdm1, rdm1_diag_2)
        grad_l += einsum('pmr, ym, pr, mr -> ym', v_pqr, lams, rdm1, rdm1)

        grad_l += einsum('mqr, yq, qm, r -> ym', v_pqr, lams, rdm1, rdm1_diag_2)
        grad_l -= einsum('mqr, yq, mr, qr -> ym', v_pqr, lams, rdm1, rdm1)
        v_pqr = None

        # linear grad
        grad_l -= einsum('ypq, qp, m -> ym', G, rdm1, rdm1_diag_2)
        grad_l += einsum('ypq, pm, qm -> ym', G, rdm1, rdm1)

    grad = mylf.pack_params(grad_l, grad_z)
    return grad

def get_grad_gglf(mylf, params=None, rdm1=None, mo_coeff=None, mo_occ=None,
                  scf_max_cycle=50, fci=False, beta=np.inf):
    """
    Analytical gradient w.r.t. params for GGLF.
    """
    raise NotImplementedError
