#!/usr/bin/env python

"""
Coherent state MP2.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from functools import partial
import numpy as np
from scipy import linalg as la

from pyscf import lib
from pyscf.lib import logger
from pyscf import mp
from pyscf.cc import gccsd_rdm

def kernel_mp2(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=True, verbose=None):
    """
    MP2 kernel to avoid zero denominator.
    """
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = np.empty((nocc,nocc,nvir,nvir), dtype=eris.oovv.dtype)
    else:
        t2 = None

    emp2 = 0
    for i in range(nocc):
        gi = np.asarray(eris.oovv[i]).reshape(nocc,nvir,nvir)
        D = lib.direct_sum('jb+a->jba', eia, eia[i])
        D[np.abs(D) < 1e-14] = 1e-14
        t2i = gi.conj() / D
        emp2 += np.einsum('jab,jab', t2i, gi) * .25
        if with_t2:
            t2[i] = t2i

    return emp2.real, t2


def kernel(mf, h_ep, w_p, zs=None, with_t2=True, make_rdm1=False, ao_repr=False):
    """
    CS-MP2.
    """
    mf = mf.to_ghf()
    mypt = mp.MP2(mf)
    eris = mypt.ao2mo()
    e_mp2, t2 = kernel_mp2(mypt, eris=eris, with_t2=with_t2)
    logger.info(mf, "CS-MP2 on the electronic part %15.8g", e_mp2)

    if make_rdm1:
        doo = lib.einsum('imef, jmef -> ij', t2.conj(), t2) * (-0.5)
        dvv = lib.einsum('mnea, mneb -> ab', t2, t2.conj()) * (0.5)
        dov = 0.0
        dvo = 0.0

    mo_coeff = np.asarray(mf.mo_coeff)
    mo_energy = np.asarray(eris.mo_energy)
    nocc = mypt.nocc

    e_coh = 0.0
    e_ia = lib.direct_sum('a - i -> ia', mo_energy[nocc:], mo_energy[:nocc])
    for x, hx in enumerate(h_ep):
        hx = la.block_diag(hx, hx)
        h_ov = np.dot(np.dot(mo_coeff[:, :nocc].conj().T, hx), mo_coeff[:, nocc:])

        if make_rdm1:
            t1 = -h_ov / (e_ia + w_p[x])
            l1 = t1

            doo -= lib.einsum('ie,je->ij', l1, t1)
            dvv += lib.einsum('ma,mb->ab', t1, l1)

            xt2 = lib.einsum('ma,me->ae', t1, l1)
            dvo -= lib.einsum('ie,ae->ai', t1, xt2)
            dvo += t1.T
            dov += l1

        h_ov **= 2
        e_mp2 -= (h_ov / (e_ia + w_p[x])).sum()

        if zs is not None:
            m = w_p[x] * zs[x] + np.einsum('pi, pq, qi ->',
                                           mo_coeff[:, :nocc].conj(), hx, mo_coeff[:, :nocc],
                                           optimize=True)
            e_coh -= m ** 2 / w_p[x]
            if make_rdm1:
                t0 = -m / w_p[x]
                dvo += t0.conj() * t1.T
                dov += t0 * l1

    logger.info(mf, "CS-MP2 on the coherent term %15.8g", e_coh)
    e_mp2 += e_coh

    if make_rdm1:
        d1 = doo, dov, dov.T, dvv
        rdm1 = gccsd_rdm._make_rdm1(mypt, d1, with_frozen=True, ao_repr=ao_repr)
        return e_mp2, (t2, rdm1)
    else:
        return e_mp2, t2

get_e_csmp2 = partial(kernel, with_t2=False, make_rdm1=False)


