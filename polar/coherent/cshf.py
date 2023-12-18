#!/usr/bin/env python

"""
Coherent state Hartree-Fock.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from functools import partial
import numpy as np
from scipy import linalg as la

from pyscf import lib
from pyscf.lib import logger
from pyscf import scf

def get_e_coh(mf, h_ep, w_p, rdm1=None):
    if rdm1 is None:
        rdm1 = mf.make_rdm1()
    rdm1 = np.asarray(rdm1)
    zs = get_zs(h_ep, w_p, rdm1)
    if w_p.ndim == 2:
        e_coh = -np.einsum("xy, x, y ->", w_p, zs.conj(), zs, optimize=True)
    else:
        e_coh = -np.einsum("x, x ->", w_p, np.abs(zs)**2)
    return e_coh

def get_v1_zs(zs, h_ep):
    return np.einsum("x, xpq -> pq", zs*2.0, h_ep)

def get_v0_zs(zs, w_p):
    if w_p.ndim == 2:
        v0 = np.einsum("xy, x, y ->", w_p, zs.conj(), zs, optimize=True)
    else:
        v0 = np.einsum("x, x ->", w_p, np.abs(zs)**2)
    return v0

def get_zs(h_ep, w_p, rdm1):
    if w_p.ndim == 2:
        if rdm1.ndim == 3: # UHF
            tmp = -np.einsum("xij, sji -> x", h_ep, rdm1, optimize=True)
        elif rdm1.shape[-1] == h_ep.shape[-1] * 2: # GHF
            nao = h_ep.shape[-1]
            rdm1 = [rdm1[:nao, :nao], rdm1[nao:, nao:]]
            tmp = -np.einsum("xij, sji -> x", h_ep, rdm1, optimize=True)
        else: # RHF
            tmp = -np.einsum("xij, ji -> x", h_ep, rdm1, optimize=True)
        zs = la.solve(w_p, tmp)
    else:
        w_p_inv = 1.0 / w_p
        if rdm1.ndim == 3: # UHF
            zs = np.einsum("xij, x, sji -> x", h_ep, -w_p_inv, rdm1, optimize=True)
        elif rdm1.shape[-1] == h_ep.shape[-1] * 2: # GHF
            nao = h_ep.shape[-1]
            rdm1 = [rdm1[:nao, :nao], rdm1[nao:, nao:]]
            zs = np.einsum("xij, x, sji -> x", h_ep, -w_p_inv, rdm1, optimize=True)
        else: # RHF
            zs = np.einsum("xij, x, ji -> x", h_ep, -w_p_inv, rdm1, optimize=True)
    return zs

def make_rdm1p(mf, zs=None):
    if zs is None: zs = mf.zs
    rdm1 = np.einsum("y, x -> xy", zs, zs)
    return rdm1

def make_rdm1p_linear(mf, zs=None):
    if zs is None: zs = mf.zs
    rdm1 = np.array(zs, copy=True)
    return rdm1

class RCSHF(scf.rhf.RHF):
    def __init__(self, mol, h_ep, w_p, zs=None):
        scf.rhf.RHF.__init__(self, mol)
        self._keys = self._keys.union(["nmode", "h_ep", "w_p", "zs", "v1_zs", "v0_zs"])
        self.nmode = len(w_p)
        self.w_p = w_p
        self.h_ep = h_ep
        self.nao = mol.nao_nr()
        assert h_ep.shape == (self.nmode, self.nao, self.nao)
        self.zs = zs
        self.v1_zs = None
        self.v0_zs = None

    def dump_flags(self, verbose=None):
        scf.rhf.RHF.dump_flags(self, verbose)
        logger.info(self, 'zs = %s', self.zs)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, zs=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if zs is None: zs = self.zs
        if zs is None:
            zs = self.zs = get_zs(self.h_ep, self.w_p, self.get_init_guess())

        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * .5
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * .5
            vhf += np.asarray(vhf_last)

        v1_zs = get_v1_zs(zs, self.h_ep)
        vhf += v1_zs
        return vhf

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        e1 = np.einsum('ij,ji->', h1e, dm).real
        e_coul = np.einsum('ij,ji->', vhf, dm).real * .5

        zs = self.zs
        v1_zs = get_v1_zs(zs, self.h_ep)
        e_eph = np.einsum("ij, ji -> ", v1_zs, dm).real
        e_coul -= e_eph * 0.5
        e_ph = get_v0_zs(zs, self.w_p)

        self.scf_summary['e1'] = e1
        self.scf_summary['e2'] = e_coul
        self.scf_summary['e_eph'] = e_eph
        self.scf_summary['e_ph'] = e_ph
        logger.info(self, 'E1 = %s  E_coul = %s', e1, e_coul)
        logger.info(self, 'zs = %s', zs)
        logger.info(self, 'E(e-ph) = %s  E(ph) = %s', e_eph, e_ph)
        return e1 + e_coul + e_eph + e_ph, e_coul

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        rdm1 = np.dot(mo_coeff*mo_occ, mo_coeff.conj().T)
        # ZHC NOTE update zs after dm is determined
        self.zs = get_zs(self.h_ep, self.w_p, rdm1)
        return rdm1

    def make_rdm1e(self, mo_coeff=None, mo_occ=None):
        return self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)

    make_rdm1p = make_rdm1p
    make_rdm1p_linear = make_rdm1p_linear

    def CSMP2(self, with_t2=False, make_rdm1=False, ao_repr=False):
        from polar.coherent import csmp
        rdm1 = self.make_rdm1()
        zs = self.zs
        h_ep = self.h_ep
        w_p = self.w_p
        e_mp2, t2 = csmp.kernel(self, h_ep, w_p, zs=zs,
                                with_t2=with_t2, make_rdm1=make_rdm1, ao_repr=ao_repr)
        if with_t2 or make_rdm1:
            return e_mp2, t2
        else:
            return e_mp2

class UCSHF(scf.uhf.UHF):
    def __init__(self, mol, h_ep, w_p, zs=None):
        scf.uhf.UHF.__init__(self, mol)
        self._keys = self._keys.union(["nmode", "h_ep", "w_p", "zs", "v1_zs", "v0_zs"])
        self.nmode = len(w_p)
        self.w_p = w_p
        self.h_ep = h_ep
        self.nao = mol.nao_nr()
        assert h_ep.shape == (self.nmode, self.nao, self.nao)
        self.zs = zs
        self.v1_zs = None
        self.v0_zs = None

    def dump_flags(self, verbose=None):
        scf.uhf.UHF.dump_flags(self, verbose)
        logger.info(self, 'zs = %s', self.zs)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, zs=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if zs is None: zs = self.zs
        if zs is None:
            zs = self.zs = get_zs(self.h_ep, self.w_p, self.get_init_guess())
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.asarray((dm*.5,dm*.5))
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += np.asarray(vhf_last)

        v1_zs = get_v1_zs(zs, self.h_ep)
        vhf += v1_zs
        return vhf

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if isinstance(dm, np.ndarray) and dm.ndim == 2:
            dm = np.array((dm*.5, dm*.5))
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
            h1e = (h1e, h1e)
        e1  = np.einsum('ij,ji->', h1e[0], dm[0])
        e1 += np.einsum('ij,ji->', h1e[1], dm[1])
        e_coul = (np.einsum('ij,ji->', vhf[0], dm[0]) +
                  np.einsum('ij,ji->', vhf[1], dm[1])) * .5
        e_coul = e_coul.real

        zs = self.zs
        v1_zs = get_v1_zs(zs, self.h_ep)
        e_eph = np.einsum("ij, sji -> ", v1_zs, dm).real
        e_coul -= e_eph * 0.5
        e_ph = get_v0_zs(zs, self.w_p)

        self.scf_summary['e1'] = e1
        self.scf_summary['e2'] = e_coul
        self.scf_summary['e_eph'] = e_eph
        self.scf_summary['e_ph'] = e_ph
        logger.info(self, 'E1 = %s  E_coul = %s', e1, e_coul)
        logger.info(self, 'zs = %s', zs)
        logger.info(self, 'E(e-ph) = %s  E(ph) = %s', e_eph, e_ph)
        return e1 + e_coul + e_eph + e_ph, e_coul

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo_a = mo_coeff[0]
        mo_b = mo_coeff[1]
        dm_a = np.dot(mo_a*mo_occ[0], mo_a.conj().T)
        dm_b = np.dot(mo_b*mo_occ[1], mo_b.conj().T)
        rdm1 = np.array((dm_a, dm_b))
        # ZHC NOTE update zs after dm is determined
        self.zs = get_zs(self.h_ep, self.w_p, rdm1)
        return rdm1

    def make_rdm1e(self, mo_coeff=None, mo_occ=None):
        return self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)

    make_rdm1p = make_rdm1p
    make_rdm1p_linear = make_rdm1p_linear

    def CSMP2(self, with_t2=False, make_rdm1=False, ao_repr=False):
        from polar.coherent import csmp
        rdm1 = self.make_rdm1()
        zs = self.zs
        h_ep = self.h_ep
        w_p = self.w_p
        e_mp2, t2 = csmp.kernel(self, h_ep, w_p, zs=zs,
                                with_t2=with_t2, make_rdm1=make_rdm1, ao_repr=ao_repr)
        if with_t2 or make_rdm1:
            return e_mp2, t2
        else:
            return e_mp2
