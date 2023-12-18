#!/usr/bin/env python

"""
Variational Lang-Firsov.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from functools import partial
import numpy as np
from scipy import linalg as la
from scipy import optimize as opt

from pyscf import gto, scf, ao2mo, lib
from pyscf.scf import hf
from pyscf.lib import logger
from polar.lang_firsov import grad
from polar.lang_firsov import thermal_average as ta
from polar.fci.fci import fc_factor

einsum = partial(np.einsum, optimize=True)

# ****************************************************************************
# Variational Lang-Firsov
# ****************************************************************************

def get_lf_ham(mylf, params=None, h0=None, h1=None, h2=None,
               h_ep=None, w_p=None):
    """
    Get h0, h1, h_ep, h2 of variational Lang-Firsov Hamiltonian.

    Args:
        mylf: LF object.
        lams: (nmode,)
        zs: (nmode,)
        h0: energy constant
        h1: (nao, nao)
        h2: (nao, nao, nao, nao), TODO, compact
        h_ep: constant
        w_p: (nmode,) or constant.

    Returns:
        H0: constant
        H1: (nao, nao)
        H_ep: (nmode, nao, nao)
        H2: (nao, nao, nao, nao)
    """
    if params is None:
        params = mylf.params
    if h0 is None:
        h0 = mylf.h0
    if h1 is None:
        h1 = mylf.h1
    if h2 is None:
        h2 = mylf.h2
    if h_ep is None:
        h_ep = mylf.h_ep
    if w_p is None:
        w_p = mylf.w_p
    lams, zs = mylf.unpack_params(params)

    nmode = mylf.nmode
    nao = mylf.nao
    g = h_ep
    fac = np.exp(-0.5 * lams**2)

    h1_diag = w_p*lams**2 - (2.0*g)*lams + (2.0*g)*zs - 2.0*w_p*zs*lams + h1[range(nao), range(nao)]
    H1 = einsum("ij, i, j -> ij", h1, fac, fac)
    H1[range(nao), range(nao)] = h1_diag

    H_ep = np.zeros((nmode, nao, nao))
    H_ep[range(nmode), range(nao), range(nao)] = g - w_p * lams

    H0 = h0 + (w_p * zs**2).sum()
    if h2 is not None:
        fac2 = np.exp(-2.0 * lams**2)
        H2_diag = (-4.0*g) * lams + (2.0 * w_p) * lams**2 + h2[range(nao), range(nao), range(nao), range(nao)]
        H2 = np.empty((nao, nao, nao, nao))
        for i in range(nao):
            for j in range(nao):
                for k in range(nao):
                    for l in range(nao):
                        if i == j and i == l and i == k:
                            # i = j = k = l
                            H2[i, j, k, l] = h2[i, j, k, l]
                        elif i == j and i == k:
                            # i = j = k != l
                            H2[i, j, k, l] = h2[i, j, k, l] * fac[l] * fac[k]
                        elif i == j and i == l:
                            # i = j = l != k
                            H2[i, j, k, l] = h2[i, j, k, l] * fac[l] * fac[k]
                        elif i == l and i == k:
                            # i = l = k != j
                            H2[i, j, k, l] = h2[i, j, k, l] * fac[i] * fac[j]
                        elif j == l and j == k:
                            # i != j = k = l
                            H2[i, j, k, l] = h2[i, j, k, l] * fac[i] * fac[j]

                        elif i == j:
                            if k == l:
                                H2[i, j, k, l] = h2[i, j, k, l]
                            else:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac[k] * fac[l]
                        elif i == k:
                            if j == l:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac2[i] * fac2[j]
                            else:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac2[i] * fac[j] * fac[l]
                        elif i == l:
                            if j == k:
                                H2[i, j, k, l] = h2[i, j, k, l]
                            else:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac[j] * fac[k]
                        elif j == k:
                            if i == l:
                                H2[i, j, k, l] = h2[i, j, k, l]
                            else:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac[i] * fac[l]
                        elif j == l:
                            if i == k:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac2[i] * fac2[j]
                            else:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac[i] * fac[k] * fac2[j]
                        elif k == l:
                            if i == j:
                                H2[i, j, k, l] = h2[i, j, k, l]
                            else:
                                H2[i, j, k, l] = h2[i, j, k, l] * fac[i] * fac[j]

                        else:
                            H2[i, j, k, l] = h2[i, j, k, l] * fac[i] * fac[j] * fac[k] * fac[l]

        H2[range(nao), range(nao), range(nao), range(nao)] = H2_diag
    else:
        H2 = h2
    return H0, H1, H2, H_ep, w_p

def solve_lf_ham(mylf, params=None, nelec=None, mp2=False, mp3=False, mp4=False,
                 nph=9, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                 scf_max_cycle=50, fci=False):
    H0, H1, H2, H_ep, w_p = mylf.get_lf_ham(params=params)

    ovlp = mylf.get_ovlp()
    nao = mylf.nao
    h1 = mylf.get_h1()
    h_ep_bare = mylf.get_h_ep()
    if nelec is None:
        nelec = mylf.nelec
    if params is None:
        params = mylf.params
    lams, zs = mylf.unpack_params(params)

    if H2 is not None:
        mf = hf.RHF(mylf.mol)
        mf.energy_nuc = lambda *args: H0
        mf.get_hcore = lambda *args: H1
        mf.get_ovlp = lambda *args: ovlp
        mf._eri = H2

        mf.direct_scf = False
        mf.max_cycle = scf_max_cycle
        mf.conv_tol = mylf.conv_tol * 0.1

        if scf_newton:
            mf = mf.newton()

        if beta < np.inf:
            from pyscf.pbc.scf.addons import smearing_
            mf = smearing_(mf, sigma=1.0/beta, method='fermi')

        e_tot = mf.kernel(dm0=dm0)
        rdm1 = mf.make_rdm1()

        mylf._scf = mf
        mylf.mo_energy = mf.mo_energy
        mylf.mo_coeff = mf.mo_coeff
        mylf.mo_occ = mf.mo_occ
        mylf.e_hf = float(e_tot)
        conv = mf.converged
        if fci:
            from pyscf import fci, ao2mo
            cisolver = fci.direct_nosym.FCI()
            cisolver.max_cycle = 100
            cisolver.conv_tol = 1e-8
            C = mf.mo_coeff
            h1_mo = C.conj().T @ mf.get_hcore() @ C
            h2_mo = ao2mo.kernel(mf._eri, C)
            e_tot, fcivec = cisolver.kernel(h1_mo, h2_mo, C.shape[-1], nelec, ecore=mf.energy_nuc())
            rdm1 = cisolver.make_rdm1(fcivec, C.shape[-1], (mylf.nelec_a, mylf.nelec_b))
            rdm1 = C @ rdm1 @ C.conj().T
    else:
        ew, ev = la.eigh(H1, ovlp)
        mo_occ = np.zeros(nao)
        if nelec == 1:
            nocc = nelec
            mo_occ[:nocc] = 1.0
        else:
            nocc = nelec_a
            mo_occ[:nocc] = 2.0
        e_tot = np.sum(ew * mo_occ) + H0
        nao, nmo = ev.shape
        rdm1 = np.dot(ev * mo_occ, ev.conj().T)
        mylf.mo_energy = ew
        mylf.mo_coeff = ev
        mylf.mo_occ = mo_occ
        mylf.e_hf = float(e_tot)
        conv = True

        if mp2 or mp3 or mp4:
            from polar.lang_firsov import mp as lfmp
            nocc = (np.asarray(mo_occ > 0.5)).sum()
            mo_energy = ew
            mo_coeff = ev
            if lams.ndim == 1:
                lf = 'lf'
            elif lams.ndim == 2:
                lf = 'glf'
            else:
                raise ValueError
            logger.info(mylf, "PT number of phonon: %d", nph)
            logger.info(mylf, "e_hf  %15.8f", e_tot)
            logger.info(mylf, "LF type %s", lf)

            if mp4:
                e_mp1, e_mp2, e_mp3, e_mp4 = lfmp.get_e_mp4(mylf.mol, h1, H_ep, w_p, lams, zs,
                                                            mo_coeff, mo_occ, mo_energy, nph,
                                                            lf=lf, h_ep_bare=h_ep_bare)
                e_tot += e_mp1
                e_tot += e_mp2
                e_tot += e_mp3
                e_tot += e_mp4
                mylf.e_mp1 = e_mp1
                mylf.e_mp2 = e_mp2
                mylf.e_mp3 = e_mp3
                mylf.e_mp4 = e_mp4

                logger.info(mylf, "e_mp1 %15.8f", e_mp1)
                logger.info(mylf, "e_mp2 %15.8f", e_mp2)
                logger.info(mylf, "e_mp3 %15.8f", e_mp3)
                logger.info(mylf, "e_mp4 %15.8f", e_mp4)
            elif mp3:
                e_mp1, e_mp2, e_mp3 = lfmp.get_e_mp3(mylf.mol, h1, H_ep, w_p, lams, zs,
                                                     mo_coeff, mo_occ, mo_energy, nph,
                                                     lf=lf, h_ep_bare=h_ep_bare)
                e_tot += e_mp1
                e_tot += e_mp2
                e_tot += e_mp3
                mylf.e_mp1 = e_mp1
                mylf.e_mp2 = e_mp2
                mylf.e_mp3 = e_mp3
                logger.info(mylf, "e_mp1 %15.8f", e_mp1)
                logger.info(mylf, "e_mp2 %15.8f", e_mp2)
                logger.info(mylf, "e_mp3 %15.8f", e_mp3)
            elif mp2 == 'slow':
                e_mp1, e_mp2 = lfmp.get_e_mp2_slow(mylf.mol, h1, H_ep, w_p, lams, zs,
                                                   mo_coeff, mo_occ, mo_energy, nph,
                                                   lf=lf, h_ep_bare=h_ep_bare)
                e_tot += e_mp1
                e_tot += e_mp2
                mylf.e_mp1 = e_mp1
                mylf.e_mp2 = e_mp2
                logger.info(mylf, "e_mp1 %15.8f", e_mp1)
                logger.info(mylf, "e_mp2 %15.8f", e_mp2)
            elif mp2:
                e_mp2 = lfmp.get_e_mp2(h1, H_ep, w_p, lams, zs, mo_coeff, mo_occ,
                                       mo_energy, nph)
                e_tot += e_mp2
                mylf.e_mp2 = e_mp2
                logger.info(mylf, "e_mp2 %15.8f", e_mp2)

        if verbose:
            logger.info(mylf, "e_tot %15.8f", e_tot)
            logger.info(mylf, "lams\n%s", lams)
            logger.info(mylf, "zs\n%s", zs)
            logger.info(mylf, "rdm1\n%s", rdm1)

    mylf.e_tot = e_tot
    return e_tot, rdm1

def solve_lf_ham_full(mylf, params=None, nelec=None, mp2=False, mp3=False, mp4=False,
                      nph=9, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                      scf_max_cycle=50, mo_coeff=None, mo_occ=None, canonicalization=True):
    if params is None:
        params = mylf.params_full
    kappa, lams, zs = mylf.unpack_params_full(params)
    params_p = mylf.pack_params(lams, zs)

    H0, H1, H2, H_ep, w_p = mylf.get_lf_ham(params=params_p)
    ovlp = mylf.get_ovlp()
    nao = mylf.nao
    h1 = mylf.get_h1()
    if nelec is None:
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
        mf.max_cycle = scf_max_cycle
        mf.conv_tol = mylf.conv_tol * 0.1

        dr = hf.unpack_uniq_var(kappa, mo_occ)
        mo_coeff = np.dot(mo_coeff, la.expm(dr))
        rdm1 = mf.make_rdm1(mo_coeff, mo_occ)
        e_tot = mf.energy_elec(dm=rdm1)[0] + mf.energy_nuc()
        fock = mf.get_fock(dm=rdm1)

        if canonicalization:
            print("-" * 79)
            mo_energy, mo_coeff = mf.canonicalize(mo_coeff, mo_occ, fock)
            homo = lumo = None
            mo_e_occ = mo_energy[mo_occ >= 1.0]
            mo_e_vir = mo_energy[mo_occ <  1.0]
            if len(mo_e_occ) > 0:
                homo = mo_e_occ.max()
            if len(mo_e_vir) > 0:
                lumo = mo_e_vir.min()
            if homo is not None:
                print ('HOMO = %15.8g'%(homo))
            if lumo is not None:
                print ('LUMO = %15.8g'%(lumo))
                if homo is not None:
                    print ("gap  = %15.8g"%(lumo - homo))
            if (lumo is not None) and (homo is not None) and (homo > lumo):
                print ('WARN: HOMO %s > LUMO %s was found in the canonicalized orbitals.'%(homo, lumo))
                print ("mo_energy:\n%s"%mo_energy)
            grad = mf.get_grad(mo_coeff, mo_occ, fock)
            grad_norm = la.norm(grad)
            print("-" * 79)
            print ("|g| = %15.8g" % grad_norm)
            print("-" * 79)
        else:
            mo_energy = einsum("pm, pq, qm -> m", mo_coeff.conj(), fock, mo_coeff)

        mylf._scf = mf
        mylf.e_hf = float(e_tot)
        conv = mf.converged
        mylf.mo_coeff = mf.mo_coeff = mo_coeff
        mylf.mo_occ = mf.mo_occ = mo_occ
        mylf.mo_energy = mf.mo_energy = mo_energy

    if mp2:
        from polar.lang_firsov import mp_glf
        logger.info(mylf, "LF-MP2 start, nph = %d", nph)
        ovlp_g = la.block_diag(ovlp, ovlp)
        hcore_g = la.block_diag(H1, H1)
        mylf._scf = mf = mf.to_ghf()
        mf.get_ovlp = lambda *args: ovlp_g
        mf.get_hcore = lambda *args: hcore_g
        e_mp2 = mp_glf.get_e_mp2(mylf, lams=lams, zs=zs, nph=nph)
        e_tot += e_mp2
        mylf.e_mp2 = e_mp2
        logger.info(mylf, "e_mp2 %15.8f", mylf.e_mp2)

    return e_tot, rdm1

def kernel(mylf, params=None, nelec=None, method='BFGS', conv_tol=None,
           max_cycle=None, gtol=None, mp2=False, mp3=False, mp4=False, nph=9,
           ntrial=None, use_num_grad=True, full_opt=False, mo_coeff=None,
           mo_occ=None, fci=False, scf_max_cycle=50, beta=np.inf):

    mylf.dump_flags()

    if params is None:
        if full_opt:
            params = mylf.params_full
        else:
            params = mylf.params
    if nelec is None:
        nelec = mylf.nelec

    if conv_tol is None:
        conv_tol = mylf.conv_tol

    if gtol is None:
        gtol = np.sqrt(conv_tol)

    if max_cycle is None:
        max_cycle = mylf.max_cycle

    if ntrial is None:
        ntrial = mylf.ntrial

    if use_num_grad:
        jac = None
    else:
        if full_opt:
            def jac(params):
                return mylf.get_grad_full(params, mo_coeff=mo_coeff, mo_occ=mo_occ)
        else:
            def jac(params):
                return mylf.get_grad(params, scf_max_cycle=scf_max_cycle, fci=fci, beta=beta)

    if full_opt:
        def cost_func(params):
            return mylf.solve_lf_ham_full(params, nelec, mo_coeff=mo_coeff,
                                          mo_occ=mo_occ, canonicalization=False)[0]
    else:
        def cost_func(params):
            return mylf.solve_lf_ham(params, nelec, scf_max_cycle=scf_max_cycle,
                                     fci=fci, beta=beta)[0]

    params_opt = None
    e_tot = 1e+9
    for i in range(ntrial):
        print ("trial %5d / %5d"%(i, ntrial), flush=True)
        if i == 1:
            params = np.zeros_like(params)
        elif i >= 2:
            params = (np.random.random(params.shape) - 0.5) * (np.max(mylf.get_h_ep()) / np.max(mylf.get_w_p()))
            if i % 2== 0:
                params *= 0.1

        res = opt.minimize(cost_func, params, jac=jac, method=method, tol=conv_tol,
                           options={"disp": True, "maxiter": max_cycle,
                                    "gtol": gtol})

        if res.fun < e_tot:
            params_opt = res.x
            e_tot = res.fun

    mylf.params_full_opt = params_opt
    mylf.params_opt = np.array(params_opt[-mylf.nparam:], copy=True)
    mylf.e_tot = e_tot

    if full_opt:
        mylf.e_tot, rdm1 = mylf.solve_lf_ham_full(params_opt, nelec, mp2=mp2, mp3=mp3, mp4=mp4,
                                                  nph=nph, verbose=True, mo_coeff=mo_coeff,
                                                  mo_occ=mo_occ)
        kappa, lams, zs = mylf.unpack_params_full(mylf.params_full_opt)
    else:
        mylf.e_tot, rdm1 = mylf.solve_lf_ham(params_opt, nelec, mp2=mp2, mp3=mp3, mp4=mp4,
                                             nph=nph, verbose=True,
                                             fci=fci, scf_max_cycle=scf_max_cycle, beta=beta)
        lams, zs = mylf.unpack_params(mylf.params_opt)
        kappa = None

    logger.info(mylf, "e_tot %15.8f", mylf.e_tot)
    logger.info(mylf, "kappa\n%s", kappa)
    logger.info(mylf, "lams\n%s", lams)
    logger.info(mylf, "zs\n%s", zs)
    logger.info(mylf, "rdm1\n%s", rdm1)
    return mylf.e_tot

class LangFirsov(object):
    conv_tol = 1e-8
    conv_tol_grad = None
    max_cycle = 1000
    ntrial = 5

    def __init__(self, h0, h1, h2, h_ep, w_p, nelec, spin=0, params=None,
                 uniform=False, lams_only=False, zs_only=False, verbose=4, mol=None):
        self.mol = gto.Mole(verbose=verbose)
        self.mol.build(dump_input=False)
        self.verbose = verbose
        self.max_memory = self.mol.max_memory
        self.stdout = self.mol.stdout

        self.h0 = h0
        self.h1 = h1
        self.h2 = h2
        self.h_ep = h_ep
        self.w_p = w_p

        self.nelec = nelec
        self.mol.nelectron = nelec
        self.mol.tot_electrons = lambda *args: self.nelec
        self.mol.incore_anyway = True
        if self.nelec == 1:
            self.spin = self.mol.spin = 1
        else:
            self.spin = self.mol.spin = spin

        self.nelec_a = (self.nelec + self.spin) // 2
        self.nelec_b = (self.nelec - self.spin) // 2
        assert self.nelec_a + self.nelec_b == self.nelec

        self.nmode = len(self.w_p)
        self.nao = self.h1.shape[-1]
        self.ovlp = np.eye(self.nao)

        self.uniform = uniform
        self.lams_only = lams_only
        self.zs_only = zs_only
        self.lf_type = 'lf'

        if params is None:
            self.params = self.get_init_params()
        else:
            self.params = params
        assert len(self.params) == self.nparam

        self.params_full = np.zeros(self.nparam_full)
        self.params_full[-self.nparam:] = self.params

        # results:
        self.chkfile = None
        self.params_opt = None
        self.params_full_opt = None
        self.e_tot = None
        self.e_hf  = None
        self.e_mp1 = None
        self.e_mp2 = None
        self.e_mp3 = None
        self.e_mp4 = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self

        log.info('\n')
        log.info('******** %s ********', self.__class__)
        method = [self.__class__.__name__]
        log.info('method  = %s', '-'.join(method))
        log.info("uniform = %s", self.uniform)
        log.info("nao     = %10d", self.nao)
        log.info("nelec   = %10s", self.nelec)
        log.info("nmode   = %10d", self.nmode)
        log.info("nparam  = %10d", self.nparam)

        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('max_cycles = %d', self.max_cycle)
        log.info("ntrial: %d", self.ntrial)

        if isinstance(self.h_ep, np.ndarray) and (self.nao > 16 or self.nmode > 16):
            log.info("h_ep:\n%s min %15.6f max %15.6f",
                     str(self.h_ep.shape), np.min(self.h_ep), np.max(self.h_ep))
        else:
            log.info("h_ep:\n%s", self.h_ep)

        if self.nmode > 16:
            log.info("w_p:\n%s min %15.6f max %15.6f",
                     str(self.w_p.shape), np.min(self.w_p), np.max(self.w_p))
        else:
            log.info("w_p:\n%s", self.w_p)

        lams, zs = self.unpack_params(self.params)
        log.info("lams shape: %s", str(lams.shape))
        log.info("lams:\n%s", lams)
        log.info("zs shape:   %s", str(zs.shape))
        log.info("zs:\n%s", zs)
        if self.chkfile:
            log.info('chkfile to save SCF result = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def get_h0(self):
        return self.h0

    def energy_nuc(self):
        return self.h0

    def get_ovlp(self):
        return self.ovlp

    def get_h1(self):
        return self.h1

    def get_hcore(self):
        return self.h1

    def get_h2(self):
        return self.h2

    def get_h_ep(self):
        return self.h_ep

    def get_w_p(self):
        return self.w_p

    def get_dm0(self, key='minao'):
        """
        get initial rdm1.
        """
        if self.mol.natm == 0:
            h1e = self.get_h1()
            s1e = self.get_ovlp()
            mo_energy, mo_coeff = la.eigh(h1e, s1e)
            mo_occ = np.zeros_like(mo_energy)
            mo_occ[:self.nelec_a] = 2.0
            dm0 = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
        else:
            if key == 'minao':
                dm0 = hf.init_guess_by_minao(self.mol)
            elif key == 'atom':
                dm0 = hf.init_guess_by_atom(self.mol)
            else:
                raise ValueError
        return dm0

    def get_init_params(self, scale=0.1):
        h_ep = self.h_ep
        w_p = self.w_p

        if self.zs_only:
            lams = np.array([])
        elif self.uniform:
            lams = np.zeros(self.nmode)
            lams[:] = (np.random.random() - 0.5) * (np.max(h_ep) / np.max(w_p) * scale)
        else:
            lams = (np.random.random(self.nlams) - 0.5) * (np.max(h_ep) / np.max(w_p) * scale)

        if self.lams_only:
            zs = np.array([])
        elif self.zs_only:
            zs = np.random.random(self.nzs)
        else:
            dm0 = self.get_dm0()
            zs = np.einsum("p, pp -> p", lams, dm0)

        params = np.append(lams, zs)

        if self.uniform:
            if self.lams_only or self.zs_only:
                params = params[[-1]]
            else:
                params = params[[0, self.nlams]]
        return params

    def unpack_params(self, params, uniform=None, lams_only=None, zs_only=None):
        if lams_only is None:
            lams_only = self.lams_only
        if zs_only is None:
            zs_only = self.zs_only
        if uniform is None:
            uniform = self.uniform
        nmode = self.nmode
        nao = self.nao

        if lams_only:
            zs = np.array([], dtype=params.dtype)
            if uniform:
                l = params
                if isinstance(self, GGLangFirsov):
                    lams = np.zeros((nmode, nao, nao), dtype=params.dtype)
                    lams[range(nmode), range(nao), range(nao)] = l
                elif isinstance(self, GLangFirsov):
                    lams = np.zeros((nmode, nao), dtype=params.dtype)
                    lams[range(nmode), range(nao)] = l
                elif isinstance(self, LangFirsov):
                    lams = np.zeros(nmode, dtype=params.dtype)
                    lams[:] = l
                else:
                    raise ValueError("unknown lf type %s"%(self))
            else:
                lams = np.array(params.reshape(nmode, -1), copy=True)
                if isinstance(self, GGLangFirsov):
                    lams = lib.unpack_tril(lams)
                    if lams.shape != (nmode, nao, nao):
                        raise ValueError("lams shape %s does not match %s"
                                         %(str(lams.shape), (nmode, nao, nao)))
                elif isinstance(self, GLangFirsov):
                    pass
                elif isinstance(self, LangFirsov):
                    lams = lams.reshape(nmode)
                else:
                    raise ValueError("unknown lf type %s"%(self))
        elif zs_only:
            if uniform:
                z = params
                zs = np.zeros(nmode, dtype=params.dtype)
                zs[:] = z
            else:
                zs = np.array(params[-nmode:], copy=True)

            if isinstance(self, GGLangFirsov):
                lams = np.zeros((nmode, nao, nao), dtype=params.dtype)
            elif isinstance(self, GLangFirsov):
                lams = np.zeros((nmode, nao), dtype=params.dtype)
            elif isinstance(self, LangFirsov):
                lams = np.zeros(nmode, dtype=params.dtype)
            else:
                raise ValueError("unknown lf type %s"%(self))
        else:
            if uniform:
                l, z = params
                zs = np.zeros(nmode, dtype=params.dtype)
                zs[:] = z
                if isinstance(self, GGLangFirsov):
                    lams = np.zeros((nmode, nao, nao), dtype=params.dtype)
                    lams[range(nmode), range(nao), range(nao)] = l
                elif isinstance(self, GLangFirsov):
                    lams = np.zeros((nmode, nao), dtype=params.dtype)
                    lams[range(nmode), range(nao)] = l
                elif isinstance(self, LangFirsov):
                    lams = np.zeros(nmode, dtype=params.dtype)
                    lams[:] = l
                else:
                    raise ValueError("unknown lf type %s"%(self))
            else:
                zs = np.array(params[-nmode:], copy=True)
                lams = np.array(params[:-nmode].reshape(nmode, -1), copy=True)
                if isinstance(self, GGLangFirsov):
                    lams = lib.unpack_tril(lams)
                    if lams.shape != (nmode, nao, nao):
                        raise ValueError("lams shape %s does not match %s"
                                         %(str(lams.shape), (nmode, nao, nao)))
                elif isinstance(self, GLangFirsov):
                    pass
                elif isinstance(self, LangFirsov):
                    lams = lams.reshape(nmode)
                else:
                    raise ValueError("unknown lf type %s"%(self))
        return lams, zs

    def pack_params(self, lams, zs):
        if self.lams_only:
            if self.uniform:
                params = np.array((lams[0],))
            else:
                params = np.hstack((lams.ravel(),))
        elif self.zs_only:
            if self.uniform:
                params = np.array((zs[0],))
            else:
                params = np.hstack((zs.ravel(),))
        else:
            if self.uniform:
                params = np.array((lams[0], zs[0]))
            else:
                params = np.hstack((lams.ravel(), zs.ravel()))
        return params

    def unpack_params_full(self, params, uniform=None):
        nocc = self.nelec_a
        nvir = self.nao - nocc
        kappa = params[:nvir*nocc]
        lams, zs = self.unpack_params(params[nvir*nocc:])
        return kappa, lams, zs

    def pack_params_full(self, kappa, lams, zs):
        return np.hstack((kappa.ravel(), self.pack_params(lams, zs).ravel()))

    @property
    def nlams(self):
        if self.zs_only:
            nlams = 0
        elif self.uniform:
            nlams = 1
        else:
            nlams = self.nmode
        return nlams

    @property
    def nzs(self):
        if self.lams_only:
            nzs = 0
        elif self.uniform:
            nzs = 1
        else:
            nzs = self.nmode
        return nzs

    @property
    def nparam(self):
        nparam = self.nlams + self.nzs
        return nparam

    @property
    def nkappa(self):
        nocc = self.nelec_a
        nvir = self.nao - nocc
        nparam = nvir * nocc
        return nparam

    @property
    def nparam_full(self):
        nparam = int(self.nkappa)
        nparam += self.nparam
        return nparam

    def get_lams_zs(self, opt=True):
        if opt:
            return self.unpack_params(self.params_opt)
        else:
            return self.unpack_params(self.params)

    get_lf_ham = get_lf_ham

    solve_lf_ham = solve_lf_ham

    solve_lf_ham_full = solve_lf_ham_full

    get_grad = grad.get_grad_lf

    get_grad_full = grad.get_grad_lf_full

    kernel = kernel

    @staticmethod
    def fc_factor(n, l, m):
        return fc_factor(n, l, m)

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        return np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)

    def get_fc1(self, lams=None):
        if lams is None:
            lams, zs = self.get_lams_zs(opt=True)
        tmp = np.exp(lams ** 2 * (-0.5))
        fc1 = np.einsum("p, q -> pq", tmp, tmp)
        nao = self.nao
        fc1[range(nao), range(nao)] = 0.0
        return fc1

    def make_rdm1e(self, mo_coeff=None, mo_occ=None, lams=None):
        """
        Electronic part of rdm1.
        gamma_pq = <LF | a^{\dag}_q a_p |LF>
        """
        rdm1 = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        fc1 = self.get_fc1(lams=lams)
        rdm1 *= fc1
        return rdm1

    def make_rdm1p(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        raise NotImplementedError

    def make_rdm1p_linear(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon linear part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        raise NotImplementedError


# ****************************************************************************
# Generalized Variational Lang-Firsov
# ****************************************************************************

def get_glf_ham(mylf, params=None, h0=None, h1=None, h2=None,
                h_ep=None, w_p=None, ph_str_l=None, ph_str_r=None):
    """
    Generalized version of LF transform.
    h_ep do not include non-local coupling.

    Args:
        params: (nmode * nao + nmode), include lams and zs
                lams: (nmode, nao, nao)
                zs: (nmode,)
        h0: float
        h1: (nao, nao)
        h2: (nao, nao, nao, nao)
        h_ep: (nmode, nao)
        w_p: (nmode,)

    Returns:
        H0: float
        H1: (nao, nao)
        H2: (nao, nao, nao, nao)
        H_ep: (nmode, nao, nao)
        w_p: (nmode,)
    """
    if h_ep is None:
        h_ep = mylf.h_ep
    if h_ep.ndim == 3:
        return get_glf_ham_non_loc_h_ep(mylf, params=params, h0=h0, h1=h1, h2=h2,
                                        h_ep=h_ep, w_p=w_p, ph_str_l=ph_str_l,
                                        ph_str_r=ph_str_r)
    if params is None:
        params = mylf.params
    if h0 is None:
        h0 = mylf.h0
    if h1 is None:
        h1 = mylf.h1
    if h2 is None:
        h2 = mylf.h2
    if w_p is None:
        w_p = mylf.w_p

    lams, zs = mylf.unpack_params(params)

    nmode = mylf.nmode
    nao = mylf.nao

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    diff **= 2
    fac = diff.sum(axis=0)
    diff = None
    fac *= (-0.5)
    fac = np.exp(fac)
    H1 = h1 * fac

    G = h_ep - np.einsum('x, xp -> xp', w_p, lams)
    v = einsum('x, xp, xp -> p', w_p, lams, lams) - 2.0 * einsum('xp, xp -> p', h_ep, lams)
    veff = v + 2.0 * np.einsum('xp, x -> p', G, zs)

    H1[range(nao), range(nao)] += veff
    H_ep = np.zeros((nmode, nao, nao))
    H_ep[:, range(nao), range(nao)] = G

    H0 = h0 + (w_p * zs**2).sum()

    if h2 is not None:
        fac = 0.0
        for x in range(nmode):
            diff = lib.direct_sum('q - p + s - r -> pqrs', lams[x], lams[x], lams[x], lams[x])
            diff **= 2
            fac += diff
            diff = None
        fac *= (-0.5)
        fac = np.exp(fac)
        H2 = h2 * fac

        v_pq = np.einsum('x, xp, xq -> pq', w_p * 2.0, lams, lams)
        v_pq -= np.einsum('xp, xq -> pq', h_ep, lams * 4.0)
        # ZHC NOTE symmetrization
        # ZHC FIXME should H2 also be permutationally symmetric?
        v_pq = (v_pq + v_pq.conj().T) * 0.5
        for p in range(nao):
            H2[p, p, range(nao), range(nao)] += v_pq[p]
    else:
        H2 = h2
    return H0, H1, H2, H_ep, w_p

def get_glf_ham_non_loc_h_ep(mylf, params=None, h0=None, h1=None, h2=None,
                             h_ep=None, w_p=None, ph_str_l=None, ph_str_r=None):
    """
    Generalized version of LF transform.
    h_ep include non-local coupling.

    Args:
        params: (nmode * nao + nmode), include lams and zs
                lams: (nmode, nao, nao)
                zs: (nmode,)
        h0: float
        h1: (nao, nao)
        h2: (nao, nao, nao, nao)
        h_ep: (nmode, nao, nao)
        w_p: (nmode,)

    Returns:
        H0: float
        H1: (nao, nao)
        H2: (nao, nao, nao, nao)
        H_ep: (nmode, nao, nao)
        w_p: (nmode,)
    """
    if params is None:
        params = mylf.params
    if h0 is None:
        h0 = mylf.h0
    if h1 is None:
        h1 = mylf.h1
    if h2 is None:
        h2 = mylf.h2
    if h_ep is None:
        h_ep = mylf.h_ep
    if w_p is None:
        w_p = mylf.w_p
    lams, zs = mylf.unpack_params(params)

    nmode = mylf.nmode
    nao = mylf.nao

    diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
    if (ph_str_l is None) or (ph_str_r is None):
        f00 = fc_factor(0, 0, diff)
        #f01 = fc_factor(0, 1, diff)
    else:
        f00 = np.zeros((nmode, nao, nao))
        for x in range(nmode):
            f00[x] = fc_factor(ph_str_l[x], ph_str_r[x], diff[x])

    diff = None
    fc1 = np.prod(f00, axis=0)

    # term 0 [z]
    H0 = h0 + (w_p * zs**2).sum()

    # term 1 [l]
    H1 = h1 * fc1
    G = h_ep * fc1

    # term 2, 3 [l] zero
    #fac_01 = f01 * fc1
    #fac_01 /= f00
    #fac_10 = -f01 * fc1
    #fac_10 /= f00
    # since fac_01 and fac_10 have opposite signs

    # term 4 [l, z]
    v4 = np.einsum('x, xpq -> pq', zs * 2.0, G)
    H1 += v4
    v4 = None

    # term 5, 6 [l]
    v56 = np.einsum('xpq, xq -> pq', G, lams)
    H1 -= v56
    H1 -= v56.conj().T
    v56 = None

    # term 7 [l]
    H1[range(nao), range(nao)] += np.einsum('x, xp -> p', w_p, lams**2)

    # term 11 [l, z]
    H1[range(nao), range(nao)] -= np.einsum('x, xp -> p', w_p * zs * 2.0, lams)

    H_ep = np.array(G, copy=True)
    H_ep[:, range(nao), range(nao)] -= np.einsum('x, xp -> xp', w_p, lams)

    if h2 is not None:
        fac = 0.0
        for x in range(nmode):
            diff = lib.direct_sum('q - p + s - r -> pqrs', lams[x], lams[x], lams[x], lams[x])
            diff **= 2
            fac += diff
            diff = None
        fac *= (-0.5)
        fac = np.exp(fac)
        H2 = fac
        # term 11
        H2 *= h2

        # term 10
        v_pq = einsum('x, xp, xq -> pq', w_p * 2.0, lams, lams)
        for p in range(nao):
            H2[p, p, range(nao), range(nao)] += v_pq[p]

        # term 8, 9
        v_pqr = np.einsum('xpq, xr -> pqr', G, lams * (-2.0))
        for r in range(nao):
            H2[:, :, r, r] += v_pqr[:, :, r]
            H2[r, r, :, :] += v_pqr[:, :, r]
    else:
        H2 = h2

    return H0, H1, H2, H_ep, w_p

class GLangFirsov(LangFirsov):
    def __init__(self, mol, h_ep, w_p, h0=None, h1=None, h2=None, ovlp=None, nelec=None,
                 spin=0, params=None, uniform=False, lams_only=False, zs_only=False, aosym='s1',
                 verbose=4):
        """
        Generalized LF.
        """
        if mol is None:
            self.mol = gto.Mole(verbose=verbose)
            self.mol.build(dump_input=False)
        else:
            self.mol = mol
        self.verbose = verbose
        self.max_memory = self.mol.max_memory
        self.stdout = self.mol.stdout

        if h0 is None:
            self.h0 = self.mol.energy_nuc()
        else:
            self.h0 = h0

        if h1 is None:
            self.h1 = hf.get_hcore(self.mol)
        else:
            self.h1 = h1

        if ovlp is None:
            if mol is None:
                self.ovlp = np.eye(self.h1.shape[-1])
            else:
                self.ovlp = hf.get_ovlp(self.mol)
        else:
            self.ovlp = ovlp

        if h2 is None:
            if mol is None:
                self.h2 = h2
            else:
                self.h2 = self.mol.intor('int2e', aosym=aosym)
        else:
            self.h2 = h2

        self.h_ep = h_ep
        self.w_p = w_p

        if nelec is None:
            self.nelec = self.mol.nelectron
            self.mol.tot_electrons = lambda *args: self.nelec
        else:
            self.nelec = nelec
            self.mol.tot_electrons = lambda *args: self.nelec
        self.mol.nelectron = nelec
        self.mol.incore_anyway = True

        if self.nelec == 1:
            self.spin = self.mol.spin = 1
        else:
            self.spin = self.mol.spin = spin
        self.nelec_a = (self.nelec + self.spin) // 2
        self.nelec_b = (self.nelec - self.spin) // 2
        assert self.nelec_a + self.nelec_b == self.nelec

        self.nmode = len(self.w_p)
        self.nao = self.h1.shape[-1]

        if self.h_ep.shape == (self.nmode, self.nao):
            self.lf_type = 'glf'
        elif self.h_ep.shape == (self.nmode, self.nao, self.nao):
            self.lf_type = 'glf'
        else:
            raise ValueError("h_ep shape %s is not supported."%(str(self.h_ep.shape)))

        self.uniform = uniform
        self.lams_only = lams_only
        self.zs_only = zs_only

        if params is None:
            self.params = self.get_init_params()
        else:
            self.params = params
        assert len(self.params) == self.nparam

        self.params_full = np.zeros(self.nparam_full)
        self.params_full[-self.nparam:] = self.params

        # results:
        self.chkfile = None
        self.params_opt = None
        self.params_full_opt = None
        self.e_tot = None
        self.e_hf  = None
        self.e_mp1 = None
        self.e_mp2 = None
        self.e_mp3 = None
        self.e_mp4 = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None

    @property
    def nlams(self):
        if self.zs_only:
            nlams = 0
        elif self.uniform:
            nlams = 1
        else:
            nlams = self.nmode * self.nao
        return nlams

    def pack_params(self, lams, zs):
        if self.lams_only:
            if self.uniform:
                params = np.array((lams[0, 0],))
            else:
                params = np.hstack((lams.ravel(),))
        elif self.zs_only:
            if self.uniform:
                params = np.array((zs[0],))
            else:
                params = np.hstack((zs.ravel(),))
        else:
            if self.uniform:
                params = np.array((lams[0, 0], zs[0]))
            else:
                params = np.hstack((lams.ravel(), zs.ravel()))
        return params

    def get_init_params(self, scale=0.1):
        h_ep = self.h_ep
        w_p = self.w_p
        if self.zs_only:
            lams = np.array([])
        elif self.uniform:
            val = (np.random.random() - 0.5) * (np.max(h_ep) / np.max(w_p) * scale)
            lams = np.zeros((self.nmode, self.nao))
            lams[range(self.nmode), range(self.nao)] = val
        else:
            lams = (np.random.random(self.nlams) - 0.5) * (np.max(h_ep) / np.max(w_p) * scale)
            lams = lams.reshape(self.nmode, self.nao)

        if self.lams_only:
            zs = np.array([])
        elif self.zs_only:
            zs = np.random.random(self.nzs)
        else:
            dm0 = self.get_dm0()
            fc1 = self.get_fc1(lams=lams)
            if self.h_ep.shape == (self.nmode, self.nao):
                zs = np.einsum("yp, pp -> y", lams, dm0) - \
                     np.einsum("yp, pp, pp -> y", h_ep, fc1, dm0) / w_p
            else:
                zs = np.einsum("yp, pp -> y", lams, dm0) - \
                     np.einsum("ypq, pq, qp -> y", h_ep, fc1, dm0) / w_p

        params = np.append(lams.ravel(), zs)

        if self.uniform:
            if self.lams_only or self.zs_only:
                params = params[[-1]]
            else:
                params = params[[0, self.nlams]]

        return params

    def get_fc1(self, lams=None):
        if lams is None:
            lams, zs = self.get_lams_zs(opt=True)
        diff = lib.direct_sum('xq - xp -> xpq', lams, lams)
        f00 = fc_factor(0, 0, diff)
        fc1 = np.prod(f00, axis=0)
        return fc1

    def make_rdm1p(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        if lams is None or zs is None:
            lams, zs = self.get_lams_zs(opt=True)
        rdm1 = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        nao = self.nao
        rdm1_diag = rdm1[range(nao), range(nao)]
        rho = np.einsum("y, x -> xy", zs, zs)

        tmp = np.einsum("xp, p -> x", lams, rdm1_diag)
        tmp = np.einsum("y, x -> xy", zs, tmp)
        rho -= tmp
        rho -= tmp.conj().T

        rho += np.einsum("yp, xp, p -> xy", lams, lams, rdm1_diag, optimize=True)
        tmp = np.einsum("p, q -> pq", rdm1_diag, rdm1_diag)
        tmp -= 0.5 * np.einsum("qp, pq -> pq", rdm1, rdm1)
        rho += np.einsum("yp, xp, pq -> xy", lams, lams, tmp, optimize=True)
        return rho

    def make_rdm1p_linear(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon linear part of rdm1.
        rho_x = <LF | b_x |LF>
        """
        if lams is None or zs is None:
            lams, zs = self.get_lams_zs(opt=True)
        rdm1 = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        nao = self.nao
        rdm1_diag = rdm1[range(nao), range(nao)]
        rho = zs - np.einsum("xp, p -> x", lams, rdm1_diag)
        return rho

    get_lf_ham = get_glf_ham

    get_grad = grad.get_grad_glf

def get_gglf_ham(mylf, params=None, h0=None, h1=None, h2=None,
                 h_ep=None, w_p=None):
    """
    Generalized version of LF transform.
    lams include non-local coupling.

    Args:
        params: (nmode*nao*(nao+1)//2 + nmode), include lams and zs
                lams: (nmode, nao, nao)
                zs: (nmode,)
        h0: float
        h1: (nao, nao)
        h2: (nao, nao, nao, nao)
        h_ep: (nmode, nao, nao)
        w_p: (nmode,)

    Returns:
        H0: float
        H1: (nao, nao)
        H2: (nao, nao, nao, nao)
        H_ep: (nmode, nao, nao)
        w_p: (nmode,)
    """
    if params is None:
        params = mylf.params
    if h0 is None:
        h0 = mylf.h0
    if h1 is None:
        h1 = mylf.h1
    if h2 is None:
        h2 = mylf.h2
    if h_ep is None:
        h_ep = mylf.h_ep
    if w_p is None:
        w_p = mylf.w_p
    lams, zs = mylf.unpack_params(params)

    nmode = mylf.nmode
    nao = mylf.nao

    # term 0 [z]
    H0 = h0 + (w_p * zs**2).sum()

    # term 1 [l]
    #H1 = h1 * fc1
    H1 = ta.bch_h1_exp(h1, lams)
    #G = h_ep * fc1
    G = np.empty_like(h_ep)
    for x in range(nmode):
        G[x] = ta.bch_h1_exp(h_ep[x], lams)

    # term 2, 3 [l] zero
    #fac_01 = f01 * fc1
    #fac_01 /= f00
    #fac_10 = -f01 * fc1
    #fac_10 /= f00
    # since fac_01 and fac_10 have opposite signs

    # term 4 [l, z]
    v4 = np.einsum('x, xpq -> pq', zs * 2.0, G)
    H1 += v4
    v4 = None

    # term 5, 6 [l]
    v56 = np.einsum('xpq, xqs -> ps', G, lams)
    H1 -= v56 + v56.conj().T
    v56 = None

    # term 7 [l]
    H1 += np.einsum('x, xpq, xqs -> ps', w_p, lams, lams)

    # term 11 [l, z]
    H1 -= np.einsum('x, xpq -> pq', w_p * zs * 2.0, lams)

    H_ep = G - np.einsum('x, xpq -> xpq', w_p, lams)

    if h2 is not None:
        H2 = ta.bch_h2_exp(h2, lams)

        # term 10
        H2 += einsum('x, xpq, xrs -> pqrs', w_p * 2.0, lams, lams)

        # term 8, 9
        v_pqrs = np.einsum('xpq, xrs -> pqrs', G, lams * (-2.0))
        H2 += v_pqrs
        H2 += v_pqrs.transpose(2, 3, 0, 1)
    else:
        H2 = h2

    assert la.norm(H1 - H1.conj().T) < 1e-10
    return H0, H1, H2, H_ep, w_p

class GGLangFirsov(GLangFirsov):
    """
    Most generalized LF.
    lams has shape (nmode, nao, nao).
    """
    @property
    def nlams(self):
        if self.zs_only:
            nlams = 0
        elif self.uniform:
            nlams = 1
        else:
            nlams = self.nmode * self.nao * (self.nao+1) // 2
        return nlams

    def pack_params(self, lams, zs):
        if self.lams_only:
            if self.uniform:
                params = np.array((lams[0, 0, 0],))
            else:
                params = np.hstack((lib.pack_tril(lams).ravel(),))
        elif self.zs_only:
            if self.uniform:
                params = np.array((zs[0],))
            else:
                params = np.hstack((zs.ravel(),))
        else:
            if self.uniform:
                params = np.array((lams[0, 0, 0], zs[0]))
            else:
                params = np.hstack((lib.pack_tril(lams).ravel(), zs.ravel()))
        return params

    def get_init_params(self, scale=0.1):
        h_ep = self.h_ep
        w_p = self.w_p
        if self.zs_only:
            lams = np.array([])
        elif self.uniform:
            val = (np.random.random() - 0.5) * (np.max(h_ep) / np.max(w_p) * scale)
            lams = np.zeros((self.nmode, self.nao, self.nao))
            lams[range(self.nmode), range(self.nao), range(self.nao)] = val
        else:
            lams = (np.random.random(self.nlams) - 0.5) * (np.max(h_ep) / np.max(w_p) * scale)
            lams = lib.unpack_tril(lams.reshape(self.nmode, -1))

        dm0 = self.get_dm0()
        if self.lams_only:
            zs = np.array([])
        elif self.zs_only:
            zs = np.random.random(self.nzs)
        else:
            #lams_diag = lams[:, range(self.nao), range(self.nao)]
            #diff = lib.direct_sum('xq - xp -> xpq', lams_diag, lams_diag)
            #f00 = fc_factor(0, 0, diff)
            #fc1 = np.prod(f00, axis=0)
            #zs = np.einsum("yp, pp -> y", lams_diag, dm0) - \
            #     np.einsum("ypq, pq, qp -> y", h_ep, fc1, dm0) / w_p
            fc1 = self.get_fc1(lams=lams)
            zs = np.einsum("ypq, qp -> y", lams, dm0) - \
                 np.einsum("ypq, pq, qp -> y", h_ep, fc1, dm0) / w_p

        if self.zs_only:
            params = zs
        else:
            params = np.append(lib.pack_tril(lams).ravel(), zs)

        if self.uniform:
            if self.lams_only or self.zs_only:
                params = params[[-1]]
            else:
                params = params[[0, self.nlams]]
        return params

    def get_fc1(self, lams=None):
        if lams is None:
            lams, zs = self.get_lams_zs(opt=True)
        h1 = np.eye(self.nao)
        fc1 = ta.bch_h1_exp(h1, lams)
        return fc1

    def make_rdm1p(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        raise NotImplementedError

    def make_rdm1p_linear(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon linear part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        raise NotImplementedError

    get_lf_ham = get_gglf_ham

    get_grad = grad.get_grad_gglf

LF = LangFirsov

GLF = GLangFirsov

GGLF = GGLangFirsov


if __name__ == "__main__":
    np.set_printoptions(4, linewidth=1000, suppress=True)
    np.random.seed(10086)

    nao = 4
    nmode = nao
    nelec = 4
    U = 2.0

    hcore = np.zeros((nao, nao))
    for i in range(nao-1):
        hcore[i,i+1] = hcore[i+1,i] = -1.0
    hcore[nao-1, 0] = hcore[0, nao-1] = -1.0  # PBC

    eri = np.zeros((nao, nao, nao, nao))
    eri[range(nao), range(nao), range(nao), range(nao)] = U

    #alpha_list = np.arange(1.0, 1.05, 0.4)
    alpha_list = np.arange(2.4, 2.45, 0.4)

    w_p_list = [0.5]
    uniform = False
    glf = 'lf'
    mp2 = True
    mp3 = False
    mp4 = False

    for w_p in w_p_list:
        print ("*" * 79)
        e_col = []
        e_mp2_col = []
        ntrial = 3

        w_p_arr = np.zeros((nmode,))
        w_p_arr[:] = w_p

        for alpha in alpha_list:
            g = np.sqrt(alpha * w_p)
            #h_ep = np.zeros((nmode, nao))
            #for x in range(nmode):
            #    h_ep[x, x] = g
            h_ep = np.zeros((nmode, nao, nao))
            for x in range(nmode):
                h_ep[x, x, x] = g

            if nelec == 1:
                eri = None
            if glf == 'lf':
                mylf = LangFirsov(h0=0.0, h1=hcore, h2=eri, h_ep=g, w_p=w_p_arr,
                                  nelec=nelec, uniform=uniform)
                use_num_grad = False
            elif glf == 'glf':
                mylf = GLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=h_ep, w_p=w_p_arr,
                                   nelec=nelec, uniform=uniform)
                mp2 = False
                use_num_grad = False
            elif glf == 'gglf':
                mylf = GGLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=h_ep, w_p=w_p_arr,
                                    nelec=nelec, uniform=uniform)
                mp2 = False
                use_num_grad = True
            else:
                raise ValueError

            if nelec == 1:
                e = mylf.kernel(mp2=mp2, mp3=mp3, mp4=mp4, nph=9, ntrial=ntrial,
                                use_num_grad=use_num_grad)
            else:
                params = np.random.random(mylf.nparam_full)

                nocc = nelec // 2
                nvir = nao - nocc

                mylf.solve_lf_ham(params=params[nvir*nocc:], nelec=None, mp2=False, mp3=False, mp4=False,
                                  nph=9, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                                  scf_max_cycle=0)
                mo_coeff = mylf.mo_coeff
                mo_occ = mylf.mo_occ

                if nelec == 1:
                    full_opt = False
                else:
                    full_opt = True
                e = mylf.kernel(params=params, mp2=mp2, mp3=mp3, mp4=mp4, nph=9, ntrial=ntrial,
                                use_num_grad=True, full_opt=full_opt,
                                mo_coeff=mo_coeff, mo_occ=mo_occ)

            print ("*" * 79)
            print ("w_p", w_p, "alpha", alpha)
            print ("*" * 79)

            e_col.append(mylf.e_hf)
            e_mp2_col.append(mylf.e_tot)

        e_col = np.asarray(e_col)
        e_mp2_col = np.asarray(e_mp2_col)
        print ("*" * 79)
        print ("Results", "w_p", w_p, "L", nao)
        print ("*" * 79)
        print ("v-LF")
        for alpha, e in zip(alpha_list, e_col):
            print ("%15.8f %15.8f"%(alpha, e))
        print ("v-LF + MP2")
        for alpha, e in zip(alpha_list, e_mp2_col):
            print ("%15.8f %15.8f"%(alpha, e))
        print ("*" * 79)
