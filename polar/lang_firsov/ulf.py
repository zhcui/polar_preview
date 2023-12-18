#!/usr/bin/env python

"""
Unrestricted version variational Lang-Firsov.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from functools import partial
import numpy as np
from scipy import linalg as la
from scipy import optimize as opt

from pyscf import gto, scf, ao2mo, lib
from pyscf.scf import hf, uhf
from pyscf.lib import logger
from polar.lang_firsov import grad_ulf as grad
from polar.lang_firsov import thermal_average as ta
from polar.fci.fci import fc_factor

from polar.lang_firsov.lang_firsov import GLangFirsov, GGLangFirsov

einsum = partial(np.einsum, optimize=True)

# ****************************************************************************
# Variational Lang-Firsov
# ****************************************************************************

def solve_lf_ham(mylf, params=None, nelec=None, spin=None, mp2=False, mp3=False, mp4=False,
                 nph=9, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                 scf_max_cycle=50, fci=False):
    H0, H1, H2, H_ep, w_p = mylf.get_lf_ham(params=params)
    ovlp = mylf.get_ovlp()
    nao = mylf.nao
    h1 = mylf.get_h1()
    if nelec is None:
        nelec = mylf.nelec
    if spin is None:
        spin = mylf.spin
    if params is None:
        params = mylf.params
    lams, zs = mylf.unpack_params(params)

    if H2 is not None:
        mf = uhf.UHF(mylf.mol)
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
            raise NotImplementedError
    else:
        raise NotImplementedError

    mylf.e_tot = e_tot
    return e_tot, rdm1

def solve_lf_ham_full(mylf, params=None, nelec=None, mp2=False, mp3=False, mp4=False,
                      nph=9, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                      scf_max_cycle=50, mo_coeff=None, mo_occ=None, canonicalization=True):
    if params is None:
        params = mylf.params
    (kappa_a, kappa_b), lams, zs = mylf.unpack_params_full(params)
    params_p = mylf.pack_params(lams, zs)

    H0, H1, H2, H_ep, w_p = mylf.get_lf_ham(params=params_p)

    ovlp = mylf.get_ovlp()
    nao = mylf.nao
    h1 = mylf.get_h1()
    if nelec is None:
        nelec = mylf.nelec

    if H2 is not None:
        mf = uhf.UHF(mylf.mol)
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

        nmo = len(mo_occ[0])
        nocc_a = mylf.nelec_a
        nocc_b = mylf.nelec_b

        nvir_a = nmo - nocc_a
        nvir_b = nmo - nocc_b

        dr_a = hf.unpack_uniq_var(kappa_a, mo_occ[0])
        mo_coeff_a = np.dot(mo_coeff[0], la.expm(dr_a))
        dr_b = hf.unpack_uniq_var(kappa_b, mo_occ[1])
        mo_coeff_b = np.dot(mo_coeff[1], la.expm(dr_b))
        mo_coeff = np.asarray([mo_coeff_a, mo_coeff_b])

        rdm1 = mf.make_rdm1(mo_coeff, mo_occ)
        e_tot = mf.energy_elec(dm=rdm1)[0] + mf.energy_nuc()
        fock = mf.get_fock(dm=rdm1)

        if canonicalization:
            print("-" * 79)
            mo_energy, mo_coeff = mf.canonicalize(mo_coeff, mo_occ, fock)

            homo_a = lumo_a = homo_b = lumo_b = None
            mo_e_occ_a = mo_energy[0][mo_occ[0] >= 0.5]
            mo_e_vir_a = mo_energy[0][mo_occ[0] <  0.5]
            if len(mo_e_occ_a) > 0:
                homo_a = mo_e_occ_a.max()
            if len(mo_e_vir_a) > 0:
                lumo_a = mo_e_vir_a.min()
            if homo_a is not None:
                print ('HOMO (a) = %15.8g'%(homo_a))
            if lumo_a is not None:
                print ('LUMO (a) = %15.8g'%(lumo_a))
                if homo_a is not None:
                    print ("gap  (a) = %15.8g"%(lumo_a - homo_a))
            if (lumo_a is not None) and (homo_a is not None) and (homo_a > lumo_a):
                print ('WARN: HOMO (a) %s > LUMO (a) %s was found in the canonicalized orbitals.'
                       %(homo_a, lumo_a))
                print ("mo_energy (a):\n%s"%mo_energy[0])

            print("-" * 79)
            mo_e_occ_b = mo_energy[1][mo_occ[1] >= 0.5]
            mo_e_vir_b = mo_energy[1][mo_occ[1] <  0.5]
            if len(mo_e_occ_b) > 0:
                homo_b = mo_e_occ_b.max()
            if len(mo_e_vir_b) > 0:
                lumo_b = mo_e_vir_b.min()
            if homo_b is not None:
                print ('HOMO (b) = %15.8g'%(homo_b))
            if lumo_b is not None:
                print ('LUMO (b) = %15.8g'%(lumo_b))
                if homo_b is not None:
                    print ("gap  (b)  = %15.8g"%(lumo_b - homo_b))
            if (lumo_b is not None) and (homo_b is not None) and (homo_b > lumo_b):
                print ('WARN: HOMO (b) %s > LUMO (b) %s was found in the canonicalized orbitals.'
                       %(homo_b, lumo_b))
                print ("mo_energy (b):\n%s"%mo_energy[1])

            grad = mf.get_grad(mo_coeff, mo_occ, fock)
            grad_norm = la.norm(grad)
            print("-" * 79)
            print ("|g| = %15.8g" % grad_norm)
            print("-" * 79)
        else:
            mo_energy = einsum("spm, spq, sqm -> sm", mo_coeff.conj(), fock, mo_coeff)

        mylf._scf = mf
        mylf.e_hf = float(e_tot)
        conv = mf.converged
        mylf.mo_coeff = mf.mo_coeff = mo_coeff
        mylf.mo_occ = mf.mo_occ = mo_occ
        mylf.mo_energy = mf.mo_energy = mo_energy

    if mp4 or mp3 or mp2:
        from polar.lang_firsov import mp_glf
        logger.info(mylf, "LF-MP2 start, nph = %d", nph)
        ovlp_g = la.block_diag(ovlp, ovlp)
        # ZHC FIXME should we use h1 or H1?
        hcore_g = la.block_diag(H1, H1)
        #hcore_g = la.block_diag(h1, h1)
        mf = mylf._scf = mf.to_ghf()
        mf.get_ovlp = lambda *args: ovlp_g
        mf.get_hcore = lambda *args: hcore_g
        mf._eri = H2

        if mp4:
            e_mp1, e_mp2, e_mp3, e_mp4 = mp_glf.get_e_mp4(mylf, lams=lams, zs=zs, nph=nph)
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
        elif mp2:
            e_mp2 = mp_glf.get_e_mp2(mylf, lams=lams, zs=zs, nph=nph)
            e_tot += e_mp2
            mylf.e_mp2 = e_mp2
            logger.info(mylf, "e_mp2 %15.8f", mylf.e_mp2)

    return e_tot, rdm1

class UGLangFirsov(GLangFirsov):

    @property
    def nkappa(self):
        nocc_a = self.nelec_a
        nvir_a = self.nao - nocc_a
        nk_a = nvir_a * nocc_a
        nocc_b = self.nelec_b
        nvir_b = self.nao - nocc_b
        nk_b = nvir_b * nocc_b
        nparam = nk_a + nk_b
        return nparam

    def unpack_params_full(self, params, uniform=None):
        nocc_a = self.nelec_a
        nvir_a = self.nao - nocc_a
        nk_a = nvir_a * nocc_a
        nocc_b = self.nelec_b
        nvir_b = self.nao - nocc_b
        nk_b = nvir_b * nocc_b

        kappa_a = params[:nk_a]
        kappa_b = params[nk_a:(nk_a+nk_b)]
        lams, zs = self.unpack_params(params[(nk_a+nk_b):])
        return (kappa_a, kappa_b), lams, zs

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        dm_a = np.dot(mo_coeff[0] * mo_occ[0], mo_coeff[0].conj().T)
        dm_b = np.dot(mo_coeff[1] * mo_occ[1], mo_coeff[1].conj().T)
        dm = np.asarray((dm_a, dm_b))
        return dm

    def make_rdm1p(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        if lams is None or zs is None:
            lams, zs = self.get_lams_zs(opt=True)
        rdm1 = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        nao = self.nao
        rdm1_diag = rdm1[:, range(nao), range(nao)]
        rdm1_diag_sum = np.sum(rdm1_diag, axis=0)
        rho = np.einsum("y, x -> xy", zs, zs)

        tmp = np.einsum("xp, p -> x", lams, rdm1_diag_sum)
        tmp = np.einsum("y, x -> xy", zs, tmp)
        rho -= tmp
        rho -= tmp.conj().T

        rho += np.einsum("yp, xp, p -> xy", lams, lams, rdm1_diag_sum, optimize=True)
        tmp = np.einsum("p, q -> pq", rdm1_diag_sum, rdm1_diag_sum)
        tmp -= np.einsum("sqp, spq -> pq", rdm1, rdm1)
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
        rdm1_diag = rdm1[:, range(nao), range(nao)].sum(axis=0)
        rho = zs - np.einsum("xp, p -> x", lams, rdm1_diag)
        return rho

    get_grad = grad.get_grad_glf

    get_grad_full = grad.get_grad_lf_full

    solve_lf_ham = solve_lf_ham

    solve_lf_ham_full = solve_lf_ham_full

class UGGLangFirsov(GGLangFirsov, UGLangFirsov):
    """
    Most generalized LF.
    lams has shape (nmode, nao, nao).
    """
    def make_rdm1p(self, mo_coeff=None, mo_occ=None, lams=None, zs=None):
        """
        Phonon part of rdm1.
        rho_xy = <LF | b^{\dag}_y b_x |LF>
        """
        raise NotImplementedError

    get_grad = grad.get_grad_gglf

GLF = GLangFirsov = UGLangFirsov
GGLF = GGLangFirsov = UGGLangFirsov


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
