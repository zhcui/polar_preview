#!/usr/bin/env python

"""
Hubbard Holstein model using variational Lang-Firsov method.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

mpi = False
if mpi:
    from mpi4pyscf.tools import mpi
import numpy as np
from scipy import linalg as la
from polar.lang_firsov import lang_firsov as lf

np.set_printoptions(4, linewidth=1000, suppress=True)
np.random.seed(10086)

# Hubbard Holstein model
nao = 4
nmode = nao
nelec = 1
U = 0.0

hcore = np.zeros((nao, nao))
for i in range(nao-1):
    hcore[i, i+1] = hcore[i+1, i] = -1.0
hcore[nao-1, 0] = hcore[0, nao-1] = -1.0  # PBC

eri = np.zeros((nao, nao, nao, nao))
eri[range(nao), range(nao), range(nao), range(nao)] = U

alpha_list = np.arange(0.0, 3.05, 0.2)
w_p_list = [0.5]

uniform = False
glf = 'lf'
mp2 = True
mp3 = False
mp4 = False
nph = 9

for w_p in w_p_list:
    print ("*" * 79)
    e_col = []
    e_mp2_col = []
    ntrial = 5

    w_p_arr = np.zeros((nmode,))
    w_p_arr[:] = w_p

    for alpha in alpha_list:
        g = np.sqrt(alpha * w_p)
        h_ep = np.zeros((nmode, nao, nao))
        for x in range(nmode):
            h_ep[x, x, x] = g

        if nelec == 1:
            if glf == 'lf':
                eri = None
                full_opt = False
            else:
                from polar.lang_firsov import ulf as lf
                full_opt = True
        else:
            full_opt = True
        if glf == 'lf':
            mylf = lf.LangFirsov(h0=0.0, h1=hcore, h2=eri, h_ep=g, w_p=w_p_arr,
                                 nelec=nelec, uniform=uniform)
            use_num_grad = False
        elif glf == 'glf':
            mylf = lf.UGLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=h_ep, w_p=w_p_arr,
                                  nelec=nelec, uniform=uniform)
            use_num_grad = False
        elif glf == 'gglf':
            mylf = lf.GGLangFirsov(mol=None, h0=0.0, h1=hcore, h2=eri, h_ep=h_ep, w_p=w_p_arr,
                                   nelec=nelec, uniform=uniform)
            use_num_grad = True
        else:
            raise ValueError

        if not full_opt:
            e = mylf.kernel(mp2=mp2, mp3=mp3, mp4=mp4, nph=nph, ntrial=ntrial,
                            use_num_grad=use_num_grad)
        else:
            params = (np.random.random(mylf.nparam_full) - 0.5)

            mylf.solve_lf_ham(params=params[mylf.nkappa:], nelec=None, mp2=False, mp3=False, mp4=False,
                              nph=nph, verbose=False, scf_newton=False, beta=np.inf, dm0=None,
                              scf_max_cycle=0)
            mo_coeff = mylf.mo_coeff
            mo_occ = mylf.mo_occ

            e = mylf.kernel(params=params, mp2=mp2, mp3=mp3, mp4=mp4, nph=nph, ntrial=ntrial,
                            use_num_grad=use_num_grad, full_opt=full_opt,
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
