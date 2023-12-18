#!/usr/bin/env python

"""
Thermal average of polaron transformed h1 and h2.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import itertools as it
from collections import Counter

import numpy as np
from scipy import linalg as la
from scipy.special import factorial as fac
from scipy.special import factorial2 as fac2
from scipy.sparse import linalg as spla
from polar.lang_firsov._expm_multiply import _expm_multiply_simple

from pyscf import lib

def get_str(nph, compact=True):
    """
    Get strings contains nph b and b+.

    Args:
        nph: number of phonon operators.

    Returns:
        ph_str: (nstr, nph), elements are False for b, True for b+.
    """
    if compact:
        ph_str = np.array(list(it.product((False, True), repeat=nph)))
        if compact:
            cre_num = np.sum(ph_str, axis=1)
            idx = (cre_num == len(ph_str)//2)
            ph_str = ph_str[idx]

            idx = (ph_str[:, -1] == True)
            ph_str = ph_str[idx]

            idx = (ph_str[:, 0] == False)
            ph_str = ph_str[idx]
    else:
        ph_str = it.product((False, True), repeat=nph)
    return ph_str

def count(string, num_bra=0):
    """
    Calculate <bra|ph_str|0> expectation value.

    Args:
        string: a string of ph ops (array of bool).
        num_bra: number of ph in bra.

    Returns:
        res: expectation value.
    """
    num = 0
    res = 1.0
    if num_bra == 0 and string[0]:
        res = 0.0
    else:
        for s in reversed(string):
            if s:
                num += 1
                res *= np.sqrt(num)
            else:
                if num == 0:
                    return 0
                else:
                    res *= np.sqrt(num)
                    num -= 1
        if num != num_bra:
            res = 0.0
    return res

def get_counts(nph, ph_str=None):
    """
    Counts of all strings in ph_str.
    """
    if ph_str is None:
        ph_str = get_str(nph, compact=False)
    res = 0.0
    for string in ph_str:
        res += count(string)
    return res

# ****************************************************************************
# H1 transform.
# ****************************************************************************

def comm(A, B):
    """
    Commuator of matrix A, B.
    """
    res  = np.dot(A, B)
    res -= np.dot(B, A)
    return res

def bch_h1_exact(h1, lams, order, H1_ref=None):
    """
    Transform of h1 using BCH expansion.
    Using exact formula, not assume lams are commutable with each other.
    This converges for small lams.

    Args:
        h1: (nao, nao) hcore matrix.
        lams: (nmode, nao, nao) parameters.
        order: int, order of BCH expansion.
        H1_ref: (nao, nao) reference transformed h1, for debug.

    Returns:
        H1: (nao, nao), transformed h1.
    """
    H1 = np.array(h1, copy=True)
    nmode = len(lams)
    for od in range(order+1)[2::2]:
        H1_od = 0.0
        fac_d = float(fac(od))

        for string in it.product(range(nmode), repeat=od):
            counts = Counter(string)
            vals = np.array(list(counts.values()))
            if np.any(vals % 2):
                continue

            tmp = h1
            for mode_id in string:
                tmp = comm(tmp, lams[mode_id])

            factor = np.prod(fac2(vals - 1)) / fac_d
            if od % 4 == 2:
                factor *= -1

            H1_od += tmp * factor

        H1 += H1_od

        if H1_ref is not None:
            print ("-" * 79)
            print ("order %5d"%od)
            print ("H1 correction\n%s"%H1_od)
            print ("H1 accumlated\n%s"%H1)
            print ("error %15.5g"%(la.norm(H1 - H1_ref)))
    return H1

def bch_h1(h1, lams, order, H1_ref=None):
    """
    Transform of H1 using BCH expansion.
    This is approximate as we assume lams are commutable with each other.
    This converges for small lams.

    Args:
        h1: (nao, nao) hcore matrix.
        lams: (nmode, nao, nao) parameters.
        order: int, order of BCH expansion.
        H1_ref: (nao, nao) reference transformed h1, for debug.

    Returns:
        H1: (nao, nao), transformed h1.
    """
    H1 = np.array(h1, copy=True)
    nmode = len(lams)
    for od in range(order+1)[2::2]:
        H1_od = 0.0

        factor = (-0.5)**(od//2) / fac(od//2)
        strs = it.product(range(nmode), repeat=(od//2))
        for string in strs:
            tmp = h1
            for mode_id in string:
                tmp = comm(tmp, lams[mode_id])
                tmp = comm(tmp, lams[mode_id])
            H1_od += tmp
        H1_od *= factor

        H1 += H1_od

        if H1_ref is not None:
            print ("-" * 79)
            print ("order %5d"%od)
            print ("H1 correction\n%s"%H1_od)
            print ("H1 accumlated\n%s"%H1)
            print ("error %15.5g"%(la.norm(H1 - H1_ref)))
    return H1

def trace_A1(lams):
    """
    Trace of A1 operator.
    """
    nao = lams.shape[-1]
    tr  = np.einsum('xmm, xnn ->', lams, lams)
    tr -= np.einsum('xmn, xnm ->', lams, lams) * nao
    return tr

def bch_h1_exp_ref(h1, lams):
    """
    Reference implementation of exponential formula of transform of h1.
    scales O(nao^4)

    Args:
        h1: (nao, nao) hcore matrix.
        lams: (nmode, nao, nao) parameters.

    Returns:
        H1: (nao, nao), transformed h1.
    """
    H1 = np.array(h1, copy=True)
    nmode, nao, _ = lams.shape

    op = 0.0
    I = np.eye(nao) * 0.5
    for x in range(nmode):
        lam_2 = np.dot(lams[x], lams[x])
        tmp = np.einsum('mi, jn -> mnij', I, lam_2)
        op -= tmp
        op -= tmp.transpose(3, 2, 1, 0)
        op += np.einsum('mi, jn -> mnij', lams[x], lams[x])
    op = op.reshape(nao*nao, nao*nao)

    assert la.norm(op - op.T) < 1e-12
    tr = np.trace(op)
    assert abs(trace_A1(lams) - tr) / abs(tr) < 1e-12

    op = la.expm(op)
    H1 = np.dot(op, h1.ravel()).reshape(nao, nao)
    return H1

def bch_h1_exp(h1, lams):
    """
    Exponential formula of transform of h1.
    scales O(nao^3).

    Args:
        h1: (nao, nao) hcore matrix.
        lams: (nmode, nao, nao) parameters.

    Returns:
        H1: (nao, nao), transformed h1.
    """
    nmode, nao, _ = lams.shape
    def A_func(h):
        h = h.reshape(nao, nao)
        res = 0.0
        for x in range(nmode):
            tmp = comm(h, lams[x])
            tmp = comm(tmp, lams[x])
            res += tmp
        res *= (-0.5)
        res = res.ravel()
        return res

    tr = trace_A1(lams)
    A_op = spla.LinearOperator((nao*nao, nao*nao), matvec=A_func, rmatvec=A_func)

    H1 = _expm_multiply_simple(A_op, h1.ravel(), traceA=tr)
    H1 = H1.reshape(nao, nao)
    return H1

# ****************************************************************************
# H2 transform.
# ****************************************************************************

def comm_h2(h2, B):
    """
    Commutator of h2 and B.
    """
    #res  = lib.einsum('pqrs, qj -> pjrs', h2, B)
    #res += lib.einsum('pqrs, sl -> pqrl', h2, B)
    #res -= lib.einsum('pqrs, pi -> iqrs', h2, B)
    #res -= lib.einsum('pqrs, rk -> pqks', h2, B)
    # use symmetry (pq|rs) = (rs|pq)
    tmp  = lib.einsum('pqrs, sl -> pqrl', h2, B)
    res  = tmp + tmp.transpose(2, 3, 0, 1)
    tmp = None
    tmp  = lib.einsum('ip, pqrs -> iqrs', B.conj().T, h2)
    res -= tmp
    res -= tmp.transpose(2, 3, 0, 1)
    return res

def bch_h2_exact(h2, lams, order, H2_ref=None):
    """
    Transform of h2 using BCH expansion.
    Using exact formula, not assume lams are commutable with each other.
    This converges for small lams.

    Args:
        h2: (nao, nao, nao, nao) eri.
        lams: (nmode, nao, nao) parameters.
        order: int, order of BCH expansion.
        H2_ref: (nao, nao, nao, nao) reference transformed h2, for debug.

    Returns:
        H2: (nao, nao, nao, nao), transformed h2.
    """
    H2 = np.array(h2, copy=True)
    nmode = len(lams)
    for od in range(order+1)[2::2]:
        H2_od = 0.0
        fac_d = float(fac(od))

        for string in it.product(range(nmode), repeat=od):
            counts = Counter(string)
            vals = np.array(list(counts.values()))
            if np.any(vals % 2):
                continue

            tmp = h2
            for mode_id in string:
                tmp = comm_h2(tmp, lams[mode_id])

            factor = np.prod(fac2(vals - 1)) / fac_d
            if od % 4 == 2:
                factor *= -1

            H2_od += tmp * factor

        H2 += H2_od

        if H2_ref is not None:
            print ("-" * 79)
            print ("order %5d"%od)
            print ("H2 correction\n%s"%H2_od[0, 0])
            print ("H2 accumlated\n%s"%H2[0, 0])
            print ("error %15.5g"%(la.norm(H2 - H2_ref)))
    return H2

def bch_h2(h2, lams, order, H2_ref=None):
    """
    Transform of h2 using BCH expansion.
    This is approximate as we assume lams are commutable with each other.
    This converges for small lams.

    Args:
        h2: (nao, nao, nao, nao) eri.
        lams: (nmode, nao, nao) parameters.
        order: int, order of BCH expansion.
        H2_ref: (nao, nao, nao, nao) reference transformed h2, for debug.

    Returns:
        H2: (nao, nao, nao, nao), transformed h2.
    """
    H2 = np.array(h2, copy=True)
    nmode = len(lams)
    for od in range(order+1)[2::2]:
        H2_od = 0.0

        factor = (-0.5)**(od//2) / fac(od//2)
        strs = it.product(range(nmode), repeat=(od//2))
        for string in strs:
            tmp = h2
            for mode_id in string:
                tmp = comm_h2(tmp, lams[mode_id])
                tmp = comm_h2(tmp, lams[mode_id])
            H2_od += tmp
        H2_od *= factor

        H2 += H2_od

        if H2_ref is not None:
            print ("-" * 79)
            print ("order %5d"%od)
            print ("H2 correction\n%s"%H2_od[0, 0])
            print ("H2 accumlated\n%s"%H2[0, 0])
            print ("error %15.5g"%(la.norm(H2 - H2_ref)))
    return H2

def trace_A2(lams):
    """
    Trace of A2 operator.
    """
    nao = lams.shape[-1]
    tr  = np.einsum('xjm, xmj ->', lams, lams) * nao**3 * (-2.0)
    tr += np.einsum('xjj, xss ->', lams, lams) * nao**2 * 2.0
    return tr

def bch_h2_exp_ref(h2, lams):
    """
    Reference implementation of exponential formula of transform of h2.
    scales O(nao^8)

    Args:
        h2: (nao, nao, nao, nao) eri.
        lams: (nmode, nao, nao) parameters.

    Returns:
        H2: (nao, nao, nao, nao), transformed h2.
    """
    H2 = np.array(h2, copy=True)
    nmode, nao, _ = lams.shape

    op = 0.0
    I = np.eye(nao)
    I_half = I * 0.5
    for x in range(nmode):
        lam = lams[x]
        lam_sq = np.dot(lam, lam)

        op -= np.einsum('pi, qj, rk, sl -> pqrsijkl', I_half, lam_sq, I, I)
        op -= np.einsum('pi, qj, rk, sl -> pqrsijkl', lam_sq, I_half, I, I)
        op -= np.einsum('pi, qj, rk, sl -> pqrsijkl', I, I, lam_sq, I_half)
        op -= np.einsum('pi, qj, rk, sl -> pqrsijkl', I, I, I_half, lam_sq)

        op -= np.einsum('pi, qj, rk, sl -> pqrsijkl', I, lam, I, lam)
        op -= np.einsum('pi, qj, rk, sl -> pqrsijkl', lam, I, lam, I)

        op += np.einsum('pi, qj, rk, sl -> pqrsijkl', lam, lam, I, I)
        op += np.einsum('pi, qj, rk, sl -> pqrsijkl', I, I, lam, lam)
        op += np.einsum('pi, qj, rk, sl -> pqrsijkl', lam, I, I, lam)
        op += np.einsum('pi, qj, rk, sl -> pqrsijkl', I, lam, lam, I)
    op = op.reshape(nao**4, nao**4)

    assert la.norm(op - op.T) < 1e-12
    tr = np.trace(op)
    print (abs(trace_A2(lams) - tr))
    assert abs(trace_A2(lams) - tr) / abs(tr) < 1e-12

    op = la.expm(op.reshape(nao**4, nao**4))
    H2 = np.dot(op, h2.ravel()).reshape(nao, nao, nao, nao)
    return H2

def bch_h2_exp(h2, lams):
    """
    Exponential formula of transform of h2.
    scales O(nao^5)

    Args:
        h2: (nao, nao, nao, nao) eri.
        lams: (nmode, nao, nao) parameters.

    Returns:
        H2: (nao, nao, nao, nao), transformed h2.
    """
    nmode, nao, _ = lams.shape
    lams_half = lams * (-0.5)

    def A_func(h):
        h = h.reshape(nao, nao, nao, nao)
        res = None
        for x in range(nmode):
            tmp = comm_h2(h, lams[x])
            tmp = comm_h2(tmp, lams_half[x])
            if res is None:
                res = tmp
            else:
                res += tmp
        res = res.ravel()
        return res

    tr = trace_A2(lams)
    A_op = spla.LinearOperator((nao**4, nao**4), matvec=A_func, rmatvec=A_func)

    H2 = _expm_multiply_simple(A_op, h2.ravel(), traceA=tr)
    H2 = H2.reshape(nao, nao, nao, nao)
    return H2

if __name__ == "__main__":

    np.random.seed(10086)
    np.set_printoptions(4, linewidth=1000, suppress=True)

    nmode = 3
    nao = 10

    lams_diag = np.random.random((nmode, nao))
    lams = np.zeros((nmode, nao, nao))
    for x in range(nmode):
        lams[x][range(nao), range(nao)] = lams_diag[x]
    print ("lams diag\n%s"%lams_diag)

    h1 = np.random.random((nao, nao)) - 0.5
    h1 = h1 + h1.T
    h1[range(nao), range(nao)] *= 5
    print ("bare h1\n%s"%h1)

    diff = lib.direct_sum('xq - xp -> xpq', lams_diag, lams_diag)
    diff **= 2
    factor = diff.sum(axis=0)
    diff = None
    factor *= (-0.5)
    factor = np.exp(factor)
    H1 = h1 * factor
    print ("polaron h1\n%s"%H1)
    print ("error: ", la.norm(H1 - h1))
    print ("-" * 79)

    H1_exact = bch_h1_exact(h1, lams, order=6, H1_ref=H1)
    H1_2 = bch_h1(h1, lams, order=6, H1_ref=H1)
    print ("Compare exact and approx formuala for diagonal lams:", la.norm(H1_exact - H1_2))
    assert la.norm(H1_exact - H1_2) < 1e-10

    H1_exp_ref = bch_h1_exp_ref(h1, lams)
    print ("H1 exp ref\n%s"%H1_exp_ref)
    diff = la.norm(H1_exp_ref - H1)
    print ("diff exp ref and expansion", diff)
    assert diff < 1e-2

    H1_exp = bch_h1_exp(h1, lams)
    diff = la.norm(H1_exp - H1_exp_ref)
    print ("diff exp fast and exp ref", diff)
    assert diff < 1e-10

    # ****************************************************************************
    # non-local parameters
    # ****************************************************************************

    print ("*" * 79)
    print ("non-local paramters")
    lams = np.random.random((nmode, nao, nao)) - 0.5
    lams = lams + lams.transpose(0, 2, 1)
    lams *= 0.1
    lams[:, range(nao), range(nao)] *= 5

    H1_exact = bch_h1_exact(h1, lams, order=10)
    H1_2 = bch_h1(h1, lams, order=10)
    H1_exp_ref = bch_h1_exp_ref(h1, lams)
    H1_exp = bch_h1_exp(h1, lams)

    print ("h1\n%s"%h1)
    print ("H1 exact\n%s"%H1_exact)
    print ("H1 approx\n%s"%H1_2)
    print ("error approx - exact", la.norm(H1_exact - H1_2))

    print ("-" * 79)
    print ("H1 exp vs expansion")
    print (la.norm(H1_exp - H1_2))
    print ("H1 exp diff")
    print (la.norm(H1_exp - H1_exp_ref))
    assert la.norm(H1_exp - H1_exp_ref) < 1e-10

    # ****************************************************************************
    # two body average
    # ****************************************************************************

    print ("*" * 79)
    print ("ERI averaging")
    print ()

    nmode = 2
    nao = 7

    from pyscf import gto, ao2mo
    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 1.1',  # in Angstrom
        basis = '321g',
        verbose = 4,
        dump_input=False)
    mf = mol.HF()
    mf.kernel()
    nao = mol.nao_nr()
    h2 = ao2mo.restore(1, mf._eri, nao)


    lams_diag = np.random.random((nmode, nao)) #* 10
    lams = np.zeros((nmode, nao, nao))
    for x in range(nmode):
        lams[x][range(nao), range(nao)] = lams_diag[x]

    factor = 0.0
    for x in range(nmode):
        diff = lib.direct_sum('q - p + s - r -> pqrs', lams_diag[x], lams_diag[x],
                              lams_diag[x], lams_diag[x])
        diff **= 2
        factor += diff
        diff = None
    factor *= (-0.5)
    factor = np.exp(factor)
    H2 = h2 * factor

    print ("lams\n%s"%lams)
    print ("h2\n%s"%h2[0, 0])
    print ("polaron h2\n%s"%H2[0, 0])
    print ("diff: ", la.norm(H2 - h2))
    print ("-" * 79)

    H2_exact = bch_h2_exact(h2, lams, order=10, H2_ref=H2)
    H2_2 = bch_h2(h2, lams, order=10, H2_ref=H2)

    H2_exp_ref = bch_h2_exp_ref(h2, lams)
    H2_exp = bch_h2_exp(h2, lams)

    print ("H2 exact vs H2")
    print (la.norm(H2_2 - H2_exact))
    assert la.norm(H2_2 - H2_exact) < 1e-10

    print ("H2 approx vs H2")
    print (la.norm(H2_2 - H2))

    print ("H2 exp fast vs H2")
    print (la.norm(H2_exp - H2))
    assert la.norm(H2_exp - H2) < 1e-10

    print ("H2 exp ref vs exp fast")
    print (la.norm(H2_exp_ref - H2_exp))
    assert la.norm(H2_exp_ref - H2_exp) < 1e-10
