# Copyright: 2015-2017 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
# All Rights Reserved.
#
# This material may be only be used, modified, or reproduced by or for the U.S.
# Government pursuant to the license rights granted under the clauses at DFARS
# 252.227-7013/7014 or FAR 52.227-14. For any other permission, please contact
# the Office of Technology Transfer at JHU/APL: Telephone: 443-778-2792,
# Email: techtransfer@jhuapl.edu, Website: http://www.jhuapl.edu/ott/
#
# NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED "AS IS." JHU/APL MAKES
# NO REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE
# MATERIALS, INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY,
# AND DISCLAIMS ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED,
# INCLUDING (BUT NOT LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE,
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF
# INTELLECTUAL PROPERTY OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL
# ASSUMES THE ENTIRE RISK AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT
# SHALL JHU/APL BE LIABLE TO ANY USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT,
# CONSEQUENTIAL, SPECIAL OR OTHER DAMAGES ARISING FROM THE USE OF, OR INABILITY
# TO USE, THE MATERIAL, INCLUDING, BUT NOT LIMITED TO, ANY DAMAGES FOR LOST
# PROFITS.

import numpy as np
from ...channel import *
from ...channel import _TrB
import scipy.linalg as la

def unitary(N, as_channel=True, rgen = np.random):
    S = stiefel(N,1,as_channel=False,rgen=rgen)
    if as_channel:
        return QuantumChannel(S[:N,:N],'unitary')
    else:
        return S[:N,:N]

def choi(N, rank = None, as_channel=True, rgen = np.random):
    if rank is None:
        rank = N**2
    elif rank <1 or rank > N**2:
        raise RuntimeError

    X = np.array(rgen.randn(N**2,rank) + 1j*rgen.randn(N**2,rank))
    Y = _TrB(X@X.conj().T)
    sqrtY = la.sqrtm(Y)
    invsY = la.inv(sqrtY)
    LR = np.array(np.kron(invsY,np.eye(N)))
    D = LR@X@X.conj().T@LR
    idx = np.diag_indices(N**2)
    D[idx[0],idx[1]] = np.real(np.diag(D))
    
    if as_channel:
        return QuantumChannel(D,'choi')
    else:
        return np.array(D)


def stiefel(N, rank = None,as_channel=True, rgen = np.random):
    if rank is None:
        rank = N**2
    elif rank < 1 or rank >N**2:
        raise RuntimeError


    X = rgen.randn(rank*N,N) + 1j*rgen.randn(rank*N,N)
    q,_ = la.qr(X)
    q = np.array(q[:,:N])

    if as_channel:
        return QuantumChannel(q,'stiefel')
    else:
        return np.array(q)

def density_matrix(N, rank = None, as_state=True, rgen=np.random):
    U = unitary(N,as_channel=False,rgen=rgen)

    if rank is None:
        rank = N
    elif rank <1 or rank >N:
        raise NotImplementedError

    d = rgen.randn(N) + 1j*rgen.randn(N)
    d[rank:]=0
    d = d/la.norm(d)
    if as_state:
        return QuantumState(np.dot(U,np.dot(np.diag(np.conj(d)*d),U.conj().T)),'dm')
    else:
        return np.dot(U,np.dot(np.diag(np.conj(d)*d),U.conj().T))

def bloch_vector(N, pure = False, as_state=True, rgen=np.random):
    x = np.array(rgen.randn(N**2-1,1))
    x = x/la.norm(x)

    if pure == False:
        U = rgen.rand(1)**(1./(N**2-1.))
        x = U[0]*x

    if as_state:
        return QuantumState(x,'bv')
    else:
        return x

