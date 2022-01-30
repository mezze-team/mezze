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

import scipy.linalg as la
import numpy as np
from .channel import _vec

def fidelity(A,B):
    AA = A.density_matrix()
    BB = B.density_matrix()

    sqA = la.sqrtm(AA)
    return np.trace(np.real(la.sqrtm(sqA@BB@sqA)))**2

def infidelity(A,B):
    return 1. - fidelity(A,B)

def process_fidelity(A,B):
    if A.rank() == 1:
        return process_fidelity_to_unitary(B,A)
    elif B.rank() == 1:
        return process_fidelity_to_unitary(A,B)
    else:
        return fidelity(A.Jamiliokowski(),B.Jamiliokowski())

def process_fidelity_to_unitary(A,U):
    N = U.kraus()[0].shape[0]

    return 1./N**2*np.real(_vec(U.kraus()[0]).conj().T@A.choi()@_vec(U.kraus()[0]))[0,0]

def process_infidelity(A,B):
    return 1. - process_fidelity(A,B)

def process_infidelity_to_unitary(A,U):
    return 1. - process_fidelity_to_unitary(A,U)

def unitarity(A):
    P = A.ptm()[1:,1:]
    d = np.sqrt(P.shape[0]+1)

    return 1./(d**2-1)*np.trace(P.conj().T@P)
