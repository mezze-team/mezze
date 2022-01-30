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

#from conversions import liou2choi, liou2chi
import numpy as np
import scipy.linalg as la
from itertools import product
import copy
from .core import *
from functools import reduce

"""

Basic idea, the _Kraus, etc., classes contain all of the "natural" operations
and QuantumChannel contains the logic for determining which representation
to convert to to get the basic information.

"""


class Basis(object):
    """
    Base Basis class
    """

    def __init__(self, num_qudits=1):
        self.num_qudits = num_qudits
        self._tensor(num_qudits)

    def _tensor(self, N):
        self.basis_list = map(lambda x: reduce(np.kron, x), product(self.basis_list, repeat=N))
        self.basis_list = [np.array(b, dtype=complex) for b in self.basis_list]

    def transform_matrix(self, normalized=False, return_normalization=True):
        A = np.concatenate(list(map(_vec, self.basis_list)), 1)

        if not normalized and not return_normalization:
            return A
        else:
            # norms_squared = np.sum(np.power(np.abs(A),2),0)
            # norm_squared = mode(np.squeeze(np.array(norms_squared))).mode[0]
            norm_squared = np.sum(np.power(np.abs(A[:, 0]), 2))
            if normalized:
                return 1. / np.sqrt(norm_squared) * A  # np.linalg.norm(A[:,0])*A
            else:
                # This is more numerically accurate than norm**2
                return A, 1. / norm_squared  # np.sum(np.power(np.abs(A[:,0]),2))


class PauliBasis(Basis):

    def __init__(self, num_qudits=1):
        self.basis_list = [s_i, s_x, s_y, s_z]
        Basis.__init__(self, num_qudits)


class GellMannBasis(Basis):

    def __init__(self, num_qudits=1):
        self.basis_list = [np.zeros((3, 3), dtype=complex) for _ in range(9)]

        self.basis_list[0] = np.eye(3, dtype=complex)  # *np.sqrt(2)/np.sqrt(3)

        self.basis_list[1][0, 1] = 1
        self.basis_list[1][1, 0] = 1

        self.basis_list[2][0, 1] = -1j
        self.basis_list[2][1, 0] = 1j

        self.basis_list[3][0, 0] = 1
        self.basis_list[3][1, 1] = -1

        self.basis_list[4][0, 2] = 1
        self.basis_list[4][2, 0] = 1

        self.basis_list[5][0, 2] = -1j
        self.basis_list[5][2, 0] = 1j

        self.basis_list[6][1, 2] = 1
        self.basis_list[6][2, 1] = 1

        self.basis_list[7][1, 2] = -1j
        self.basis_list[7][2, 1] = 1j

        self.basis_list[8][0, 0] = 1
        self.basis_list[8][1, 1] = 1
        self.basis_list[8][2, 2] = -2

        self.basis_list[1:8] = [np.sqrt(3) / np.sqrt(2) * x for x in self.basis_list[1:8]]
        self.basis_list[8] = self.basis_list[8] / np.sqrt(2)

        Basis.__init__(self, num_qudits)


class QuantumChannel(object):
    """

    Representation of the quantum channel. Handles conversions to different
    superoperator representations.  Supports Liouvillian, Choi, process matrix
    (Chi), Kraus, and Pauli transfer matrix

    """

    def __init__(self, rep, in_type = 'liou', basis = None):

        self._liou = None
        self._choi = None
        self._chi = None
        self._kraus = None
        self._stiefel = None
        self._ptm = None
        self._basis = None
        self._stiefel2 = None

        if in_type == 'liou' or in_type == 'liouvillian':
            self.natural_type = 'liou'
            self._liou = _Liouvillian(rep)
        elif in_type == 'choi':
            self.natural_type = 'choi'
            self._choi = _ChoiMatrix(rep)
        elif in_type == 'chi':
            self.natural_type = 'chi'
            self._chi = _ChiMatrix(rep, basis = basis)
        elif in_type == 'kraus':
            self.natural_type = 'kraus'
            self._kraus = _Kraus(rep)
        elif in_type == 'unitary':
            self.natural_type = 'kraus'
            self._kraus = _Kraus([rep])
        elif in_type == 'stiefel':
            self.natural_type = 'stiefel'
            self._stiefel = _StiefelMatrix(rep)
        elif in_type == 'stiefel2':
            self.natural_type = 'stiefel2'
            self._stiefel2 = _StiefelMatrix2(rep)
        elif in_type == 'ptm':
            self.natural_type = 'ptm'
            self._ptm = _PauliTransferMatrix(rep)
        else:
            raise NotImplementedError

    def __xor__(self,X):
        """

        This seemed like a good symbol for the Kronecker product

        """
        
        return self.kron(X)

    def kron(self,X):
        """
        
        This is the kronecker product. If X is an int it will tensor with n copies of itself
        
        """

        if type(X)==int and X>1:
            
            out = copy.copy(self)
            for i in range(X-1):
                out = out.kron(self)
            return out

        return QuantumChannel([np.kron(a,b) for a,b in product(self.kraus(),X.kraus())],
                              in_type='kraus')
        


    def __mul__(self,X):
        """
        
        If X is a channel, it performs composition, if X is a state it performs
        the action of the channel on the state

        """
        if type(X) == QuantumChannel:
            return self.dot(X)
        elif type(X) == QuantumState:
            return self.map(X)

    def dot(self,X):
        """
        
        Multiplication operation, preserves natural type as PTM if *both*
        arguments are PTMs, otherwise defaults to Liouvillians

        """
        if self.natural_type == 'ptm' and X.natural_type == 'ptm':
            return QuantumChannel(self.ptm()@X.ptm(),'ptm')
        else:
            return QuantumChannel(self.liouvillian()@X.liouvillian(),'liou')

    def map(self,rho):
        """
        
        Maps input to output state.  Returns a state whose natural_type is
        determined by the natural_type of the channel.
        

        """
        #Consider using apply methods in sub-classes
        if self.natural_type == 'ptm':
            ptm = self.ptm()
            return QuantumState(np.dot(ptm[1:,1:],rho.bloch_vector())+ptm[1:,0][:,np.newaxis],'bv')
        elif self.natural_type in ['liou','chi','choi']:
            L = self.liouvillian()
            return QuantumState(np.dot(L,rho.density_vector()),'dv')
        elif self.natural_type in ['kraus','stiefel','stiefel2']:
            ops = self.kraus()
            phi = np.sum([np.dot(A,np.dot(rho.density_matrix(),A.conj().T)) for A in ops],0)
            return QuantumState(phi,'dm')
        else:
            raise NotImplementedError

    def TrA(self):
        """

        Returns the partial trace on the Choi matrix over the first Hilbert
        space

        """
        return _TrA(self.choi())

    def TrB(self):

        """

        Returns the partial trace on the Choi matrix over the second Hilbert
        space

        """


        return _TrB(self.choi())

    def Jamiliokowski(self):
        """

        Returns the QuantumState of the scaled choi matrix

        """

        choi = self.choi()
        return QuantumState(choi/np.trace(choi),'dm')

    def dual(self):

        if self.natural_type in ['kraus','stiefel','stiefel2']:
            As = self.kraus()
            return QuantumChannel([A.conj().T for A in As],'kraus')
        elif self.natural_type == 'ptm':
            return QuantumChannel(self.ptm().conj().T,'ptm')
        else:
            return QuantumChannel(self.liouvillian().conj().T,'liou')

    def liouvillian(self):

        if self._liou is None:

            if self._choi is not None:
                self._liou = self._choi.get_liouvillian()

            elif self._chi is not None:
                self._choi = self._chi.get_choi_matrix()
                return self.liouvillian()

            elif self._kraus is not None:
                self._liou = self._kraus.get_liouvillian()

            elif self._stiefel is not None:
                self._kraus = self._stiefel.get_kraus()
                return self.liouvillian()

            elif self._stiefel2 is not None:
                self._kraus = self._stiefel2.get_kraus()
                return self.liouvillian()
            
            elif self._ptm is not None:
                self._liou = self._ptm.get_liouvillian()

        return self._liou.liouvillian

    def choi(self):

        if self._choi is None:

            if self._liou is not None:
                self._choi = self._liou.get_choi_matrix()

            elif self._chi is not None:
                self._choi = self._chi.get_choi_matrix()

            elif self._kraus is not None:
                self._choi = self._kraus.get_choi_matrix()

            elif self._stiefel is not None:
                self._kraus = self._stiefel.get_kraus()
                return self.choi()

            elif self._stiefel2 is not None:
                self._kraus = self._stiefel2.get_kraus()
                return self.choi()
            
            elif self._ptm is not None:
                self._liou = self._ptm.get_liouvillian()
                return self.choi()

        return self._choi.choi

    def kraus(self):
         
        if self._kraus is None:

            if self._stiefel is not None:
                self._kraus = self._stiefel.get_kraus()

            if self._stiefel2 is not None:
                self._kraus = self._stiefel2.get_kraus()
            
            elif self._choi is not None:
                self._kraus = self._choi.get_kraus()

            elif self._liou is not None:
                self._choi = self._liou.get_choi_matrix()
                return self.kraus()

            elif self._chi is not None:
                self._choi = self._chi.get_choi_matrix()
                return self.kraus()

            elif self._ptm is not None:
                self._liou = self._ptm.get_liouvillian()
                return self.kraus()

        return self._kraus.op_list

    def chi(self):

        if self._chi is None:
            
            self.choi()
            self._chi = self._choi.get_chi_matrix()

        return self._chi.chi_matrix

    def stiefel(self):

        if self._stiefel is None:

            self.kraus()
            self._stiefel = self._kraus.get_stiefel_matrix()
        
        return self._stiefel.stiefel_matrix

    def stiefel2(self):

        if self._stiefel2 is None:

            self.kraus()
            self._stiefel2 = self._kraus.get_stiefel2_matrix()
        
        return self._stiefel2.stiefel_matrix2
    
    def ptm(self):

        if self._ptm is None:

            self.liouvillian()
            self._ptm = self._liou.get_pauli_transfer_matrix()

        return self._ptm.ptm

    def is_valid(self, tol=1e-12):

        if self.natural_type =='choi':
            return self._choi.is_CP(tol) and self._choi.is_TP(tol)
        elif self.natural_type == 'stiefel':
            return self._stiefel.is_TP(tol)
        elif self.natural_type == 'kraus':
            return self._kraus.is_TP(tol)
        else:
            self.choi()
            return self._choi.is_CP(tol) and self._choi.is_TP(tol)

    def rank(self, tol=1e-12):
        if self.natural_type == 'kraus' or self.natural_type == 'stiefel':
            kraus = self.kraus()
            return np.sum([la.norm(_vec(k))**2>tol for k in kraus])
        else:
            vals = la.eigvalsh(self.choi())
            return np.sum(vals>tol)


    def is_unital(self, tol=1e-12):
        self.kraus()
        return self._kraus.is_unital(tol)

    def is_extremal(self, tol=1e-12):

        C = self.choi()
        N = int(np.sqrt(C.shape[0]))
        R = self.rank(tol)
        if R>N:
            return False
        elif R == 1 or self.natural_type =='unitary':
            return True
        
        
        vals,vecs = la.eigh(C)
        #vecs = np.array(vecs)
        Zi = [vecs[:,-i,np.newaxis]@vecs[:,-i,np.newaxis].conj().T  for i in range(1,R+1)]
        Zij = [[vecs[:,-i,np.newaxis]@vecs[:,-j,np.newaxis].conj().T  + vecs[:,-j,np.newaxis]@vecs[:,-i,np.newaxis].conj().T  for i in range(j+1,R+1)] for j in range(1,R+1)]
        Yij = [[1j*vecs[:,-i,np.newaxis]@vecs[:,-j,np.newaxis].conj().T - 1j*vecs[:,-j,np.newaxis]@vecs[:,-i,np.newaxis].conj().T  for i in range(j+1,R+1)] for j in range(1,R+1)]

        TrBZi = [_TrB(Z) for Z in Zi]
        TrBZij = [_TrB(Z) for Z in sum(Zij,[])]
        TrBYij = [_TrB(Z) for Z in sum(Yij,[])]

        A1 = np.concatenate([_vec(x) for x in TrBZi],1)
        A2 = np.concatenate([_vec(x) for x in TrBZij],1)
        A3 = np.concatenate([_vec(x) for x in TrBYij],1)

        A = np.concatenate([A1,A2,A3],1)

        return np.linalg.matrix_rank(A,tol)==N**2

    def stiefel_tangent(self, tol=1e-9, full=False):

        X = self.stiefel()
        N = X.shape[1]
        #X[N*self.rank():,:] = 0
        U,_,_ = la.svd(X,True)
        Xperp = U[:,N:]
        XXperp = np.concatenate((X,Xperp),1)

        vals,vecs = la.eig(XXperp)
        TX = np.dot(vecs,np.dot(np.diag(np.log(vals)),vecs.conj().T))

        while la.norm(TX[N:,N:])>tol:
            Xperp = np.dot(Xperp, la.expm(-TX[N:,N:]))
            XXperp = np.concatenate((X,Xperp),1)
            #TX = la.logm(XXperp)
            vals, vecs = la.eig(XXperp)
            TX = np.dot(vecs, np.dot(np.diag(np.log(vals)), vecs.conj().T))

        if full:
            TX[N:,N:]=0
        else:
            TX = TX[:,:N]

        return np.array(TX)
        

class QuantumState(object):
    """
    Representation of the quantum state in an analogous fashion to
    QuantumChannel
    """

    def __init__(self, state, in_type = 'dm'):
                
        self._dm = None
        self._dv = None
        self._bv = None
        self._sm = None
        self._rank = None
        
        if in_type == 'dm':
            self.natural_type = 'dm'
            self._dm = _DensityMatrix(state)
        elif in_type == 'dv':
            self.natural_type = 'dv'
            self._dv = _DensityVector(state)
        elif in_type == 'bv':
            self.natural_type = 'bv'
            self._bv = _BlochVector(state)
        elif in_type == 'sm':
            self.natural_type = 'sm'
            self._sm = _StateMixture(state[0],state[1])
            self._rank = len(state)
        else:
            raise NotImplementedError

    def kron(self, rho2):
        #can this be one without dms?
        
        if type(rho2)==int and rho2>1:
            
            out = copy.copy(self)
            for i in range(rho2-1):
                out = out.kron(self)
            return out

        dm = np.kron(self.density_matrix(),rho2.density_matrix())
        return QuantumState(dm,'dm')

    def __xor__(self,X):
        """

        This seemed like a good symbol for the Kronecker product

        """
        
        return self.kron(X)

    def TrA(self,n=None):
        return QuantumState(_TrA(self.density_matrix(),n),'dm')

    def TrB(self,n=None):
        return QuantumState(_TrB(self.density_matrix(),n),'dm')

    def density_matrix(self):

        if self._dm is None:

            if self._dv is not None:
                self._dm = self._dv.get_density_matrix()

            elif self._sm is not None:
                self._dm = self._sm.get_density_matrix()

            elif self._bv is not None:
                self._dv = self._bv.get_density_vector()
                return self.density_matrix()

        return self._dm.density_matrix

    def density_vector(self):

        if self._dv is None:

            if self._dm is not None:
                self._dv = self._dm.get_density_vector()

            elif self._bv is not None:
                self._dv = self._bv.get_density_vector()

            elif self._sm is not None:
                self._dm = self._sm.get_density_matrix()
                return self.density_vector()

        return self._dv.density_vector

    def bloch_vector(self):
        
        if self._bv is None:

            if self._dv is not None:
                self._bv = self._dv.get_bloch_vector()

            elif self._dm is not None:
                self._dv = self._dm.get_density_vector()
                return self.bloch_vector()

            elif self._sm is not None:
                self._dm = self._sm.get_density_matrix()
                return self.bloch_vector()

        return self._bv.bloch_vector

    def state_mixture(self):

        if self._sm is None:

            if self._dm is not None:
                self._sm = self._dm.get_state_mixture()

            elif self._dv is not None:
                self._dm = self._dv.get_density_matrix()
                return self.state_mixture()

            elif self._bv is not None:
                self._dv = self._bv.get_density_vector()
                return self.state_mixture()

        return self._sm.weights, self._sm.basis

    def is_valid(self,tol=1e-12):

        if self.natural_type == 'bv':
            return la.norm(self.bloch_vector()) - 1 < tol
        
        else:
            if self.natural_type == 'sm':
                vals,_ = self.state_mixture()
            else:
                vals = la.eigvalsh(self.density_matrix())

            part1 = np.all(np.imag(vals)==0)
            part2 = np.all(np.real(vals)>-tol)
            part3 = la.norm(self.density_matrix().conj().T-self.density_matrix().conj().T)==0
            part4 = np.abs(np.trace(self.density_matrix())-1)<tol

            return part1 and part2 and part3 and part4

    def is_pure(self,tol=1e-12):

        if self.natural_type == 'sm':
            w,b = self.state_mixture()
            if len(w) == 1 and w[0]==1:
                return True
            elif np.sum(w) == 1 and np.max(w) >= 1.-tol and all(w>=0):
                return True
            else:
                return False
        
        elif self.natural_type == 'bv':
            return np.abs(la.norm(self.bloch_vector())-1)<tol
        else:
            vals = la.eigvalsh(self.density_matrix())
            part1 = np.abs(np.sum(vals)-1)<tol
            part2 = np.abs(np.max(vals)-1)<tol
            part3 = np.all(vals>-tol)
            return part1 and part2 and part3

    def rank(self,tol=1e-12):

        if self.natural_type == 'sm':
            w,b = self.state_mixture()
            return np.sum(w>tol)
        else:
            vals = la.eigvalsh(self.density_matrix())
            return np.sum(vals>tol)

class _DensityMatrix(object):

    def __init__(self, density_matrix):

        self.density_matrix = np.array(density_matrix,dtype=complex)

    def get_density_vector(self):
        return _DensityVector(_vec(self.density_matrix))

    def get_state_mixture(self):
        weights,basis = la.eigh(self.density_matrix)
        weights = weights
        basis = [basis[:,i][:,np.newaxis] for i in range(basis.shape[1])]
        return _StateMixture(weights,basis)

class _DensityVector(object):

    def __init__(self, density_vector):

        self.density_vector = np.array(density_vector,dtype=complex)
        if self.density_vector.shape[1] !=1:
            self.density_vector=self.density_vector.T

    def get_density_matrix(self):
        return _DensityMatrix(_inv_vec(self.density_vector))

    def get_bloch_vector(self):
        N = int(np.sqrt(self.density_vector.shape[0]))
        basis = _make_basis(N)
        T,c = basis.transform_matrix(False,True)
        bv = 1./np.sqrt(N-1)*T.conj().T @ self.density_vector

        return _BlochVector(bv[1:])

class _BlochVector(object):

    def __init__(self, bloch_vector):
        self.bloch_vector = np.array(np.real(bloch_vector))
        
        # Error/issue: tuple index out of range
        if len(self.bloch_vector.shape)==1:
           self.bloch_vector=self.bloch_vector[:,np.newaxis]

        if self.bloch_vector.shape[1]!=1:
            self.bloch_vector = self.bloch_vector.T

    def get_density_vector(self):
        N = int(np.sqrt(self.bloch_vector.shape[0]+1))
        basis = _make_basis(N)
        T,c = basis.transform_matrix(False,True)
        dv=c*np.sqrt(N-1)*T@np.concatenate((np.array([[1./np.sqrt(N-1)]]),self.bloch_vector))
        return _DensityVector(dv)
        

class _StateMixture(object):
    """

    Purposefully uses square of wavefunction weights for numerical reasons with
    the output of eigh

    """
    def __init__(self, weights, basis):
        self.weights = np.array(np.real(weights))
        if self.weights.shape == ():
            self.weights = np.array([np.real(weights)])

        self.basis = [np.array(b,dtype=complex) for b in basis]
        for i,b in enumerate(self.basis):
            ##Added a reshape feature to account for 1 dimensional arrays
            b = b.reshape(b.shape[0],-1)
            if b.shape[1]!=1:
                self.basis[i]=b.T
            else:
                self.basis[i]=b

    def get_density_matrix(self):
        return _DensityMatrix(np.sum([w*b@b.conj().T for w,b in zip(self.weights,self.basis)],axis=0))


class _Kraus(object):

    def __init__(self, op_list):

        self.op_list = [np.array(op,dtype=complex) for op in op_list]
        self.N = self.op_list[0].shape[0]

    def get_choi_matrix(self):
        return _ChoiMatrix(np.sum([np.dot(_vec(op),_vec(op).conj().T) for op in self.op_list], axis=0))

    def get_liouvillian(self):
        return _Liouvillian(np.sum([np.kron(np.conj(op),op) for op in self.op_list], axis =0))

    def get_stiefel_matrix(self):
        return _StiefelMatrix(np.concatenate(self.op_list,0))

    def get_stiefel2_matrix(self):

        s = np.concatenate([np.concatenate([o[np.newaxis,i,:] for o in self.op_list],0) for i in range(self.N)],0)
        return _StiefelMatrix2(s)

    def is_CP(self, tol=0.0):
        #by Choi's theorem, always true
        return True

    def is_TP(self, tol=0.0):

        eye = np.array(np.eye(self.op_list[0].shape[0]))
        out = np.sum([op.conj().T@eye@op for op in self.op_list],0)
        return la.norm(eye-out)<=tol

    def is_unital(self, tol=0.0):
        return la.norm(np.sum([A@A.conj().T for A in
                               self.op_list],0)-np.eye(self.N))<=tol

class _Liouvillian(object):

    def __init__(self, liouvillian):

        self.liouvillian = np.array(liouvillian,dtype=complex)
        self.N = int(np.sqrt(self.liouvillian.shape[0]))

    def get_choi_matrix(self):
        return _ChoiMatrix(_liou_choi_involution(self.liouvillian))

    def get_pauli_transfer_matrix(self):
        basis = _make_basis(self.N)
        T,c = basis.transform_matrix(False,True)
        return _PauliTransferMatrix(c*np.dot(T.conj().T ,np.dot(self.liouvillian,T)))
        
    def is_TP(self, tol = 0.0):
        rho = _vec(np.eye(self.N))
        return la.norm(self.liouvillian.conj().T@rho - rho) <= tol

    def is_unital(self, tol = 0.0):
        rho = _vec(np.eye(self.N))
        return la.norm(self.liouvillian@rho - rho) <= tol

class _PauliTransferMatrix(object):

    def __init__(self, ptm):
        self.ptm = np.array(np.real(ptm))
        self.N = int(np.sqrt(self.ptm.shape[0]))

    def get_liouvillian(self):
        basis = _make_basis(self.N)
        T,c = basis.transform_matrix(False,True)
        return _Liouvillian(c*np.dot(T,np.dot(self.ptm,T.conj().T)))

    def is_TP(self, tol = 0.0):
        phi = np.array(np.zeros((self.N**2,1)))
        phi[0,0] = 1
        return la.norm(self.ptm.conj().T@phi-phi)<=tol

    def is_unital(self, tol = 0.0):
        phi = np.array(np.zeros((self.N**2,1)))
        phi[0,0] = 1
        return la.norm(self.tm@phi-phi)<=tol

class _ChoiMatrix(object):

    def __init__(self,choi):
        self.choi = np.array(choi,dtype=complex)

        #Diagonal should be real
        idx = np.diag_indices(self.choi.shape[0])
        self.choi[idx[0],idx[1]] = np.real(np.diag(self.choi))
        self.N = int(np.sqrt(self.choi.shape[0]))

    def get_kraus(self):
        vals, vecs = la.eigh(self.choi)
        vecs = np.array(vecs)
        return _Kraus([np.sqrt(vals[i])*_inv_vec(vecs[:,i]) for i in range(len(vals))[::-1] if vals[i]>0])

    def get_liouvillian(self):
        return _Liouvillian(_liou_choi_involution(self.choi))

    def get_chi_matrix(self):
        basis = _make_basis(self.N)
        T,c = basis.transform_matrix(False,True)
        return _ChiMatrix(1./self.N*c*np.dot(T.conj().T ,np.dot(self.choi,T)))

    def is_TP(self, tol=0.0):
        return la.norm(_TrB(self.choi)-np.eye(self.N))<=tol

    def is_CP(self,tol=0.0):
        vals = la.eigvalsh(self.choi)
        return np.min(vals)>=-tol

class _ChiMatrix(object):

    def __init__(self, chi_matrix, basis = None):

        self.chi_matrix = np.array(chi_matrix,dtype=complex)

        if basis is not None:
            self._basis = basis
        else:
            self._basis = _make_basis(int(np.sqrt(self.chi_matrix.shape[0])))

    def get_choi_matrix(self):
        T,c = self._basis.transform_matrix(False,True)
        N = self._basis.basis_list[0].shape[0]
        return _ChoiMatrix(N*c*np.dot(T,np.dot(self.chi_matrix,T.conj().T )))

class _StiefelMatrix(object):
    # Corresponds to stacking Kraus operators
    def __init__(self, stiefel_matrix):

        self.stiefel_matrix = np.array(stiefel_matrix,dtype=complex)
        if self.stiefel_matrix.shape[0] != self.stiefel_matrix.shape[1]**3:
            s =np.zeros((self.stiefel_matrix.shape[1]**3,self.stiefel_matrix.shape[1]),
                       dtype=complex)
            s = np.array(s)
            s[0:stiefel_matrix.shape[0],0:stiefel_matrix.shape[1]]=stiefel_matrix

            self.stiefel_matrix = s
        
    def get_kraus(self):

        N = self.stiefel_matrix.shape[1]

        As = [self.stiefel_matrix[idx*N:(idx+1)*N,:] for idx in range(N**2)]

        As = [A for A in As if la.norm(A)>0.0]

        return _Kraus(As)

    def is_CP(self, tol=0.0):
        return True

    def is_TP(self, tol=0.0):
        err = self.stiefel_matrix.conj().T@self.stiefel_matrix
        err -= np.eye(self.stiefel_matrix.shape[1])
        return la.norm(err)<=tol

class _StiefelMatrix2(object):
    #Corresponds to stacking columns of matrix square root of choi matrix into
    #N**3xN matrix.  Also a reshuffling of Stiefel form

    def  __init__(self, stiefel_matrix):

        self.stiefel_matrix2 = np.array(stiefel_matrix,dtype=complex)

        #Issue: Tuple index out of range error issue (16,) resolved

        if len(self.stiefel_matrix2.shape)==1:
           self.stiefel_matrix2 = self.stiefel_matrix2[np.newaxis,:]

        N = self.stiefel_matrix2.shape[1]

        #Issue: Cannot broadcast input array from shape, 
        #Fixed by swapping [:,np.newaxis] to [np.newaxis,:]

        if self.stiefel_matrix2.shape[0] != N**3:
            M = self.stiefel_matrix2.shape[0]//N
            X = np.array(np.zeros((N**3,N)),dtype=complex)
            for i in range(N):
                X[i*N**2:i*N**2+M,:] = self.stiefel_matrix2[i*M:(i+1)*M,:]

            self.stiefel_matrix2 = X


    def get_kraus(self):

        N = self.stiefel_matrix2.shape[1]
        stride = self.stiefel_matrix2.shape[0]//N

        As = [self.stiefel_matrix2[idx::stride,:] for idx in range(N**2)]

        return _Kraus(As)

    def is_CP(self, tol=0.0):
        return True

    def is_TP(self, tol=0.0):
        err = self.stiefel_matrix.conj().T@self.stiefel_matrix
        err -= np.eye(self.stiefel_matrix.shape[1])
        return la.norm(err)<=tol


def _make_basis(dim):
    """
    Eventually use this to replace _T2, _T2inv and _sigma*
    """
    
    if 2**int(np.log2(dim))==dim:
        #PauliBasis
        return PauliBasis(int(np.log2(dim)))
    elif 3**int(np.log(dim)/np.log(3))==dim:
        return GellMannBasis(int(np.log(dim)/np.log(3)))
    else:
        raise NotImplementedError
    
def _vec(X):
    return np.reshape(X.T,(np.prod(X.shape),1))

def _inv_vec(X):
    dim = int(np.sqrt(X.shape[0]))
    return np.reshape(X, (dim,dim)).T

_sigmaI = np.array(np.eye(2))
_sigmaX = np.array([[0,1],[1,0]])
_sigmaY = np.array([[0,-1j],[1j,0]])
_sigmaZ = np.array([[1,0],[0,-1]])

#Being lazy and just defining qubit bloch transform matrix
_T2 = .5*np.array(np.concatenate((_vec(_sigmaI),_vec(_sigmaX),_vec(_sigmaY),_vec(_sigmaZ)),axis =1))
_T2inv = 2*_T2.conj().T

def _liou_choi_involution(op):
    """

    Exploits fact that the NxN NxN blocks of a Liouvillian are un-vectorized
    columns of the Choi matrix.  Starting with a Choi matrix, the inner-reshape
    creates a higher dimensional array of the un-vectorized columns that can be
    concatenated row to form block-rows and concatenated again to form the
    Liouvillian.  Since this is an involution, it is its own inverse.

    """

    NN = int(np.sqrt(op.shape[0]))
    return np.array(np.concatenate(np.concatenate(np.reshape(np.array(op.T), (NN,NN,NN,NN), order = 'F'),axis=1),axis=1))


def _TrA(X,n=None):
    # n x n matrix comes back
    if n is None:
        n = int(np.sqrt(X.shape[0]))

    N = X.shape[0]

    s = np.zeros((n,n),dtype=complex)
    for i in range(int(N/n)):
        s += X[i*n:(i+1)*n,i*n:(i+1)*n]
    return np.array(s)

def _TrB(X,n=None):
    # n x n matrix comes back

    #Added to convert 0 dim arrays to 2d arrays
    if X.shape == ():
        X = np.reshape(X, (-1,1))

    if n is None:
        n = int(np.sqrt(X.shape[0]))

    N = X.shape[0]
    m = int(N/n)
    s = np.zeros((n,n),dtype=complex)
    for i in range(n):
        for j in range(n):
            s[i,j] = np.trace(X[i*m:(i+1)*m,j*m:(j+1)*m])
    return np.array(s)
