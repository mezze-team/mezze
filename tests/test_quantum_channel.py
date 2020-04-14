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

import unittest
import mezze.channel as ch
import numpy as np
import scipy.linalg as la
import itertools
import mezze.random.uniform as uni

class TestQuantumChannel(unittest.TestCase):

    def test_liou_to_choi(self):

        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        for i,gate in enumerate(paulis):
            L = np.kron(gate.conj(),gate)
            D = ch.QuantumChannel(L,'liou')
            choi = D.choi()
            choi_true = ch._vec(gate)*ch._vec(gate).H

            self.assertEqual(la.norm(choi-choi_true),0)

    def test_liou_to_ptm(self):

        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        for i,gate in enumerate(paulis):
            L = np.kron(gate.conj(),gate)
            D = ch.QuantumChannel(L,'liou')
            ptm = D.ptm()
            ptm_true = np.eye(4)
            if i>0:
                for j in range(1,4):
                    if i!=j:
                        ptm_true[j,j] = -1

            self.assertEqual(la.norm(ptm-ptm_true),0)

    def test_liou_to_chi(self):

        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        for i,gate in enumerate(paulis):
            L = np.kron(gate.conj(),gate)
            D = ch.QuantumChannel(L,'liou')
            chi = D.chi()
            chi_true = np.zeros((4,4))
            chi_true[i,i] = 1

            self.assertEqual(la.norm(chi-chi_true),0)


    def test_liou_to_kraus_stiefel(self):

        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        for i,gate in enumerate(paulis):
            L = np.kron(gate.conj(),gate)
            D = ch.QuantumChannel(L,'liou')
            kraus = D.kraus()

            #self.assertEqual(len(kraus),1)
            L_kraus = np.sum([np.kron(k.conj(),k) for k in kraus],0)
            self.assertAlmostEqual(la.norm(L-L_kraus),0,12)

            L = np.kron(gate.conj(),gate)
            D = ch.QuantumChannel(L,'liou')
            S = D.stiefel()

            errs = [la.norm(kraus[i]-S[i*2:(i+1)*2,:]) for i in range(len(kraus))]
            self.assertEqual(np.max(np.abs(errs)),0)

            errs2 = [la.norm(S[i*2:(i+1)*2:]) for i in range(len(kraus),4)]
            self.assertEqual(np.max(np.abs(errs2)),0)

    def test_kraus_to_all(self):
        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        weights = np.random.randn(4)+1j*np.random.randn(4)
        weights = weights/la.norm(weights)
        
        kraus = [w*p for w,p  in zip(weights,paulis)]
        D = ch.QuantumChannel(kraus,'kraus')
        kraus2 = D.kraus()
        errs = [la.norm(k1-k2) for k1,k2 in zip(kraus,kraus2)]
        self.assertEqual(np.max(errs),0)
        
        
        S = np.zeros((8,2),dtype=np.complex)
        for i in range(4):
            S[i*2:(i+1)*2,:] = kraus[i]

        D = ch.QuantumChannel(kraus,'kraus')
        S2 = D.stiefel()
        self.assertEqual(la.norm(S-S2),0)
        
        S = np.zeros((8,2),dtype=np.complex)
        for i in range(4):
            S[i::4,:] = kraus[i]

        D = ch.QuantumChannel(kraus,'kraus')
        S2 = D.stiefel2()
        self.assertEqual(la.norm(S-S2),0)

        D = ch.QuantumChannel(kraus,'kraus')
        L = np.sum([w.conj()*w*np.kron(p.conj(),p) for w,p in zip(weights,paulis)],0)
        L2= D.liouvillian()
        self.assertAlmostEqual(la.norm(L-L2),0,15)

        ptms = [np.eye(4) for _ in paulis]
        for i in range(1,4):
            for j in range(1,4):
                if i!=j:
                    ptms[i][j,j] = -1

        PTM = np.sum([w.conj()*w*p for w,p in zip(weights,ptms)],0)
        D=ch.QuantumChannel(kraus,'kraus')
        PTM2 = D.ptm()
        self.assertAlmostEqual(la.norm(PTM-PTM2),0)

        choi = np.sum([w.conj()*w*ch._vec(p)*ch._vec(p).H for w,p in zip(weights,paulis)], 0)
        D=ch.QuantumChannel(kraus,'kraus')
        choi2 = D.choi()
        self.assertAlmostEqual(la.norm(choi-choi2),0)

        chi = np.diag([w.conj()*w for w in weights])
        D=ch.QuantumChannel(kraus,'kraus')
        chi2 = D.chi()
        self.assertAlmostEqual(la.norm(chi-chi2),0,15)


    def test_stiefel_to_all(self):
        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        weights = np.random.randn(4)+1j*np.random.randn(4)
        weights = weights/la.norm(weights)
        
        kraus = [w*p for w,p  in zip(weights,paulis)]
        S = np.zeros((8,2),dtype=np.complex)
        for i in range(4):
            S[i*2:(i+1)*2,:] = kraus[i]

        D = ch.QuantumChannel(S,'stiefel')
        S2 = D.stiefel()
        self.assertEqual(la.norm(S-S2),0)
        
        S2test = np.zeros((8,2),dtype=np.complex)
        for i in range(4):
            S2test[i::4,:] = kraus[i]

        D = ch.QuantumChannel(S,'stiefel')
        S2 = D.stiefel2()
        self.assertEqual(la.norm(S2-S2test),0)
        
        D = ch.QuantumChannel(S,'stiefel')
        kraus2 = D.kraus()
        errs = [la.norm(k1-k2) for k1,k2 in zip(kraus,kraus2)]
        self.assertEqual(np.max(errs),0)

        D = ch.QuantumChannel(S,'stiefel')
        L = np.sum([w.conj()*w*np.kron(p.conj(),p) for w,p in zip(weights,paulis)],0)
        L2= D.liouvillian()
        self.assertAlmostEqual(la.norm(L-L2),0,15)

        ptms = [np.eye(4) for _ in paulis]
        for i in range(1,4):
            for j in range(1,4):
                if i!=j:
                    ptms[i][j,j] = -1

        PTM = np.sum([w.conj()*w*p for w,p in zip(weights,ptms)],0)
        D=ch.QuantumChannel(S,'stiefel')
        PTM2 = D.ptm()
        self.assertAlmostEqual(la.norm(PTM-PTM2),0)

        choi = np.sum([w.conj()*w*ch._vec(p)*ch._vec(p).H for w,p in zip(weights,paulis)], 0)
        D=ch.QuantumChannel(S,'stiefel')
        choi2 = D.choi()
        self.assertAlmostEqual(la.norm(choi-choi2),0)

        chi = np.diag([w.conj()*w for w in weights])
        D=ch.QuantumChannel(S,'stiefel')
        chi2 = D.chi()
        self.assertAlmostEqual(la.norm(chi-chi2),0,15)

    def test_ptm_to_all(self):
        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        weights = np.random.randn(4)+1j*np.random.randn(4)
        weights = weights/la.norm(weights)
        
        ptms = [np.eye(4) for _ in paulis]
        for i in range(1,4):
            for j in range(1,4):
                if i!=j:
                    ptms[i][j,j] = -1
        
        PTM = np.sum([w.conj()*w*p for w,p in zip(weights,ptms)],0)
        D=ch.QuantumChannel(PTM,'ptm')
        PTM2 = D.ptm()
        self.assertEqual(la.norm(PTM-PTM2),0)
        
        
        kraus = [w*p for w,p  in zip(weights,paulis)]

        D=ch.QuantumChannel(PTM,'ptm')
        kraus2 = D.kraus()
        #errs = [la.norm(k1-k2) for k1,k2 in zip(kraus,kraus2)]
        errs = [np.min([la.norm(np.kron(k1.conj(),k1)-np.kron(k2.conj(),k2)) for k2 in kraus2]) for k1 in kraus]
        self.assertAlmostEqual(la.norm(errs),0,12)

        #Skip stiefel since it is tedious due to phase ambiguity
        #S = np.zeros((8,2),dtype=np.complex)
        #for i in range(4):
        #    S[i*2:(i+1)*2,:] = kraus[i]
        D=ch.QuantumChannel(PTM,'ptm')
        S2 = D.stiefel()
        #errs = [np.min([la.norm(np.kron(k1.conj(),k1)-np.kron(k2.conj(),k2)) for k2 in kraus2]) for k1 in kraus]
        #self.AssertEqual(la.norm(S-S2),0)
        
        D=ch.QuantumChannel(PTM,'ptm')
        L = np.sum([w.conj()*w*np.kron(p.conj(),p) for w,p in zip(weights,paulis)],0)
        L2= D.liouvillian()
        self.assertAlmostEqual(la.norm(L-L2),0,12)



        choi = np.sum([w.conj()*w*ch._vec(p)*ch._vec(p).H for w,p in zip(weights,paulis)], 0)
        D=ch.QuantumChannel(PTM,'ptm')
        choi2 = D.choi()
        self.assertAlmostEqual(la.norm(choi-choi2),0,12)

        chi = np.diag([w.conj()*w for w in weights])
        D=ch.QuantumChannel(PTM,'ptm')
        chi2 = D.chi()
        self.assertAlmostEqual(la.norm(chi-chi2),0,12)

    def test_chi_to_all(self):
        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        weights = np.random.randn(4)+1j*np.random.randn(4)
        weights = weights/la.norm(weights)
        
        ptms = [np.eye(4) for _ in paulis]
        for i in range(1,4):
            for j in range(1,4):
                if i!=j:
                    ptms[i][j,j] = -1
        
        chi = np.diag([w.conj()*w for w in weights])
        D=ch.QuantumChannel(chi,'chi')
        chi2 = D.chi()
        self.assertEqual(la.norm(chi-chi2),0)
        
        choi = np.sum([w.conj()*w*ch._vec(p)*ch._vec(p).H for w,p in zip(weights,paulis)], 0)
        D=ch.QuantumChannel(chi,'chi')
        choi2 = D.choi()
        self.assertAlmostEqual(la.norm(choi-choi2),0,12)
        
        D=ch.QuantumChannel(chi,'chi')
        L = np.sum([w.conj()*w*np.kron(p.conj(),p) for w,p in zip(weights,paulis)],0)
        L2= D.liouvillian()
        self.assertAlmostEqual(la.norm(L-L2),0,12)
        
        kraus = [w*p for w,p  in zip(weights,paulis)]
        D=ch.QuantumChannel(chi,'chi')
        kraus2 = D.kraus()
        #errs = [la.norm(k1-k2) for k1,k2 in zip(kraus,kraus2)]
        errs = [np.min([la.norm(np.kron(k1.conj(),k1)-np.kron(k2.conj(),k2)) for k2 in kraus2]) for k1 in kraus]
        self.assertAlmostEqual(la.norm(errs),0,12)

        #Skip stiefel since it is tedious due to phase ambiguity
        #S = np.zeros((8,2),dtype=np.complex)
        #for i in range(4):
        #    S[i*2:(i+1)*2,:] = kraus[i]
        D=ch.QuantumChannel(chi,'chi')
        S2 = D.stiefel()
        #errs = [np.min([la.norm(np.kron(k1.conj(),k1)-np.kron(k2.conj(),k2)) for k2 in kraus2]) for k1 in kraus]
        #self.AssertEqual(la.norm(S-S2),0)
        
        PTM = np.sum([w.conj()*w*p for w,p in zip(weights,ptms)],0)
        D=ch.QuantumChannel(chi,'chi')
        PTM2 = D.ptm()
        self.assertAlmostEqual(la.norm(PTM-PTM2),0,12)
        
        
    def test_choi_to_all(self):
        paulis = [ch._sigmaI,ch._sigmaX,ch._sigmaY,ch._sigmaZ]
        weights = np.random.randn(4)+1j*np.random.randn(4)
        weights = weights/la.norm(weights)
        
        ptms = [np.eye(4) for _ in paulis]
        for i in range(1,4):
            for j in range(1,4):
                if i!=j:
                    ptms[i][j,j] = -1
        
        choi = np.sum([w.conj()*w*ch._vec(p)*ch._vec(p).H for w,p in zip(weights,paulis)], 0)
        D=ch.QuantumChannel(choi,'choi')
        choi2 = D.choi()
        self.assertEqual(la.norm(choi-choi2),0)
        
        chi = np.diag([w.conj()*w for w in weights])
        D=ch.QuantumChannel(choi,'choi')
        chi2 = D.chi()
        self.assertAlmostEqual(la.norm(chi-chi2),0,12)
        
        D=ch.QuantumChannel(choi,'choi')
        L = np.sum([w.conj()*w*np.kron(p.conj(),p) for w,p in zip(weights,paulis)],0)
        L2= D.liouvillian()
        self.assertEqual(la.norm(L-L2),0)
        
        kraus = [w*p for w,p  in zip(weights,paulis)]
        D=ch.QuantumChannel(choi,'choi')
        kraus2 = D.kraus()
        #errs = [la.norm(k1-k2) for k1,k2 in zip(kraus,kraus2)]
        errs = [np.min([la.norm(np.kron(k1.conj(),k1)-np.kron(k2.conj(),k2)) for k2 in kraus2]) for k1 in kraus]
        self.assertAlmostEqual(la.norm(errs),0,12)

        #Skip stiefel since it is tedious due to phase ambiguity
        #S = np.zeros((8,2),dtype=np.complex)
        #for i in range(4):
        #    S[i*2:(i+1)*2,:] = kraus[i]
        D=ch.QuantumChannel(choi,'choi')
        S2 = D.stiefel()
        #errs = [np.min([la.norm(np.kron(k1.conj(),k1)-np.kron(k2.conj(),k2)) for k2 in kraus2]) for k1 in kraus]
        #self.AssertEqual(la.norm(S-S2),0)
        
        PTM = np.sum([w.conj()*w*p for w,p in zip(weights,ptms)],0)
        D=ch.QuantumChannel(choi,'choi')
        PTM2 = D.ptm()
        self.assertAlmostEqual(la.norm(PTM-PTM2),0,12)

    def test_choi_real_diagonal(self):

        choi = .25*np.eye(4) + 1e-8*1j*np.eye(4)

        D = ch.QuantumChannel(choi,'choi')
        diag = np.diag(D._choi.choi)
        self.assertEqual(la.norm(np.imag(diag)),0)
        
        diag = np.diag(D.choi())
        self.assertEqual(la.norm(np.imag(diag)),0)


    def test_is_unital(self):
        
        U1 = uni.choi(2,1)
        self.assertTrue(U1.is_unital())
        U2 = uni.choi(2,1)
        self.assertTrue(U1.is_unital())

        mix = np.random.rand()
        unital_map = ch.QuantumChannel(mix*U1.choi()+(1-mix)*U2.choi(),'choi')

        self.assertTrue(unital_map.is_unital())

        gamma = np.random.rand()
        amp_damp = ch.QuantumChannel([[[1,0],[0,np.sqrt(1-gamma)]],[[0,np.sqrt(gamma)],[0,0]]],'kraus')

        self.assertFalse(amp_damp.is_unital())

    def test_is_extremal(self):


        U1 = uni.choi(2,1)
        self.assertTrue(U1.is_extremal())
        U2 = uni.choi(2,1)
        self.assertTrue(U2.is_extremal())

        mix = np.random.rand()
        unital_map = ch.QuantumChannel(mix*U1.choi()+(1-mix)*U2.choi(),'choi')

        self.assertFalse(unital_map.is_extremal())

        gamma = np.random.rand()
        amp_damp = ch.QuantumChannel([[[1,0],[0,np.sqrt(1-gamma)]],[[0,np.sqrt(gamma)],[0,0]]],'kraus')

        self.assertTrue(amp_damp.is_extremal())

        C3 = uni.choi(2,3)
        self.assertFalse(C3.is_extremal())

        C4 = uni.choi(2,4)
        self.assertFalse(C4.is_extremal())

    def test_stiefel_tangent(self):

        C = uni.choi(2,1)
        TX1 = C.stiefel_tangent()
        TX2 = C.stiefel_tangent(full=True)

        self.assertEqual(TX1.shape,(8,2))
        self.assertEqual(TX2.shape,(8,8))
        self.assertEqual(la.norm(TX1-TX2[:,:2]),0)

        C2 = ch.QuantumChannel(la.expm(TX2)[:,:2],'stiefel')

        self.assertAlmostEqual(la.norm(C.choi()-C2.choi()),0)
        self.assertEqual(C2.rank(),1)
        self.assertTrue(C2.is_valid())

        C = uni.choi(3,9)
        TX1 = C.stiefel_tangent()
        TX2 = C.stiefel_tangent(full=True)

        self.assertEqual(TX1.shape,(27,3))
        self.assertEqual(TX2.shape,(27,27))
        self.assertEqual(la.norm(TX1-TX2[:,:3]),0)

        C2 = ch.QuantumChannel(la.expm(TX2)[:,:3],'stiefel')

        self.assertAlmostEqual(la.norm(C.choi()-C2.choi()),0)
        self.assertEqual(C2.rank(),9)
        self.assertTrue(C2.is_valid())

if __name__ == '__main__':
    unittest.main()
