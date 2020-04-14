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
import mezze.random.uniform as uni

class TestQuantumState(unittest.TestCase):

    def test_simple(self):

        zero = np.matrix(np.eye(2))
        zero[1,1] = 0
        p = ch.QuantumState(zero,'dm')
        err = la.norm(p.bloch_vector()-np.matrix([[0],[0],[1]]))
        self.assertEqual(err,0)

        one = np.matrix(np.eye(2))
        one[0,0] = 0
        p = ch.QuantumState(one,'dm')
        err = la.norm(p.bloch_vector()-np.matrix([[0],[0],[-1]]))
        self.assertEqual(err,0)

    def test_bv_to_dm(self):

        p = ch.QuantumState(np.matrix(np.zeros((3,1))),'bv')
        err = la.norm(p.density_matrix()-np.eye(2)/2.)
        self.assertEqual(err,0.)

        p = ch.QuantumState(np.matrix([[1],[0],[0]]),'bv')
        err = la.norm(p.density_matrix()-.5*(ch._sigmaX+ch._sigmaI))
        self.assertEqual(err,0.)

    def test_bv_to_sm(self):

        p = ch.QuantumState(np.matrix(np.zeros((3,1))),'bv')
        b,w = p.state_mixture()
        
        self.assertEqual(la.norm(b-.5*np.ones(2)),0)
        err = la.norm(p.density_matrix()-.5*(np.eye(2)))
        self.assertEqual(err,0.)

    def test_dm_to_dm(self):

        G = np.random.randn(2,2)+1j*np.random.randn(2,2)
        Q,R = la.qr(G)
        d = np.random.rand(2)
        d = d/np.sum(d)
        Q = np.matrix(Q)
        dm = np.dot(Q,np.dot(np.diag(d),Q.H))

        self.assertAlmostEqual(np.real(np.trace(dm)),1,12)
        self.assertAlmostEqual(la.norm(dm-dm.H),0,12)

        p = ch.QuantumState(dm)

        err1 = la.norm(dm-p.density_matrix())
        self.assertEqual(err1,0)

        dv = p.density_vector()
        p2 = ch.QuantumState(dv,'dv')

        err2 = la.norm(dm-p2.density_matrix())
        self.assertEqual(err2,0)

        bv = p.bloch_vector()
        p3 = ch.QuantumState(bv,'bv')
        err3 = la.norm(dm-p3.density_matrix())
        self.assertAlmostEqual(err3,0,12)

        w,b = p.state_mixture()
        p4 = ch.QuantumState((w,b),'sm')
        err4 = la.norm(dm-p4.density_matrix())
        self.assertAlmostEqual(err4,0,12)

    def test_dv_to_dv(self):
        
        G = np.random.randn(2,2)+1j*np.random.randn(2,2)
        Q,R = la.qr(G)
        d = np.random.rand(2)
        d = d/np.sum(d)
        Q = np.matrix(Q)
        dm = np.dot(Q,np.dot(np.diag(d),Q.H))

        dv = ch._vec(dm)

        p = ch.QuantumState(dv,'dv')

        err1 = la.norm(dv-p.density_vector())
        self.assertEqual(err1,0)

        dm = p.density_matrix()
        p2 = ch.QuantumState(dm,'dm')

        err2 = la.norm(dv-p2.density_vector())
        self.assertEqual(err2,0)

        bv = p.bloch_vector()
        self.assertLessEqual(la.norm(bv),1.)

        p3 = ch.QuantumState(bv,'bv')
        err3 = la.norm(dv-p3.density_vector())
        self.assertAlmostEqual(err3,0,12)
        
        w,b = p.state_mixture()
        p4 = ch.QuantumState((w,b),'sm')
        err4 = la.norm(dv-p4.density_vector())
        self.assertAlmostEqual(err4,0,12)

    def test_bv_to_bv(self):

        bv = np.random.rand(3,1)
        if la.norm(bv)>1:
            bv = bv/la.norm(bv)

        self.assertLessEqual(la.norm(bv),1.)

        p = ch.QuantumState(bv,'bv')
        err1 = la.norm(bv-p.bloch_vector())
        self.assertEqual(err1,0)

        dv = p.density_vector()
        p2 = ch.QuantumState(dv,'dv')
        err2 = la.norm(bv-p2.bloch_vector())
        self.assertAlmostEqual(err2,0)

        dm = p.density_matrix()
        p3 = ch.QuantumState(dm)
        err3 = la.norm(bv-p3.bloch_vector())
        self.assertAlmostEqual(err3,0)
        self.assertEqual(err2,err2)

        w,b=p.state_mixture()
        p4 = ch.QuantumState((w,b),'sm')
        err4 = la.norm(bv-p4.bloch_vector())

    def test_sm_to_sm(self):

        G = np.random.randn(2,2)+1j*np.random.randn(2,2)
        Q,R = la.qr(G)
        d = np.random.rand(2)
        d = d/np.sum(d)
        Q = np.matrix(Q)
        dm = np.dot(Q,np.dot(np.diag(d),Q.H))


        w,b = la.eigh(dm)
        b = [b[:,i][:,np.newaxis] for i in range(b.shape[1])]

        self.assertAlmostEqual(np.sum(w),1,12)

        p = ch.QuantumState((w,b),'sm')
        ww,bb = p.state_mixture()

        weq = [np.min(np.abs(w1-ww)) for w1 in w]
        beq = [np.min([la.norm(b1-b2)for b2 in bb]) for b1 in b]

        self.assertEqual(la.norm(weq),0)
        self.assertEqual(la.norm(beq),0)


        dm = p.density_matrix()
        p2 = ch.QuantumState(dm,'dm')
        ww,bb = p2.state_mixture()
        
        weq = [np.min(np.abs(w1-ww)) for w1 in w]
        beq = [np.min([la.norm(b1-b2)for b2 in bb]) for b1 in b]

        self.assertAlmostEqual(la.norm(weq),0,12)
        self.assertAlmostEqual(la.norm(beq),0,12)

        dv = p.density_vector()
        p3 = ch.QuantumState(dv,'dv')
        
        ww,bb = p3.state_mixture()
        
        weq = [np.min(np.abs(w1-ww)) for w1 in w]
        beq = [np.min([la.norm(b1-b2)for b2 in bb]) for b1 in b]

        self.assertAlmostEqual(la.norm(weq),0,12)
        self.assertAlmostEqual(la.norm(beq),0,12)


        bv = p.density_vector()
        p4 = ch.QuantumState(bv,'dv')
        
        ww,bb = p4.state_mixture()
        
        weq = [np.min(np.abs(w1-ww)) for w1 in w]
        beq = [np.min([la.norm(b1-b2)for b2 in bb]) for b1 in b]

        self.assertAlmostEqual(la.norm(weq),0,12)
        self.assertAlmostEqual(la.norm(beq),0,12)


    def test_kron(self):
        
        p1 = ch.QuantumState(.5*np.eye(2),'dm')
        p2 = ch.QuantumState(.5*np.eye(2),'dm')

        p3 = p1.kron(p2)
        p4 = p1^p2

        err1 = la.norm(p3.density_matrix()-.25*np.eye(4))
        err2 = la.norm(p4.density_matrix()-.25*np.eye(4))

        self.assertEqual(err1,0)
        self.assertEqual(err2,0)

    def test_kron_n(self):

        X = uni.density_matrix(2)
        X3_a = X.kron(3)
        X3_b = X^X^X

        self.assertEqual(la.norm(X3_a.density_matrix()-X3_b.density_matrix()),0)

    def test_partial_trace(self):

        p0 = np.zeros((2,2))
        p1 = np.zeros((2,2))
        p0[0,0] = 1
        p1[1,1] = 1

        outA = (ch.QuantumState(p0)^ch.QuantumState(p1)).TrB()
        outB = (ch.QuantumState(p0)^ch.QuantumState(p1)).TrA()

        self.assertEqual(la.norm(outA.density_matrix()-p0),0)
        self.assertEqual(la.norm(outB.density_matrix()-p1),0)

    def test_is_pure_dm(self):

        rho = np.zeros((5,5))
        idx = np.random.randint(5)
        rho[idx,idx]=1

        rho = ch.QuantumState(rho,'dm')
        self.assertTrue(rho.is_pure())
        self.assertEqual(rho.rank(),1)

        dv = rho.density_vector()
        self.assertTrue(ch.QuantumState(dv,'dv').is_pure())
        self.assertEqual(ch.QuantumState(dv,'dv').rank(),1)

        sm = rho.state_mixture()
        self.assertTrue(ch.QuantumState(sm,'sm').is_pure())
        self.assertEqual(ch.QuantumState(sm,'sm').rank(),1)

    def test_is_pure_bv(self):
        bv = np.zeros((5**2-1,1))
        bv[np.random.randint(5**2-1)]=1.

        bv = ch.QuantumState(bv,'bv')
        
        #Add back in when arbitrary bloch transform introduced
        #self.assertTrue(bv.is_pure())

    def test_purity_threshold_dm(self):
        rho = np.zeros((2,2))
        rho[0,0] = 1.-1.e-13
        rho[1,1] = 1.e-13

        dm = ch.QuantumState(rho,'dm')
        self.assertTrue(dm.is_pure())
        self.assertFalse(dm.is_pure(tol=1e-14))
    
    def test_purity_threshold_bv(self):
        phi = np.matrix([[1-1e-13,0,0]]).T

        bv = ch.QuantumState(phi,'bv')
        self.assertTrue(bv.is_pure())
        self.assertFalse(bv.is_pure(tol=1e-14))

    def test_purity_threshold_sm(self):

        s = (1.-1e-13,1e-13)
        m = (np.eye(2)[:,1],np.eye(2)[:,0])

        sm = ch.QuantumState((s,m),'sm')
        self.assertTrue(sm.is_pure())
        self.assertFalse(sm.is_pure(tol=1e-14))

    def test_rank_sm(self):
        s = [.5-1e-13, .4-5e-13, 1e-13, 5e-13, 0,0]
        m = [np.eye(6)[:,i] for i in range(6)]

        sm = ch.QuantumState((s,m),'sm')
        self.assertEqual(sm.rank(),2)
        self.assertEqual(sm.rank(tol=1e-13),3)
        self.assertEqual(sm.rank(tol=1e-14),4)

        sm = ch.QuantumState((s[:4],m[:4]),'sm')
        self.assertEqual(sm.rank(),2)
        self.assertEqual(sm.rank(tol=1e-13),3)
        self.assertEqual(sm.rank(tol=1e-14),4)

    def test_rank_dm(self):
        dm = np.diag([.5-1e-13, .4-5e-13, 1e-13, 5e-13, 0,0])

        dm = ch.QuantumState(dm,'dm')

        self.assertEqual(dm.rank(),2)
        self.assertEqual(dm.rank(tol=1e-13),3)
        self.assertEqual(dm.rank(tol=1e-14),4)
    
    def test_validity_bv(self):
        bv = np.matrix([1+1e-13,0,0]).T
        SO,_ = la.qr(np.random.randn(3,3)) 
        bv = np.dot(SO,bv)

        bv = ch.QuantumState(bv,'bv')
        self.assertTrue(bv.is_valid())
        self.assertFalse(bv.is_valid(tol=1e-14))

    def test_validity_dm_sm(self):
        dm = np.diag([.5, .5+1e-13])

        dm = ch.QuantumState(dm,'dm')
        self.assertTrue(dm.is_valid())
        self.assertFalse(dm.is_valid(tol=1e-14))
        
        sm = ch.QuantumState(dm.state_mixture(),'sm')
        self.assertTrue(sm.is_valid())
        self.assertFalse(sm.is_valid(tol=1e-14))

        dm = np.diag([.5, .5-1e-13])

        dm = ch.QuantumState(dm,'dm')
        self.assertTrue(dm.is_valid())
        self.assertFalse(dm.is_valid(tol=1e-14))

        sm = ch.QuantumState(dm.state_mixture(),'sm')
        self.assertTrue(sm.is_valid())
        self.assertFalse(sm.is_valid(tol=1e-14))

        dm = np.diag([1, -1e-13])

        dm = ch.QuantumState(dm,'dm')
        self.assertTrue(dm.is_valid())
        self.assertFalse(dm.is_valid(tol=1e-14))

        sm = ch.QuantumState(dm.state_mixture(),'sm')
        self.assertTrue(sm.is_valid())
        self.assertFalse(sm.is_valid(tol=1e-14))

        dm = np.diag([1+1e-13, -1e-13])

        dm = ch.QuantumState(dm,'dm')
        self.assertTrue(dm.is_valid())
        self.assertFalse(dm.is_valid(tol=1e-14))

        sm = ch.QuantumState(sm.state_mixture(),'sm')
        self.assertTrue(sm.is_valid())
        self.assertFalse(sm.is_valid(tol=1e-14))

    def test_partial_trace(self):
        a = uni.density_matrix(2)
        b = uni.density_matrix(2)

        tot = a^b

        self.assertAlmostEqual(la.norm(a.density_matrix() -tot.TrB().density_matrix()),0)
        self.assertAlmostEqual(la.norm(b.density_matrix() -tot.TrA().density_matrix()),0)

    def test_partial_trace_asym(self):
        a = uni.density_matrix(2)
        b = uni.density_matrix(3)

        tot = a^b

        self.assertAlmostEqual(la.norm(a.density_matrix() - tot.TrB(2).density_matrix()),0)
        self.assertAlmostEqual(la.norm(b.density_matrix() - tot.TrA(3).density_matrix()),0)

        tot = b^a

        self.assertAlmostEqual(la.norm(b.density_matrix() - tot.TrB(3).density_matrix()),0)
        self.assertAlmostEqual(la.norm(a.density_matrix() - tot.TrA(2).density_matrix()),0)


if __name__ == '__main__':
    unittest.main()
