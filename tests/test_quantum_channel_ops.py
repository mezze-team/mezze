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

class TestQuantumChannelOps(unittest.TestCase):

    def test_map_X0(self):

        zero = np.zeros((2,2))
        zero[1,1] = 0
        p = ch.QuantumState(zero,'dm')

        X = ch.QuantumChannel(ch._sigmaX,'unitary')

        out = X.map(p)
        out2 = X*p

        one = np.zeros((2,2))
        one[0,0] = 0

        self.assertEqual(la.norm(out.density_matrix()-one),0)
        self.assertEqual(la.norm(out2.density_matrix()-one),0)

    def test_map_concatenation(self):

        zero = np.zeros((2,2))
        zero[1,1] = 0
        p = ch.QuantumState(zero,'dm')

        X = ch.QuantumChannel(ch._sigmaX,'unitary')

        out = X*X*p
        out2 = (X*X)*p
        out3 = X*(X*p)

        self.assertEqual(la.norm(out.density_matrix()-zero),0)
        self.assertEqual(la.norm(out2.density_matrix()-zero),0)
        self.assertEqual(la.norm(out3.density_matrix()-zero),0)

    def test_kron_I(self):

        a = ch.QuantumChannel(np.eye(2),'unitary')
        b = ch.QuantumChannel(np.eye(2),'unitary')
        #a = ch.QuantumChannel(np.eye(4),'liou')
        #b = ch.QuantumChannel(np.eye(4),'liou')

        c1 = a.kron(b)
        c2 = a^b

        L1 = c1.liouvillian()
        L2 = c2.liouvillian()

        self.assertEqual(la.norm(L1-np.eye(16)),0)
        self.assertEqual(la.norm(L2-np.eye(16)),0)

    def test_kron_n(self):
        
        X = uni.choi(2)
        X3_a = X.kron(3)
        X3_b = X^X^X

        self.assertEqual(la.norm(X3_a.choi()-X3_b.choi()),0)

    def test_composite_map(self):

        X = ch.QuantumChannel(ch._sigmaX,'unitary')
        I = ch.QuantumChannel(ch._sigmaX,'unitary')

        tot = (X^X)*(I^X)

        zero = np.zeros((2,2))
        zero[1,1] = 0
        pz = ch.QuantumState(zero,'dm')
        
        one = np.zeros((2,2))
        one[1,1] = 0
        po = ch.QuantumState(one,'dm')

        out = np.kron(one,one)
        out1 = tot*(pz^po)
        out2 = (X*I*pz)^(X*X*po)
        out3 = po^po

        err1 = la.norm(out1.density_matrix()-out)
        err2 = la.norm(out2.density_matrix()-out)
        err3 = la.norm(out3.density_matrix()-out)

        self.assertEqual(err1,0)
        self.assertEqual(err2,0)
        self.assertEqual(err3,0)

    def test_CNOT(self):

        cnot_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

        CNOT = ch.QuantumChannel(cnot_mat,'unitary')

        zero = np.zeros((2,2))
        zero[1,1] = 0
        pz = ch.QuantumState(zero,'dm')
        
        one = np.zeros((2,2))
        one[1,1] = 0
        po = ch.QuantumState(one,'dm')

        out00 = CNOT*(pz^pz)
        out01 = CNOT*(pz^po)
        out10 = CNOT*(po^pz)
        out11 = CNOT*(po^po)

        err1 = out00.density_matrix() - (pz^pz).density_matrix()
        err2 = out01.density_matrix() - (pz^po).density_matrix()
        err3 = out10.density_matrix() - (po^po).density_matrix()
        err4 = out11.density_matrix() - (po^pz).density_matrix()

        self.assertEqual(la.norm(err1),0)
        self.assertEqual(la.norm(err2),0)
        self.assertEqual(la.norm(err3),0)
        self.assertEqual(la.norm(err4),0)

    def test_partial_trace(self):

        LX = np.kron(ch._sigmaX.conj(),ch._sigmaX)
        LY = np.kron(ch._sigmaY.conj(),ch._sigmaY)

        tot = ch.QuantumChannel(LX)^ch.QuantumChannel(LY)

        #outA = tot.TrB()
        outB = tot.TrA()

        #self.assertEqual(la.norm(outA.liouvillian()-LX),0)
        self.assertAlmostEqual(la.norm(outB-np.eye(4)),0,12)

    def test_jamiliokowski(self):

        phi = ch.QuantumChannel(np.eye(4),'choi')

        rho = phi.Jamiliokowski()

        self.assertEqual(la.norm(rho.density_matrix()-.25*np.eye(4)),0)


    def test_dual(self):

        D = uni.choi(N=3)
        DD = D.dual()
        DDD = DD.dual()

        self.assertEqual(la.norm(D.choi()-DDD.choi()),0)

        P = ch.QuantumChannel(D.ptm(),'ptm')
        PP = P.dual()
        PPP = PP.dual()

        self.assertEqual(la.norm(P.ptm()-PPP.ptm()),0)

        K = ch.QuantumChannel(D.kraus(),'kraus')
        KK = K.dual()
        KKK = KK.dual()

        self.assertEqual(la.norm(K.stiefel()-KKK.stiefel()),0)

        self.assertAlmostEqual(la.norm(DD.choi()-KK.choi()),0)
        self.assertAlmostEqual(la.norm(DD.choi()-PP.choi()),0)

if __name__ == '__main__':
    unittest.main()
