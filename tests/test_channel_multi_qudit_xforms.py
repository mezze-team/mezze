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
import mezze.random.uniform as uniform

class TestMultiXforms(unittest.TestCase):

    def test_2_qubit_bv(self):

        zero = ch.QuantumState([[1,0],[0,0]],'dm')
        zz = zero^zero

        #test = np.zeros((15,1))
        #test[0] = 1
        #self.assertEqual(la.norm(test-zz.bloch_vector()),0)
        self.assertEqual(la.norm(zz.bloch_vector()),1)

    def test_2_qubit_bv_random(self):

        bv = uniform.bloch_vector(4,pure=True)

        self.assertAlmostEqual(np.trace(bv.density_matrix()),1,12)

        dm = uniform.density_matrix(4,rank=1)

        self.assertAlmostEqual(la.norm(dm.bloch_vector()),1,12)

        bv2 = ch.QuantumState(dm.bloch_vector(),'bv')

        dv1 = dm.density_vector()
        dv2 = bv2.density_vector()

        self.assertAlmostEqual(la.norm(dm.density_matrix()-bv2.density_matrix()),0,15)

    def test_2_qubit_chi(self):

        I = ch.QuantumChannel(np.eye(2),'unitary')
        X = ch.QuantumChannel([[0,1],[1,0]],'unitary')

        comp = I^X

        out_chi = np.zeros((16,16))
        out_chi[1,1] = 1.

        self.assertEqual(la.norm(out_chi-comp.chi()),0)

    def test_2_qubit_ptm(self):
        
        I2 = ch.QuantumChannel(np.eye(4),'unitary')

        self.assertEqual(la.norm(np.eye(16)-I2.ptm()),0)

    def test_2_qubit_ptm_O(self):

        U = uniform.choi(N=4,rank=1)
    
        self.assertAlmostEqual(np.abs(la.det(U.ptm()[1:,1:])),1.0,12)

    def test_1_qutrit_ptm_O(self):

        U = uniform.choi(N=3,rank=1)
    
        self.assertAlmostEqual(np.abs(la.det(U.ptm()[1:,1:])),1.0,12)
    
    def test_2_qutrit_ptm_O(self):

        U = uniform.choi(N=9,rank=1)
    
        self.assertAlmostEqual(np.abs(la.det(U.ptm()[1:,1:])),1.0,10)
    
    def test_2_qubit_chi_rand(self):

        chi1 = np.random.rand(4)
        chi1 = chi1/np.sum(chi1)
        g1 = ch.QuantumChannel(np.diag(chi1),'chi')

        chi2 = np.random.rand(4)
        chi2 = chi1/np.sum(chi2)
        g2 = ch.QuantumChannel(np.diag(chi2),'chi')

        g = g1^g2

        self.assertAlmostEqual(la.norm(np.diag(np.kron(chi1,chi2))-g.chi()),0,15)

    def test_2_qutrit_chi(self):
        I = ch.QuantumChannel(np.eye(3),'unitary')


    def test_2_qutrit_chi_rand(self):

        chi1 = np.random.rand(9)
        chi1 = chi1/np.sum(chi1)
        g1 = ch.QuantumChannel(np.diag(chi1),'chi')

        chi2 = np.random.rand(9)
        chi2 = chi1/np.sum(chi2)
        g2 = ch.QuantumChannel(np.diag(chi2),'chi')

        g = g1^g2

        self.assertAlmostEqual(la.norm(np.diag(np.kron(chi1,chi2))-g.chi()),0,14)

    def test_2_qubit_random_op(self):

        state = uniform.density_matrix(4)
        op = uniform.choi(4)

        out = op*state

        op2 = ch.QuantumChannel(op.ptm(),'ptm')

        self.assertAlmostEqual(la.norm(op.choi()-op2.choi()),0,14)

        #This illustrates the ugliness of wanting a unit sphere
        bvout = op.ptm()[1:,1:]@state.bloch_vector()+op.ptm()[1:,0,np.newaxis]*1./np.sqrt(3)
        bvout2 = ch.QuantumState(bvout,'bv')

        self.assertAlmostEqual(la.norm(out.bloch_vector()-bvout),0,15)

        self.assertAlmostEqual(la.norm(out.density_matrix()-bvout2.density_matrix()),0,14)


    def test_1_gubit_random_op(self):

        state = uniform.density_matrix(2)
        op = uniform.choi(2)

        out = op*state

        op2 = ch.QuantumChannel(op.ptm(),'ptm')

        self.assertAlmostEqual(la.norm(op.choi()-op2.choi()),0,14)
        
        bvout = op.ptm()[1:,1:]@state.bloch_vector()+op.ptm()[1:,0,np.newaxis]
        bvout2 = ch.QuantumState(bvout,'bv')

        self.assertAlmostEqual(la.norm(out.bloch_vector()-bvout),0,14)

        self.assertAlmostEqual(la.norm(out.density_matrix()-bvout2.density_matrix()),0,14)
      

    def test_1_qutrit_random_op(self):

        state = uniform.density_matrix(3)
        op = uniform.choi(3)

        out = op*state

        op2 = ch.QuantumChannel(op.ptm(),'ptm')

        self.assertAlmostEqual(la.norm(op.choi()-op2.choi()),0,14)
        
        bvout = op.ptm()[1:,1:]@state.bloch_vector()+op.ptm()[1:,0,np.newaxis]*1./np.sqrt(2)
        bvout2 = ch.QuantumState(bvout,'bv')

        self.assertAlmostEqual(la.norm(out.bloch_vector()-bvout),0,14)

        self.assertAlmostEqual(la.norm(out.density_matrix()-bvout2.density_matrix()),0,14)


    def test_2_qutrit_random_op(self):

        state = uniform.density_matrix(9)
        op = uniform.choi(9)

        out = op*state

        op2 = ch.QuantumChannel(op.ptm(),'ptm')

        self.assertAlmostEqual(la.norm(op.choi()-op2.choi()),0,14)

        #This illustrates the ugliness of wanting a unit sphere
        bvout = op.ptm()[1:,1:]@state.bloch_vector()+op.ptm()[1:,0,np.newaxis]*1./np.sqrt(8)
        bvout2 = ch.QuantumState(bvout,'bv')

        self.assertAlmostEqual(la.norm(out.bloch_vector()-bvout),0,15)

        self.assertAlmostEqual(la.norm(out.density_matrix()-bvout2.density_matrix()),0,14)

if __name__ == '__main__':
    unittest.main()
