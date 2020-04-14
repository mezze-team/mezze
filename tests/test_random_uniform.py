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
from mezze.random import uniform

class TestRandomUniformQuantumObjects(unittest.TestCase):

    def test_unitary_as_channel(self):
        u = uniform.unitary(2)

        self.assertTrue(len(u.kraus())==1)
        self.assertTrue(u.kraus()[0].shape == (2,2))

        u = uniform.unitary(5)

        self.assertTrue(len(u.kraus())==1)
        self.assertTrue(u.kraus()[0].shape == (5,5))

    def test_unitary_as_matrix(self):
        u = uniform.unitary(3,as_channel=False)

        self.assertAlmostEqual(la.norm(u*u.H - np.eye(3)),0,14)
        self.assertAlmostEqual(np.abs(la.det(u)),1,14)
        self.assertEqual(type(u),np.matrix)
    
    def test_choi_as_channel(self):
        choi = uniform.choi(2)

        self.assertTrue(choi.is_valid())

    def test_choi_as_matrix(self):
        choi = uniform.choi(3,as_channel=False)

        self.assertTrue(choi.shape == (9,9))
        self.assertEqual(type(choi),np.matrix)
        self.assertEqual(choi.dtype, np.complex)

        self.assertTrue(ch.QuantumChannel(choi,'choi').is_valid())
    
    def test_stiefel_as_channel(self):
        stiefel = uniform.stiefel(2)

        self.assertTrue(stiefel.is_valid())

    def test_stiefel_as_matrix(self):
        stiefel = uniform.stiefel(3,as_channel=False)

        self.assertTrue(stiefel.shape == (27,3))
        self.assertEqual(type(stiefel),np.matrix)
        self.assertEqual(stiefel.dtype, np.complex)

        self.assertTrue(ch.QuantumChannel(stiefel,'stiefel').is_valid())

    def test_bloch_vector_as_state(self):
        bv = uniform.bloch_vector(3)

        self.assertTrue(bv.is_valid())

    def test_bloch_vector_as_matrix(self):
        bv = uniform.bloch_vector(4,as_state=False)

        self.assertTrue(bv.shape == (15,1))
        self.assertLessEqual(la.norm(bv),1)
        self.assertEqual(type(bv),np.matrix)
        self.assertTrue(ch.QuantumState(bv,'bv').is_valid())

    def test_pure_bloch_vector(self):
        bv = uniform.bloch_vector(2,pure=True)

        self.assertTrue(bv.is_valid() and bv.is_pure())

    def test_density_matrix_as_state(self):
        dm = uniform.density_matrix(3)

        self.assertTrue(dm.is_valid())

    def test_density_matrix_as_matrix(self):
        dm = uniform.density_matrix(4,as_state=False)

        self.assertTrue(dm.shape == (4,4))
        self.assertTrue(type(dm), np.matrix)
        self.assertTrue(dm.dtype == np.complex)
        self.assertTrue(ch.QuantumState(dm).is_valid())

    def test_rank_choi(self):

        for i in range(1,10):
            choi = uniform.choi(3,i)

            self.assertTrue(choi.rank()==i)

    def test_rank_stiefel(self):

        for i in range(1,10):
            s = uniform.stiefel(3,i)

            self.assertTrue(s.rank()==i)

    def test_rank_density_matrix(self):

        for i in range(1,10):
            dm = uniform.density_matrix(9,i)

            self.assertTrue(dm.rank()==i)

    def test_unitary_seed(self):

        rgen1 = np.random.RandomState(1)
        U1 = uniform.unitary(2,False,rgen1)
        
        rgen2 = np.random.RandomState(1)
        U2 = uniform.unitary(2,False,rgen2)
        
        self.assertEqual(la.norm(U1-U2),0)

    def test_choi_seed(self):

        rgen1 = np.random.RandomState(1)
        choi1 = uniform.choi(2,4,False,rgen1)

        rgen2 = np.random.RandomState(1)
        choi2 = uniform.choi(2,4,False,rgen2)

        self.assertEqual(la.norm(choi1-choi2),0)

    def test_stiefel_seed(self):

        rgen1 = np.random.RandomState(1)
        s1 = uniform.stiefel(2,4,False,rgen1)

        rgen2 = np.random.RandomState(1)
        s2 = uniform.stiefel(2,4,False,rgen2)

        self.assertEqual(la.norm(s1-s2),0)
        
    def test_density_matrix_seed(self):

        rgen1 = np.random.RandomState(1)
        dm1 = uniform.density_matrix(2,2,False,rgen1)

        rgen2 = np.random.RandomState(1)
        dm2 = uniform.density_matrix(2,2,False,rgen2)
        
        self.assertEqual(la.norm(dm1-dm2),0)
        
    def test_bloch_vector_seed(self):

        rgen1 = np.random.RandomState(1)
        bv1 = uniform.bloch_vector(2,False,False,rgen1)

        rgen2 = np.random.RandomState(1)
        bv2 = uniform.bloch_vector(2,False,False,rgen2)

        self.assertEqual(la.norm(bv1-bv2),0)

if __name__ == '__main__':
    unittest.main()
