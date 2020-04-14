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
import scipy.linalg as la
import mezze.core as core
import numpy as np
from mezze.channel import *
from mezze.channel import _vec as vec

class TestBasis(unittest.TestCase):

    def test_pauli_1(self):

        pb = PauliBasis()

        for i,b in enumerate([core.s_i, core.s_x, core.s_y, core.s_z]):
            self.assertEqual(la.norm(b-pb.basis_list[i]),0)
            self.assertEqual(type(pb.basis_list[i]),np.matrix)
            self.assertEqual(pb.basis_list[i].dtype, np.complex)

    def test_pauli_2(self):
        
        pb = PauliBasis(2)

        self.assertEqual(len(pb.basis_list),16)
        self.assertTrue([b.shape==(16,16) for b in pb.basis_list])
        self.assertTrue([type(b)==np.matrix for b in pb.basis_list])

        idx = 0

        for b1 in [core.s_i, core.s_x, core.s_y, core.s_z]:
            for b2 in [core.s_i, core.s_x, core.s_y, core.s_z]:
                self.assertEqual(la.norm(np.kron(b1,b2)-pb.basis_list[idx]),0)
                idx+=1

    def test_pauli_xform_matrix_1(self):

        pb = PauliBasis()
        
        A = pb.transform_matrix(False,False)

        for i in range(A.shape[1]):
            self.assertEqual(la.norm(A[:,i] - vec(pb.basis_list[i])),0)

        self.assertEqual(type(A),np.matrix)
        self.assertEqual(A.dtype,np.complex)
        self.assertEqual(la.norm(A.H*A-2*np.eye(4)),0)

        A,c = pb.transform_matrix(False,True)
        self.assertEqual(c,.5)
        self.assertEqual(la.norm(c*A.H*A-np.eye(4)),0)

        A = pb.transform_matrix(True,False)
        self.assertAlmostEqual(la.norm(A.H*A-np.eye(4)),0,15)


    def test_pauli_xform_matrix_2(self):

        pb = PauliBasis(2)
        
        A = pb.transform_matrix(False,False)

        for i in range(A.shape[1]):
            self.assertEqual(la.norm(A[:,i] - vec(pb.basis_list[i])),0)

        self.assertEqual(type(A),np.matrix)
        self.assertEqual(A.dtype, np.complex)
        self.assertEqual(la.norm(A.H*A-4*np.eye(16)),0)

        A,c = pb.transform_matrix(False,True)
        self.assertEqual(c,.25)
        self.assertEqual(la.norm(c*A.H*A-np.eye(16)),0)

        A = pb.transform_matrix(True,False)
        self.assertAlmostEqual(la.norm(A.H*A-np.eye(16)),0,15)

    def test_gm_1(self):

        gm = GellMannBasis()
        for  b in gm.basis_list:
            self.assertTrue(type(b), np.matrix)
            self.assertEqual(b.dtype, np.complex)

    def test_gm_xform_matrix_1(self):

        gm = GellMannBasis()

        A = gm.transform_matrix(False,False)

        for i in range(A.shape[1]):
            self.assertEqual(la.norm(A[:,i] - vec(gm.basis_list[i])),0)
        
        A,c = gm.transform_matrix(False,True)

        self.assertEqual(c,1./3.)
        self.assertAlmostEqual(la.norm(c*A*A.H-np.eye(9)),0,14)

    def test_gm_xform_matrix_2(self):

        gm = GellMannBasis(2)

        A = gm.transform_matrix(False,False)

        for i in range(A.shape[1]):
            self.assertEqual(la.norm(A[:,i] - vec(gm.basis_list[i])),0)
        
        A,c = gm.transform_matrix(False,True)

        self.assertEqual(c,1./9.)
        self.assertAlmostEqual(la.norm(c*A*A.H-np.eye(81)),0,14)

if __name__ == '__main__':
    unittest.main()
