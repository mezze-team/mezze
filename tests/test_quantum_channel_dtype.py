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


class TestQuantumChannelDtype(unittest.TestCase):

    def test_liou_dtype(self):

        D = ch.QuantumChannel(np.eye(4,dtype=np.complex),'liou')
        self.assertEqual(type(D._liou.liouvillian),np.matrix)
        self.assertEqual(D._liou.liouvillian.dtype,np.complex)
        
        D = ch.QuantumChannel(np.eye(4),'liou')
        self.assertEqual(type(D._liou.liouvillian),np.matrix)
        self.assertEqual(D._liou.liouvillian.dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel()),np.matrix)
        self.assertEqual(D.stiefel().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))


    def test_chi_dtype(self):

        D = ch.QuantumChannel(.25*np.eye(4,dtype=np.complex),'chi')
        self.assertEqual(type(D._chi.chi_matrix),np.matrix)
        self.assertEqual(D._chi.chi_matrix.dtype,np.complex)
        
        D = ch.QuantumChannel(.25*np.eye(4),'chi')
        self.assertEqual(type(D._chi.chi_matrix),np.matrix)
        self.assertEqual(D._chi.chi_matrix.dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.stiefel()),np.matrix)
        self.assertEqual(D.stiefel().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))

    def test_stiefel_dtype(self):

        D = ch.QuantumChannel(np.eye(2),'stiefel')
        self.assertEqual(type(D._stiefel.stiefel_matrix),np.matrix)
        self.assertEqual(D._stiefel.stiefel_matrix.dtype,np.complex)
        self.assertEqual(D._stiefel.stiefel_matrix.shape,(8,2))

        D = ch.QuantumChannel(np.eye(2,dtype=np.complex),'stiefel')
        self.assertEqual(type(D._stiefel.stiefel_matrix),np.matrix)
        self.assertEqual(D._stiefel.stiefel_matrix.dtype,np.complex)
        self.assertEqual(D._stiefel.stiefel_matrix.shape,(8,2))

        self.assertEqual(type(D.stiefel()),np.matrix)
        self.assertEqual(D.stiefel().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))
    
    def test_stiefel2_dtype(self):

        D = ch.QuantumChannel(np.concatenate((np.eye(2),np.zeros((6,2))),0),'stiefel2')
        self.assertEqual(type(D._stiefel2.stiefel_matrix2),np.matrix)
        self.assertEqual(D._stiefel2.stiefel_matrix2.dtype,np.complex)
        self.assertEqual(D._stiefel2.stiefel_matrix2.shape,(8,2))

        D = ch.QuantumChannel(np.concatenate((np.eye(2,dtype=np.complex),np.zeros((6,2),dtype=np.complex)),0),'stiefel2')
        self.assertEqual(type(D._stiefel2.stiefel_matrix2),np.matrix)
        self.assertEqual(D._stiefel2.stiefel_matrix2.dtype,np.complex)
        self.assertEqual(D._stiefel2.stiefel_matrix2.shape,(8,2))

        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))

    def test_kraus_dtype(self):

        D = ch.QuantumChannel([np.eye(2,dtype=np.complex)],'kraus')
        self.assertTrue(all([type(A)==np.matrix for A in D._kraus.op_list]))
        self.assertTrue(all([A.dtype==np.complex for A in D._kraus.op_list]))
        
        D = ch.QuantumChannel([np.eye(2)],'kraus')
        self.assertTrue(all([type(A)==np.matrix for A in D._kraus.op_list]))
        self.assertTrue(all([A.dtype==np.complex for A in D._kraus.op_list]))

        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))
    
        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel()),np.matrix)
        self.assertEqual(D.stiefel().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
    def test_choi_dtype(self):

        D = ch.QuantumChannel(.5*np.eye(4,dtype=np.complex),'choi')
        self.assertEqual(type(D._choi.choi),np.matrix)
        self.assertEqual(D._choi.choi.dtype,np.complex)
        
        D = ch.QuantumChannel(.5*np.eye(4),'choi')
        self.assertEqual(type(D._choi.choi),np.matrix)
        self.assertEqual(D._choi.choi.dtype,np.complex)

        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel()),np.matrix)
        self.assertEqual(D.stiefel().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))
    
    def test_ptm_dtype(self):

        D = ch.QuantumChannel(np.eye(4),'ptm')
        self.assertEqual(type(D._ptm.ptm),np.matrix)
        self.assertEqual(D._ptm.ptm.dtype,np.float)
        
        D = ch.QuantumChannel(np.eye(4,dtype=np.complex),'ptm')
        self.assertEqual(type(D._ptm.ptm),np.matrix)
        self.assertEqual(D._ptm.ptm.dtype,np.float)

        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertEqual(type(D.choi()),np.matrix)
        self.assertEqual(D.choi().dtype,np.complex)

        self.assertEqual(type(D.liouvillian()),np.matrix)
        self.assertEqual(D.liouvillian().dtype,np.complex)

        self.assertEqual(type(D.chi()),np.matrix)
        self.assertEqual(D.chi().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel()),np.matrix)
        self.assertEqual(D.stiefel().dtype,np.complex)
        
        self.assertEqual(type(D.stiefel2()),np.matrix)
        self.assertEqual(D.stiefel2().dtype,np.complex)
        
        self.assertEqual(type(D.ptm()),np.matrix)
        self.assertEqual(D.ptm().dtype,np.float)
        
        self.assertTrue(all([type(A)==np.matrix for A in D.kraus()]))
        self.assertTrue(all([A.dtype==np.complex for A in D.kraus()]))
if __name__ == '__main__':
    unittest.main()
