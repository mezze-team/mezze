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


class TestQuantumStateDtype(unittest.TestCase):

    def test_dm_dtype(self):

        D = ch.QuantumState(np.eye(2,dtype=np.complex128)*.5,'dm')
        self.assertEqual(type(D._dm.density_matrix),type(np.array([]))) #Changed Type
        self.assertEqual(D._dm.density_matrix.dtype,np.complex128)

        D = ch.QuantumState(np.eye(2)*.5,'dm')
        self.assertEqual(type(D._dm.density_matrix),type(np.array([]))) #Changed Type
        self.assertEqual(D._dm.density_matrix.dtype,np.complex128)

        self.assertEqual(type(D.density_matrix()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_matrix().dtype,np.complex128)

        self.assertEqual(type(D.density_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_vector().dtype,np.complex128)

        self.assertEqual(type(D.bloch_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.bloch_vector().dtype,np.float64)

        w,b = D.state_mixture()
        self.assertEqual(type(w),np.ndarray)
        self.assertEqual(w.dtype,np.float64)
        self.assertTrue(all([type(bb)==type(np.array([])) for bb in b])) #Changed Type
        self.assertTrue(all([bb.dtype==np.complex128 for bb in b]))

    def test_dv_dtype(self):

        D = ch.QuantumState(ch._vec(np.eye(2,dtype=np.complex128)*.5),'dv')
        self.assertEqual(type(D._dv.density_vector),type(np.array([]))) #Changed Type
        self.assertEqual(D._dv.density_vector.dtype,np.complex128)

        D = ch.QuantumState(ch._vec(np.eye(2)*.5),'dv')
        self.assertEqual(type(D._dv.density_vector),type(np.array([]))) #Changed Type
        self.assertEqual(D._dv.density_vector.dtype,np.complex128)

        self.assertEqual(type(D.density_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_vector().dtype,np.complex128)

        self.assertEqual(type(D.density_matrix()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_matrix().dtype,np.complex128)

        self.assertEqual(type(D.bloch_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.bloch_vector().dtype,np.float64)

        w,b = D.state_mixture()
        self.assertEqual(type(w),np.ndarray)
        self.assertEqual(w.dtype,np.float64)
        self.assertTrue(all([type(bb)==type(np.array([])) for bb in b])) #Changed Type
        self.assertTrue(all([bb.dtype==np.complex128 for bb in b]))

    def test_bv_dtype(self):

        D = ch.QuantumState(np.eye(3)[:,0],'bv')
        self.assertEqual(type(D._bv.bloch_vector),type(np.array([]))) #Changed Type
        self.assertEqual(D._bv.bloch_vector.dtype,np.float64)

        D = ch.QuantumState(np.eye(3,dtype=np.complex128)[:,0],'bv')
        self.assertEqual(type(D._bv.bloch_vector),type(np.array([]))) #Changed Type
        self.assertEqual(D._bv.bloch_vector.dtype,np.float64)

        self.assertEqual(type(D.bloch_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.bloch_vector().dtype,np.float64)

        self.assertEqual(type(D.density_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_vector().dtype,np.complex128)

        self.assertEqual(type(D.density_matrix()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_matrix().dtype,np.complex128)

        w,b = D.state_mixture()
        self.assertEqual(type(w),np.ndarray)
        self.assertEqual(w.dtype,np.float64)
        self.assertTrue(all([type(bb)==type(np.array([])) for bb in b])) #Changed Type
        self.assertTrue(all([bb.dtype==np.complex128 for bb in b]))

    def test_sm_dtype(self):

        D = ch.QuantumState((1.,[np.eye(2,dtype=np.complex128)[:,0]]),'sm')
        self.assertEqual(type(D._sm.weights),np.ndarray)
        self.assertEqual(D._sm.weights.dtype,np.float64)
        self.assertTrue(all([type(b)==type(np.array([])) for b in D._sm.basis])) #Changed Type
        self.assertTrue(all([b.dtype==np.complex128 for b in D._sm.basis]))

        D = ch.QuantumState((1.,[np.eye(2)[:,0]]),'sm')
        self.assertEqual(type(D._sm.weights),np.ndarray)
        self.assertEqual(D._sm.weights.dtype,np.float64)
        self.assertTrue(all([type(b)==type(np.array([])) for b in D._sm.basis])) #Changed Type
        self.assertTrue(all([b.dtype==np.complex128 for b in D._sm.basis]))

        w,b = D.state_mixture()
        self.assertEqual(type(w),np.ndarray)
        self.assertEqual(w.dtype,np.float64)
        self.assertTrue(all([type(bb)==type(np.array([])) for bb in b])) #Changed Type
        self.assertTrue(all([bb.dtype==np.complex128 for bb in b]))

        self.assertEqual(type(D.bloch_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.bloch_vector().dtype,np.float64)

        self.assertEqual(type(D.density_vector()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_vector().dtype,np.complex128)

        self.assertEqual(type(D.density_matrix()),type(np.array([]))) #Changed Type
        self.assertEqual(D.density_matrix().dtype,np.complex128)


if __name__ == '__main__':
    unittest.main()
