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
import mezze.random.SchWARMA as SchWARMA
import numpy as np
import scipy.linalg as la
import copy
import scipy.special as sp
import mezze.channel as ch

s_i = np.array([[1,0],[0,1]],dtype=complex)
s_x = np.array([[0,1],[1,0]],dtype=complex)
s_y = np.array([[0,-1j],[1j,0]],dtype=complex)
s_z = np.array([[1,0],[0,-1]],dtype=complex)



class TestSchAR(unittest.TestCase):

    def test_init(self):

        ARMAS = [SchWARMA.ARMA(np.random.randn(np.random.randint(5))+2) for _ in range(3)]
        D = {'I': ch.QuantumChannel(np.array(np.eye(4,dtype=complex)))}
        AR = SchWARMA.SchWAR(ARMAS, [s_x, s_y, s_z], D)

    def test_bad_init(self):
        ARMAS = [SchWARMA.ARMA(np.random.randn(5), np.random.randn(2)) for _ in range(3)]
        D = {'I': ch.QuantumChannel(np.array(np.eye(4,dtype=complex)))}
        self.assertRaises(AssertionError, SchWARMA.SchWAR, 
                                        *[ARMAS, [s_x, s_y, s_z], D])


    def test_1d_fit(self):

        D = {'I': ch.QuantumChannel(np.array(np.eye(4)))}
        AR = SchWARMA.SchWAR([], [s_z], D)

        alpha = np.sqrt(.00001)
        sigma = 1.#1./2.

        arg = lambda t: alpha**2*sigma*(np.sqrt(np.pi)*t*sp.erf(t/sigma)+sigma*(np.exp(-t**2/sigma**2)-1))
        args = np.exp(-2*np.array([arg(t) for t in range(100)]))
        args = (1.-args)/2.

        AR.fit([args], 5)
        self.assertEqual(5,len(AR.models[0].a))
        self.assertEqual(1,len(AR.models[0].b))
        AR.init_ARMAs()
        AR['I']


        #Note now I have brackets on the second argument
        AR.fit([args], [7])
        self.assertEqual(7,len(AR.models[0].a))
        self.assertEqual(1,len(AR.models[0].b))
        AR.init_ARMAs()
        L = AR['I'].liouvillian()

        #Known structure of \sigma_z arma
        self.assertEqual(la.norm(L-np.diag(np.diag(L))),0.0)
        self.assertEqual(L[0,0], L[3,3])
        self.assertAlmostEqual(L[0,0], 1.,9)
        self.assertEqual(L[1,1], np.conj(L[2,2]))



    def test_3d_fit(self):
        D = {'I': ch.QuantumChannel(np.array(np.eye(4,dtype=complex)))}
        AR = SchWARMA.SchWAR([], [s_z], D)

        alpha = np.sqrt(.00001)
        sigma = 1.#1./2.

        arg = lambda t: alpha**2*sigma*(np.sqrt(np.pi)*t*sp.erf(t/sigma)+sigma*(np.exp(-t**2/sigma**2)-1))
        args = np.exp(-2*np.array([arg(t) for t in range(100)]))
        args = (1.-args)/2.

        #Note now I have brackets on the second argument
        AR.fit([args, args, args], [3,5,7])
        self.assertEqual(3,len(AR.models[0].a))
        self.assertEqual(5,len(AR.models[1].a))
        self.assertEqual(7,len(AR.models[2].a))
        AR.init_ARMAs()
        L = AR['I']


if __name__ == '__main__':
    unittest.main()
