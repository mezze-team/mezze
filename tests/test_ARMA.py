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
from mezze.random.SchWARMA import ARMA
import numpy as np

class TestARMA(unittest.TestCase):

    def test_init_and_step(self):
        model = ARMA(a = [1,2,3], b = [4,5])

        self.assertTrue(np.all(model.a == [1,2,3]) and \
                        np.all(model.b == [4,5]))

        for i in range(100):
            model.step(np.random.randn(1))

        #now check other inputs
        model.step(1.)
        model.step(0)
        model.step([2])

    def test_MA_in_depth(self):
        model = ARMA( a=[],b = [1., .5, .25])
        
        y = model.step(1)

        self.assertEqual(y,1)

        y = model.step(2)

        self.assertEqual(y, 2.5)

        y = model.step(3)

        self.assertEqual(y, 4.25)

        y = model.step(0)

        self.assertEqual(y,2)

    def test_AR_in_depth(self):

        model = ARMA(a = [1,.5], b = [1])

        y = model.step(1)

        self.assertEqual(y,1)

        y = model.step(1)

        self.assertEqual(y,2)

        y = model.step(1)
        
        self.assertEqual(y,3.5)

if __name__ == '__main__':
    unittest.main()
