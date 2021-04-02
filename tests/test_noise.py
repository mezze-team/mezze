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
from mezze.noise import *
import numpy as np


class TestNoise(unittest.TestCase):

    def test_factory(self):
        factory = NoiseFactory(100, 1.0)

        # White
        n = factory.get({'type': 'white', 'amp': 0.3})
        self.assertIsInstance(n, WhiteNoise)

        # White with cutoff
        n = factory.get({'type': 'white', 'amp': 0.3, 'omega_min': 2*np.pi, 'omega_max': 2*np.pi*1e9})
        self.assertIsInstance(n, WhiteCutoffNoise)

        # Pink with cutoff
        n = factory.get({'type': 'pink', 'amp': 0.3, 'omega_min': 2*np.pi, 'omega_max': 2*np.pi*1e9})
        self.assertIsInstance(n, PinkCutoffNoise)

        # Gaussian
        n = factory.get({'type': 'gauss', 'amp': 17.0, 'corr_time': 0.8})
        self.assertIsInstance(n, GaussianNoise)
        self.assertEqual(17.0, n.amplitude)
        self.assertEqual(0.8, n.corr_time)

        # Exponential
        n = factory.get({'type': 'exp', 'amp': 14.0, 'corr_time': 0.2})
        self.assertIsInstance(n, ExponentialNoise)

        # Telegraph
        n = factory.get({'type': 'telegraph', 'vals': 12.0, 'taus': [2.0, 5.0]})
        self.assertIsInstance(n, TelegraphNoise)

        # User Defined
        n = factory.get({'type':'time', 'corrfn': lambda t: t + 1})
        self.assertIsInstance(n, UserDefinedNoise)

        # Null
        n = factory.get({'type': 'null'})
        self.assertIsInstance(n, NullNoise)

        with self.assertRaises(NotImplementedError):
            factory.get({'type': 'red'})


    def test_generation(self):
        factory = NoiseFactory(1, 1000)
        n = factory.get({'type': 'pink', 'amp': 0.3, 'omega_min': 2*np.pi, 'omega_max': 2*np.pi*1e9})
        self.assertIsInstance(n, PinkCutoffNoise)

        v = n.generate()
        self.assertEqual(len(v),1000)

if __name__ == '__main__':
    unittest.main()
