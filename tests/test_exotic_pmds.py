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

import mezze.implementations as implementations
import mezze.simulation as simulation
import mezze.hamiltonian as hamiltonian
import mezze.channel as basis
import unittest
import mezze.core as core
import numpy as np
import scipy.linalg as la


class TestExoticPMDs(unittest.TestCase):

    def test_blatt_pmd_comp_1(self):
        config=simulation.SimulationConfig(time_length=.1,num_steps=50)
        config.num_runs=1
        pmd = implementations.BlattPMDComp(1)
        pmd.noise_off()
        ham = hamiltonian.HamiltonianFunction(pmd)
        ctrls = np.zeros((50,0))
        pham =  hamiltonian.PrecomputedHamiltonian(pmd,ctrls,config,
                                                   ham.get_function())
        sim = simulation.Simulation(config,pmd,pham)
        report = sim.run()
        self.assertEqual(la.norm(report.channel.liouvillian()-np.eye(4)),0)

    def test_blatt_pmd_comp_3(self):
        config=simulation.SimulationConfig(time_length=.1,num_steps=50)
        config.num_runs=1
        pmd = implementations.BlattPMDComp(3)
        pmd.noise_off()
        ham = hamiltonian.HamiltonianFunction(pmd)
        ctrls = np.zeros((50,0))
        pham =  hamiltonian.PrecomputedHamiltonian(pmd,ctrls,config,
                                                   ham.get_function())
        sim = simulation.Simulation(config,pmd,pham)
        report = sim.run()
        self.assertEqual(la.norm(report.channel.liouvillian()-np.eye(64)),0)


if __name__ == '__main__':
    unittest.main()
