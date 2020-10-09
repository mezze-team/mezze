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
import mezze.simulation as simul
import mezze.hamiltonian as hamil
import mezze.channel as ch
from mezze.pmd import PMD
import numpy as np

class TestPMD(PMD):
    def __init__(self):
        control_dim = 0
        noise_dim = 0
        num_lindblad = 0
        PMD.__init__(self,control_dim,noise_dim,num_lindblad,1)

class TestSimOutputs(unittest.TestCase):

    def test_sequential(self):
        config = simul.SimulationConfig(1., 10)
        config.parallel = False
        config.num_runs = 3
        config.time_sampling = False
        config.realization_sampling = False

        pmd = TestPMD()
        ham = hamil.HamiltonianFunction(pmd)
        ctrls = np.zeros((10,0))
        pham = hamil.PrecomputedHamiltonian(pmd,ctrls,config,ham.get_function())
        sim = simul.Simulation(config,pmd,pham)
        report = sim.run()

        self.assertIsInstance(report.channel,ch.QuantumChannel)
        self.assertIs(report.liouvillian_samples,None)
        self.assertIs(report.time_samples,None)

    def test_sequential_time(self):
        config = simul.SimulationConfig(1., 10)
        config.parallel = False
        config.num_runs = 3
        config.time_sampling = True
        config.sampling_interval = 5
        config.realization_sampling = False

        pmd = TestPMD()
        ham = hamil.HamiltonianFunction(pmd)
        ctrls = np.zeros((10,0))
        pham = hamil.PrecomputedHamiltonian(pmd,ctrls,config,ham.get_function())
        sim = simul.Simulation(config,pmd,pham)
        report = sim.run()

        self.assertIsInstance(report.channel,ch.QuantumChannel)
        self.assertIs(report.liouvillian_samples,None)
        self.assertIsInstance(report.time_samples,list)
        self.assertEqual(len(report.time_samples),2)
        self.assertTrue(all([type(c) == ch.QuantumChannel for c in report.time_samples]))
    
    def test_sequential_real(self):
        config = simul.SimulationConfig(1., 10)
        config.parallel = False
        config.num_runs = 3
        config.time_sampling = False
        config.realization_sampling = True

        pmd = TestPMD()
        ham = hamil.HamiltonianFunction(pmd)
        ctrls = np.zeros((10,0))
        pham = hamil.PrecomputedHamiltonian(pmd,ctrls,config,ham.get_function())
        sim = simul.Simulation(config,pmd,pham)
        report = sim.run()

        self.assertIsInstance(report.channel,ch.QuantumChannel)
        self.assertIsInstance(report.liouvillian_samples,list)
        self.assertIs(report.time_samples,None)
        self.assertEqual(len(report.liouvillian_samples), 3)
        self.assertTrue(all([type(c) == ch.QuantumChannel for c in report.liouvillian_samples]))

    def test_sequential_real_time(self):
        config = simul.SimulationConfig(1., 10)
        config.parallel = False
        config.num_runs = 3
        config.time_sampling = True
        config.sampling_interval = 5
        config.realization_sampling = True

        pmd = TestPMD()
        ham = hamil.HamiltonianFunction(pmd)
        ctrls = np.zeros((10,0))
        pham = hamil.PrecomputedHamiltonian(pmd,ctrls,config,ham.get_function())
        sim = simul.Simulation(config,pmd,pham)
        report = sim.run()

        self.assertIsInstance(report.channel,ch.QuantumChannel)
        self.assertIsInstance(report.liouvillian_samples,list)
        self.assertIsInstance(report.time_samples,list)
        self.assertEqual(len(report.liouvillian_samples), 3)
        self.assertTrue(all([type(c) == ch.QuantumChannel for c in report.liouvillian_samples]))
        self.assertEqual(len(report.liouvillian_samples), 3)
        self.assertTrue(all([len(row) ==2 for row in report.time_samples]))
        self.assertTrue(all([all([type(c) == ch.QuantumChannel for c in row]) for row in report.time_samples]))

if __name__ == '__main__':
    unittest.main()
