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
from mezze.random.SchWARMA import acv_from_b, acv_2_var, var_2_fid, extract_and_downconvert_from_sim, make_gauss_approximation
import mezze
import numpy as np
import scipy.signal as si


class TestSchWARMAConversion(unittest.TestCase):

    def test_fits_gauss(self):

        T_max = 100  # Doing 100 gates
        trotter_ratio = 50  # mezze trotter time steps per unit of time (small so I can do a lot of MC)

        NN_spec = 10
        corr_time = 10.0

        amp = .002

        config_specs = {'T_end': T_max, 'steps': T_max * trotter_ratio, 'Δsteps': 10, 'num_runs': 10,
                        'shsteps': NN_spec}
        # steps : the time step for simulations, Δsteps: recording interval, shteps: the number of SchWARMA time steps
        noise_specs = {'type': 'gauss', 'amp': amp, 'corr_time': corr_time}  # noise parameters.

        # First define config, that determines the simulation parameters for the time of evolution and timesteps
        config = mezze.simulation.SimulationConfig(time_length=config_specs['T_end'], num_steps=config_specs['steps'])
        config.parallel = True  # parallelizes over available cpu cores
        config.num_runs = config_specs['num_runs']  # number of monte-carlo runs
        config.num_cpus = 4  # number of cpus to parallelize over
        # config.get_unitary = True #report unitary
        config.time_sampling = True  # set to true to get the channel at intermediate times
        config.sampling_interval = config_specs['Δsteps']  # records the channel after every 10 time-steps
        config.realization_sampling = False  # set true if you want to record every noise(monte-carlo) realization

        # # Setup PMD
        pmd0 = mezze.pmd.PMD(control_dim=0, noise_dim=1, num_qubits=1, num_lindblad=0)
        pmd0.single_qubit_noise[3] = lambda c, gamma: gamma[
            0]  # sets single_qubit noise along z direction.  [3] corresponds to σz pauli basis
        pmd0.noise_hints[0] = noise_specs
        # build Hamiltonian and simulation
        ham = mezze.hamiltonian.HamiltonianFunction(pmd0)
        ctrls = np.zeros(
            (config_specs['steps'], 1))  # ctrls matrix as a function of time is zero since there is no control
        pham = mezze.hamiltonian.PrecomputedHamiltonian(pmd0, ctrls, config, ham.get_function())

        sim = mezze.simulation.Simulation(config, pmd0, pham)

        rx = SchWARMA.extract_sim_autocorrs(sim, 1. / sim.config.dt)[0]

        Tgate = 1.0
        Tgate2 = 2.5
        b_fit = extract_and_downconvert_from_sim(sim, Tgate)[0]
        b_fit2 = extract_and_downconvert_from_sim(sim, Tgate2)[0]

        Tgauss = 1.5
        b_gauss = make_gauss_approximation(sim, Tgauss)[0]

        # print(np.max(np.abs(b_fit-b_fit2)),np.max(b_fit))

        fids_fast = var_2_fid(acv_2_var(rx, ks=np.arange(T_max / sim.config.dt)))
        fids_slow = var_2_fid(acv_2_var(acv_from_b(b_fit), np.arange(T_max / Tgate)))
        fids_slow2 = var_2_fid(acv_2_var(acv_from_b(b_fit2), np.arange(T_max / Tgate2)))
        fids_gauss = var_2_fid(acv_2_var(acv_from_b(b_gauss), np.arange(T_max / Tgauss)))

        self.assertEqual(fids_fast[0], 1.0)
        self.assertEqual(fids_fast[0], fids_slow[0])
        self.assertEqual(fids_fast[0], fids_slow2[0])
        self.assertEqual(fids_fast[0], fids_gauss[0])

        ff = fids_fast[::int(Tgate/sim.config.dt)]
        fs = fids_slow
        self.assertAlmostEqual(ff[1], fs[1])
        self.assertLess(np.max(np.abs(ff-fs)), 1e-4)

        ff = fids_fast[::int(Tgate2/sim.config.dt)]
        fs = fids_slow2
        self.assertAlmostEqual(ff[1], fs[1])
        self.assertLess(np.max(np.abs(ff-fs)), 1e-4)

        ff = fids_fast[::int(Tgauss/sim.config.dt)]
        fs = fids_gauss
        self.assertAlmostEqual(ff[1], fs[1], 6)
        self.assertLess(np.max(np.abs(ff-fs)), 1e-4)

    def test_downconvert_from_SchWARMA(self):

        b = si.firwin(128, .2)
        b2 = SchWARMA.downconvert_from_SchWARMA(b, 2)

        fb = var_2_fid(acv_2_var(acv_from_b(b), np.arange(100)))[::2]
        fb2 = var_2_fid(acv_2_var(acv_from_b(b2), np.arange(100)))[:50]

        self.assertEqual(fb[0], 1.0)
        self.assertEqual(fb2[0], 1.0)

        self.assertLess(np.max(np.abs(fb-fb2)),1e-3)

if __name__ == '__main__':
    unittest.main()
