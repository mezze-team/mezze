# Copyright: 2015-2020 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
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
import numpy as np
import scipy.signal as si
import cirq

from mezze.tfq import *

import tensorflow as tf
import tensorflow_quantum as tfq


class TestTFQ(unittest.TestCase):

    def test_single_z_dephasing(self):
        qbit = cirq.GridQubit.rect(1,1)[0]
        circ = cirq.Circuit(cirq.X(qbit)**.5 for _ in range(10))

        S = SimpleDephasingSchWARMAFier(.1*np.ones(1),[1,])
        noisy_circ, syms = S.schwarmafy(circ)

        self.assertEqual(len(circ.moments),len(syms))
        self.assertEqual(2*len(circ.moments),len(noisy_circ.moments))
        self.assertTrue(all([circ.moments[i]==noisy_circ.moments[2*i] for i in range(len(circ.moments))]))
        self.assertTrue(all([type(noisy_circ.moments[2*i+1].operations[0].gate)==cirq.ops.common_gates.ZPowGate for i in range(len(circ.moments))]))

        noise = S.gen_noise_instances(circ,5)

        self.assertEqual(noise.shape[0],5)
        self.assertEqual(noise.shape[1], len(circ.moments))

    def test_single_x_dephasing(self):
        qbit = cirq.GridQubit.rect(1,1)[0]
        circ = cirq.Circuit(cirq.X(qbit)**.5 for _ in range(10))

        S = SimpleDephasingSchWARMAFier(.1*np.ones(1),[1,], op=cirq.rx)
        noisy_circ, syms = S.schwarmafy(circ)

        self.assertEqual(2*len(circ.moments),len(noisy_circ.moments))
        self.assertTrue(all([circ.moments[i]==noisy_circ.moments[2*i] for i in range(len(circ.moments))]))
        self.assertTrue(all([type(noisy_circ.moments[2*i+1].operations[0].gate)==cirq.ops.common_gates.XPowGate for i in range(len(circ.moments))]))

        noise = S.gen_noise_instances(circ,5)

        self.assertEqual(noise.shape[0],5)
        self.assertEqual(noise.shape[1], len(circ.moments))

    def test_multi_dephasing(self):

        # Note this may have identities in it which can mess with moments
        circ = cirq.generate_boixo_2018_supremacy_circuits_v2_grid(2,2,10,0)

        S = SimpleDephasingSchWARMAFier([0,],[1,-.9])

        noisy_circ, syms = S.schwarmafy(circ)

        self.assertEqual(len(syms), len(circ.moments)*len(circ.all_qubits()))
        self.assertEqual(2*len(circ.moments),len(noisy_circ.moments))

        sim = TensorFlowSchWARMASim(circ, S)

        psi = sim.state_sim(1)
        psi2 = tfq.layers.State()(circ).to_tensor().numpy()

        self.assertEqual(np.linalg.norm(psi-psi2), 0.0)

        dm = sim.dm_sim(1)
        dm2 = psi2.T@psi.conj()

        self.assertEqual(np.linalg.norm(dm-dm2), 0.0)

    def test_max_time_corr(self):

        S = SimpleDephasingSchWARMAFier([1,],[1,-.999999999])

        max_time=4096
        corr_time = S.corr_time(max_time=max_time)
        self.assertEqual(corr_time,max_time)

    def test_additive_schwarmafier(self):

        S1 = SimpleDephasingSchWARMAFier([.01,],[1,-.9])
        S2 = SimpleDephasingSchWARMAFier([.2,],[1,])
        S = AdditiveSchWARMAFier([S1,S2])

        qbit = cirq.GridQubit.rect(1,1)[0]
        circ = cirq.Circuit(cirq.X(qbit)**.5 for _ in range(10))

        noisy_circ, syms = S.schwarmafy(circ)

        self.assertEqual(len(circ.moments),len(syms))
        self.assertEqual(2*len(circ.moments),len(noisy_circ.moments))
        self.assertTrue(all([circ.moments[i]==noisy_circ.moments[2*i] for i in range(len(circ.moments))]))
        self.assertTrue(all([type(noisy_circ.moments[2*i+1].operations[0].gate)==cirq.ops.common_gates.ZPowGate for i in range(len(circ.moments))]))

        noise = S.gen_noise_instances(circ,5)

        self.assertEqual(noise.shape[0],5)
        self.assertEqual(noise.shape[1], len(circ.moments))

    def test_compare_cirq_tf(self):

        # Note this may have identities in it which can mess with moments
        circ = cirq.generate_boixo_2018_supremacy_circuits_v2_grid(2,2,10,0)
        S = SimpleDephasingSchWARMAFier([0,],[1,-.9])

        noisy_circ, syms = S.schwarmafy(circ)

        sim1 = CirqSchWARMASimulator(circ,S)
        sim2 = TensorFlowSchWARMASim(circ,S)

        rgen1 = np.random.RandomState(0)
        rgen2 = np.random.RandomState(0)

        dm1 = sim1.dm_sim(num_MC=5, rgen=rgen1)
        dm2 = sim2.dm_sim(num_MC=5, rgen=rgen2)

        self.assertLessEqual(np.linalg.norm(dm1-dm2),1e-5)

        st1 = sim1.state_sim(num_MC=5, rgen=rgen1)
        st2 = sim2.state_sim(num_MC=5, rgen=rgen2)
        
        self.assertLessEqual(np.max(np.abs(st1-st2)),1e-5)

    def test_noise_mc(self):

        circ = cirq.generate_boixo_2018_supremacy_circuits_v2_grid(2,2,10,0)

        S1 = SimpleDephasingSchWARMAFier([0,],[1,-.9])
        S2 = SimpleDephasingSchWARMAFier([0,],[1,-.9])

        rgen1 = np.random.RandomState(0)
        rgen2 = np.random.RandomState(0)

        out1 = S1.gen_noise_instances(circ,5,rgen1)
        out2 = S2.gen_noise_instances(circ,5,rgen2)

        self.assertEqual(np.linalg.norm(out1-out2),0.0)




if __name__ == '__main__':
    unittest.main()
