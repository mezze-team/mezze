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
import mezze.channel as ch

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

        sim1 = CirqSchWARMASim(circ,S)
        sim2 = TensorFlowSchWARMASim(circ,S)

        rgen1 = np.random.RandomState(0)
        rgen2 = np.random.RandomState(0)

        dm1 = sim1.dm_sim(num_MC=5, rgen=rgen1)
        dm2 = sim2.dm_sim(num_MC=5, rgen=rgen2)

        self.assertLessEqual(np.linalg.norm(dm1-dm2),1e-5)

        st1 = sim1.state_sim(num_MC=5, rgen=rgen1)
        st2 = sim2.state_sim(num_MC=5, rgen=rgen2)
        
        self.assertLessEqual(np.max(np.abs(st1-st2)),1e-5)

    def test_unitary_props_sim(self):

        circ = cirq.generate_boixo_2018_supremacy_circuits_v2_grid(2,2,10,0)
        S = SimpleDephasingSchWARMAFier([0,],[1,-.9])

        noisy_circ, syms = S.schwarmafy(circ)

        sim = CirqSchWARMASim(circ,S)

        props = sim.unitary_props_sim(num_MC=10,rgen=np.random.RandomState(124))
        self.assertEqual(len(props),10)
        self.assertEqual(type(props[0]),ch.QuantumChannel)

        dm = sim.dm_sim(num_MC=1,rgen=np.random.RandomState(124))

        state = ch.QuantumState(np.eye(4**4)[0,:,np.newaxis],'dv')

        dm2 = props[0]*state

        self.assertLess(np.linalg.norm(dm-dm2.density_matrix()),1e-6)

        props = sim.unitary_props_sim(num_MC=10,as_channel=False,rgen=np.random.RandomState(124))
        self.assertEqual(len(props),10)
        self.assertEqual(type(props[0]),np.ndarray)

    def test_noise_mc(self):

        circ = cirq.generate_boixo_2018_supremacy_circuits_v2_grid(2,2,10,0)

        S1 = SimpleDephasingSchWARMAFier([0,],[1,-.9])
        S2 = SimpleDephasingSchWARMAFier([0,],[1,-.9])

        rgen1 = np.random.RandomState(0)
        rgen2 = np.random.RandomState(0)

        out1 = S1.gen_noise_instances(circ,5,rgen1)
        out2 = S2.gen_noise_instances(circ,5,rgen2)

        self.assertEqual(np.linalg.norm(out1-out2),0.0)

    def test_cirq_channel_conversions_simple(self):

        q0 = cirq.GridQubit(0,0)
        q1 = cirq.GridQubit(0,1)
        q2 = cirq.GridQubit(0,2)
        
        #circ = cirq.Circuit([cirq.CZ(q0,q1), cirq.X.on(q0), cirq.CZ(q0,q1)])
        circ = cirq.Circuit([cirq.CCX(q0,q1,q2),cirq.CZ(q0,q2),cirq.Y.on(q0), cirq.H.on(q2), cirq.CZ(q1,q2), cirq.X.on(q1), cirq.CZ(q0,q1),cirq.H.on(q0)])

        C = cirq_to_total_channel(circ)

        self.assertEqual(np.linalg.norm(C.kraus()[0]-circ.unitary()),0)

        Clist = cirq_moments_to_channel_list(circ)

        circ2 = channel_list_to_circuit(Clist)

        self.assertEqual(np.linalg.norm(circ.unitary()-circ2.unitary()),0)

        Clist2 = [ch.QuantumChannel(cirq.Circuit(m).unitary(),'unitary') for m in circ]
        diffs = [np.linalg.norm(c.liouvillian()-c.liouvillian())==0 for c,cc in zip(Clist,Clist2)]
        self.assertTrue(all(diffs))

        Prop = [Clist[0]]
        for CC in Clist[1:]:
            Prop.append(CC*Prop[-1])

        self.assertTrue(np.linalg.norm(Prop[-1].liouvillian()-C.liouvillian())<1e-14)
        
        diffs = [np.linalg.norm(Prop[i-1].liouvillian()-ch.QuantumChannel(cirq.Circuit(circ[:i]).unitary(),'unitary').liouvillian())<=1e-14 for i in range(1,len(Prop))]
        
        self.assertTrue(all(diffs))

    def test_cirq_channel_conversions_supremacy(self):

        circ = cirq.generate_boixo_2018_supremacy_circuits_v2_grid(2,2,
                                    4,np.random.randint(2**16))
        C = cirq_to_total_channel(circ)

        self.assertEqual(np.linalg.norm(C.kraus()[0]-circ.unitary()),0)

        Clist = cirq_moments_to_channel_list(circ)
        
        circ2 = channel_list_to_circuit(Clist)

        self.assertAlmostEqual(np.linalg.norm(circ.unitary()-circ2.unitary()),0,14)

        Clist2 = [ch.QuantumChannel(cirq.Circuit(m).unitary(),'unitary') for m in circ]
        diffs = [np.linalg.norm(c.liouvillian()-c.liouvillian())==0 for c,cc in zip(Clist,Clist2)]
        self.assertTrue(all(diffs))

        Prop = [Clist[0]]
        for CC in Clist[1:]:
            Prop.append(CC*Prop[-1])

        self.assertLessEqual(np.linalg.norm(Prop[-1].liouvillian()-C.liouvillian()),1e-13)
        
        diffs = [np.linalg.norm(Prop[i-1].liouvillian()-ch.QuantumChannel(cirq.Circuit(circ[:i]).unitary(),'unitary').liouvillian())<=1e-14 for i in range(1,len(Prop))]
        
        self.assertTrue(all(diffs))

    def test_one_qubit_rb_dep_ff(self):

        cliffords = cirq.experiments.qubit_characterizations._single_qubit_cliffords()
        cliffords = cliffords.c1_in_xy
        clifford_mats = np.array([cirq.experiments.qubit_characterizations._gate_seq_to_mats(gates) for gates in cliffords])

        q0 = cirq.GridQubit(0,0)
        circ =cirq.experiments.qubit_characterizations._random_single_q_clifford(
                    q0,20,cliffords,clifford_mats)

        b = np.random.randn(128)
        b = b/np.sum(b**2)*.025

        S = SimpleDephasingSchWARMAFier(b)

        sim = TensorFlowSchWARMASim(circ,S)
        ps = np.real(sim.dm_sim(1000)[0,0])

        pred = compute_Z_dephasing_prob(circ,S)

        self.assertTrue(np.abs(ps-pred)<1e-2)

    def test_single_qubit_control(self):

        q0 = cirq.GridQubit(0,0)
        q1 = cirq.GridQubit(1,0)
        circ = cirq.Circuit([cirq.rx(1).on(q0),cirq.X(q1)**.5,cirq.ry(1).on(q0),cirq.ry(3).on(q1),cirq.X(q0), cirq.Y(q1)**.25])
        S = SingleQubitControlSchWARMAFier([.1],[1],[cirq.ops.common_gates.XPowGate, cirq.ops.pauli_gates._PauliX], cirq_gate_multiplicative)
        noisy_circ, h_array = S.schwarmafy(circ)

        self.assertEqual(len(h_array),6)
        self.assertEqual(len(noisy_circ),5)
        self.assertEqual(len(noisy_circ[-1]),1)
        op = list(noisy_circ[-1])[0]
        self.assertEqual(type(op._gate), cirq.ops.common_gates.XPowGate)
        self.assertTrue(op.gate.exponent.__str__().split('*'), '1.0' )

        S = SingleQubitControlSchWARMAFier([.1],[1],
            [cirq.ops.common_gates.XPowGate, cirq.ops.pauli_gates._PauliX,
            cirq.ops.common_gates.YPowGate, cirq.ops.pauli_gates._PauliY],
            cirq_gate_multiplicative)

        noisy_circ, h_array = S.schwarmafy(circ)

        self.assertEqual(len(h_array),6)
        self.assertEqual(len(noisy_circ),6)
        self.assertEqual(len(noisy_circ[-1]),2)
        
        op = list(noisy_circ[-1])[0]
        self.assertEqual(type(op._gate), cirq.ops.common_gates.XPowGate)
        self.assertEqual(op.gate.exponent.__str__().split('*')[0], '1.0' )
        
        op = list(noisy_circ[-1])[1]
        self.assertEqual(type(op._gate), cirq.ops.common_gates.YPowGate)
        self.assertEqual(op.gate.exponent.__str__().split('*')[0], '0.25' )

        sim = TensorFlowSchWARMASim(circ,S).dm_sim(1)
        sim = TensorFlowSchWARMASim(circ,S).dm_sim(3)

    def test_gate_qubit_dependent_error(self):

        q0 = cirq.GridQubit(0,0)
        q1 = cirq.GridQubit(1,0)
        q2 = cirq.GridQubit(2,0)

        circ = cirq.Circuit([cirq.CX(q0,q1),cirq.CX(q1,q2),cirq.CX(q0,q2),cirq.CZ(q0,q1), cirq.CX(q0,q1)])

        S  = GateQubitDependentSchWARMAFier([.1,],[1,], cirq.ops.common_gates.CXPowGate, [q0,q1], cirq_gate_multiplicative)
        noisy_circ, h_array = S.schwarmafy(circ)

        self.assertEqual(len(noisy_circ),len(circ)+2)
        self.assertEqual(len(h_array),len(circ))
        
        op = list(noisy_circ[1])[0]
        self.assertEqual(type(op._gate),cirq.ops.common_gates.CXPowGate)
        self.assertEqual(op.gate.exponent.__str__(),'1.0*h_0')
        
        op = list(noisy_circ[-1])[0]
        self.assertEqual(type(op._gate),cirq.ops.common_gates.CXPowGate)
        self.assertEqual(op.gate.exponent.__str__(),'1.0*h_4')
        
        sim = TensorFlowSchWARMASim(circ,S).dm_sim(3)
        
        S  = GateQubitDependentSchWARMAFier([.1,],[1,], cirq.ops.common_gates.CXPowGate, [q1,q2], cirq_gate_multiplicative)
        noisy_circ, h_array = S.schwarmafy(circ)

        self.assertEqual(len(noisy_circ),len(circ)+1)
        self.assertEqual(len(h_array),len(circ))
        
        op = list(noisy_circ[2])[0]
        self.assertEqual(type(op._gate),cirq.ops.common_gates.CXPowGate)
        self.assertEqual(op.gate.exponent.__str__(),'1.0*h_1')

        sim = TensorFlowSchWARMASim(circ,S).dm_sim(3)
        
        S  = GateQubitDependentSchWARMAFier([.1,],[1,], cirq.ops.common_gates.CZPowGate, [q0,q1], cirq_gate_multiplicative)
        noisy_circ, h_array = S.schwarmafy(circ)

        self.assertEqual(len(noisy_circ),len(circ)+1)
        self.assertEqual(len(h_array),len(circ))
        
        op = list(noisy_circ[4])[0]
        self.assertEqual(type(op._gate),cirq.ops.common_gates.CZPowGate)
        self.assertEqual(op.gate.exponent.__str__(),'1.0*h_3')

        sim = TensorFlowSchWARMASim(circ,S).dm_sim(3)

    def test_SequentialSchWARMAFier(self):

        q0 = cirq.GridQubit(0,0)
        q1 = cirq.GridQubit(1,0)
        q2 = cirq.GridQubit(2,0)

        circ = [cirq.CX(q0,q1),cirq.CX(q1,q2), cirq.I(q0), cirq.Z(q0)**.5,cirq.Z(q1)**.5,cirq.Z(q2)**.5, cirq.CX(q2,q1), cirq.CX(q1,q0),cirq.I(q2), cirq.Z(q0)**.25,cirq.Z(q1)**.25, cirq.Z(q2)**.25]
        circ = cirq.Circuit(circ)

        S1 = SingleQubitControlSchWARMAFier([.1],[1],[cirq.ops.common_gates.ZPowGate, cirq.ops.pauli_gates._PauliZ],cirq_gate_multiplicative)
        Ss = [GateQubitDependentSchWARMAFier([.1],[1], cirq.ops.common_gates.CXPowGate, qq, error_gen=cirq_gate_multiplicative, sym=sym) for sym, qq in zip(['a','b','c','d'],[(q0,q1),(q1,q2),(q2,q1),(q1,q0)])]

        Stot = SequentialSchWARMAFier([S1]+Ss)

        noisy_circ, h_array = Stot.schwarmafy(circ)

        self.assertEqual(len(noisy_circ),2*len(circ))
        self.assertEqual(len(h_array), (3+4)*len(circ))

        sim = CirqSchWARMASim(circ, Stot).dm_sim(1)
        sim = TensorFlowSchWARMASim(circ, Stot).dm_sim(3)

    def test_SequentialSchWARMAFier(self):

        q0 = cirq.GridQubit(0,0)
        q1 = cirq.GridQubit(1,0)
        q2 = cirq.GridQubit(2,0)

        circ = [cirq.CX(q0,q1),cirq.CX(q1,q2), cirq.I(q0), cirq.Z(q0)**.5,cirq.Z(q1)**.5,cirq.Z(q2)**.5, cirq.CX(q2,q1), cirq.CX(q1,q0),cirq.I(q2), cirq.Z(q0)**.25,cirq.Z(q1)**.25, cirq.Z(q2)**.25]
        circ = cirq.Circuit(circ)

        S1 = SingleQubitControlSchWARMAFier([.1],[1],[cirq.ops.common_gates.ZPowGate, cirq.ops.pauli_gates._PauliZ],cirq_gate_multiplicative)
        Ss = [GateQubitDependentSchWARMAFier([.1],[1], cirq.ops.common_gates.CXPowGate, qq, cirq_gate_multiplicative, sym=sym) for sym, qq in zip(['a','b','c','d'],[(q0,q1),(q1,q2),(q2,q1),(q1,q0)])]

        Stot = SequentialSchWARMAFier([S1]+Ss)

        circ = Stot.gen_noisy_circuit(circ)

        self.assertIsInstance(circ, cirq.Circuit)

        out = cirq.Simulator().simulate(circ).final_state_vector

        circs = Stot.gen_noisy_circuit(circ, 3)

        self.assertIsInstance(circs[0], cirq.Circuit)
        self.assertEqual(len(circs), 3)

        out = [cirq.Simulator().simulate(circ).final_state_vector for circ in circs]

if __name__ == '__main__':
    unittest.main()
