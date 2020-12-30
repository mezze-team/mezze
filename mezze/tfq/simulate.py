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

import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
import scipy.signal as si
import sympy
import cirq
import mezze.channel as ch

from abc import ABC, abstractmethod


class SchWARMAFier(ABC):
    """
    Abstract Base Class for SchWARMAFiers -- objects that take cirq.Cirquits and
    generate symbolic noisy circuits and concrete noise instances for Monte Carlo
    """

    @abstractmethod
    def schwarmafy(self, circ):
        """
        Args:
            circ: cirq Circuit object to add schwarma noise hooks to

        Returns:
            Circuit with symbolic gates, array of sympy symbols
        """
        return circ, []  # circuit and symbol list

    @abstractmethod
    def gen_noise_instances(self, circ, num_MC):
        """
        Args:
            circ: noiseless cirq Circut
            num_MC: number of MC SchWARMA sequences to generate

        Returns:
            num_MC x #symbolic gates numpy array of concrete values

        """
        return np.zeros((num_MC, 0))

    def gen_noisy_circuit(self, circ, num=1,rgen=np.random):

        noisy_circuit, syms = self.schwarmafy(circ)

        vals = self.gen_noise_instances(circ, num, rgen)
        circuits = []
        for row in vals:
            resolver = cirq.ParamResolver({str(h):row[i] for i,h in enumerate(syms)})
            circuits.append(cirq.resolve_parameters(noisy_circuit,resolver))

        if num == 1:
            return circuits[0]
        else:
            return circuits


class SimpleDephasingSchWARMAFier(SchWARMAFier):
    """
    Class that adds a dephasing error after every gate on every qubit
    """

    def __init__(self, b, a=[1, ], op=cirq.rz, sym='h'):

        self.b = np.array(b)
        self.a = np.array(a)
        self.op = op
        self.sym = 'h'

        # This will pass if the dtype is complex
        assert (np.isreal(self.b).all())
        assert (np.isreal(self.a).all())

        # This makes the dtype real
        self.b = np.real(self.b)
        self.a = np.real(self.a)

    def psd(self, worN=8192, whole=False):
        w, h = si.freqz(self.b, self.a, worN=worN, whole=False)
        return w, np.abs(h) ** 2

    def corrfn(self, worN=8192):
        _, P = self.psd(worN * 2, whole=True)
        R = np.real(np.fft.ifft(P))
        return R[:worN]

    def corr_time(self, thresh=0.01, max_time=8192):
        R = self.corrfn(max_time)
        
        if R[0] == 0:
            return 0

        idx = R / R[0] < thresh
        if not idx.any():
            return max_time
        else:
            return int(np.min(np.arange(len(R))[np.abs(R) / R[0] < thresh]))

    def schwarmafy(self, circ):

        # Create array of symbolic variables and reshape to natural circuit parameterization
        h = sympy.symbols(''.join(['{0}_{1} '.format(self.sym, i) for i in range(len(circ.moments) * len(circ.all_qubits()))]),
                          positive=True)
        h_array = np.asarray(h).reshape((len(circ.all_qubits()), len(circ.moments)))

        # Symbolicly SchWARMAfy the input circuit
        noisy_circuit = cirq.Circuit()
        ordered_qubits = cirq.ops.QubitOrder.as_qubit_order(cirq.ops.QubitOrder.DEFAULT).order_for(circ.all_qubits())
        for i, moment in enumerate(circ.moments):
            noisy_circuit.append(moment)
            for j, q in enumerate(ordered_qubits):
                noisy_circuit.append(self.op(h_array[j, i]).on(q))

        return noisy_circuit, h

    def gen_noise_instances(self, circ, num_MC,rgen=np.random):
        num_qubits = len(circ.all_qubits())
        num_moments = len(circ.moments)

        delay = np.maximum(len(self.b), self.corr_time())

        # white = np.random.randn(num_qubits,num_moments+delay)
        vals = [np.reshape(si.lfilter(self.b, self.a, rgen.randn(num_qubits, num_moments + delay))[:, delay:],
                           (1, num_qubits * num_moments)) for _ in range(num_MC)]

        if num_MC > 1:
            vals = np.squeeze(vals)
        else:
            return np.array([np.squeeze(vals)])

        return vals

def cirq_gate_multiplicative(orig_op, h):
    """
    Error generatation function for multiplicative gate errors
    A similar function will need to be custom implemented fo non-standard gates
    """  
        
    return (type(orig_op.gate)()**(orig_op.gate._exponent*h)).on(*orig_op._qubits)


class SingleQubitControlSchWARMAFier(SimpleDephasingSchWARMAFier):
    
    def __init__(self, b, a, op_list, error_gen=cirq_gate_multiplicative, sym = 'h'):
        
        self.b = b
        self.a = a
        self.op_list = op_list
        self.error_gen=error_gen
        self.sym=sym
        
    def schwarmafy(self,circ):
        h = sympy.symbols(''.join(['{0}_{1} '.format(self.sym, i) for i in range(len(circ.moments) * len(circ.all_qubits()))]),
                  positive=True)
        h_array = np.asarray(h).reshape((len(circ.all_qubits()), len(circ.moments)))

        noisy_circ = cirq.Circuit([])
        ordered_qubits = cirq.ops.QubitOrder.as_qubit_order(cirq.ops.QubitOrder.DEFAULT).order_for(circ.all_qubits())
        
        for i, mom in enumerate(circ):

            noisy_circ.append(mom)
            ops = list(mom)
            for op in ops:
                if type(op._gate) in self.op_list:
                    qnum = np.argmax([op.qubits[0] == q for q in ordered_qubits])
                    noisy_circ.append(self.error_gen(op, h_array[qnum,i]))
                    
        return noisy_circ, h


class NullSchWARMAFier(SimpleDephasingSchWARMAFier):
    """
    A SchWARMAFier that does nothing, useful for noiseless simulations
    """
    def __init__(self):
        self.b = np.array([0,])
        self.a = np.array([1,])
        
    def schwarmafy(self, circ):
        return circ, []

    def gen_noise_instances(self, circ, num_MC, rgen=np.random):
        return np.zeros((num_MC,0))


class GateQubitDependentSchWARMAFier(SimpleDephasingSchWARMAFier):
    
    def __init__(self, b, a, op, qubits, error_gen=cirq_gate_multiplicative, sym = 'h'):
        """
        b: list-like of floats containing MA coefficients for use with scipy.signal.lfilter
        a: list-like of floats containing AR coefficients for use with scipy.signal.lfilter
        op: circ.Gate defining the gate the error acts on
        qubits: list of circ.GridQubits that error acts on
        error_gen: function that takes a gate of thesame type as op, 
                a symbolic variable, and the qubit list the gate acts on 
                and returns a symbolic, imperfect gate
        """
        
        self.b = b
        self.a = a
        self.op = op
        self.qubits = qubits
        self.error_gen = error_gen
        self.sym = sym
        
    def schwarmafy(self,circ):

        h = sympy.symbols(''.join(['{0}_{1} '.format(self.sym, i) for i in range(len(circ.moments))]))
        h_array = np.asarray(h)#.reshape((len(circ.all_qubits()), len(circ.moments)))

        if len(h_array.shape)==0:
            h_array = np.asarray([h])

        noisy_circ = cirq.Circuit([])
        for i, mom in enumerate(circ):

            noisy_circ.append(mom)
            ops = list(mom)
            for op in ops:
                if type(op._gate) == self.op:
                    if all([q1==q2 for q1,q2 in zip(op._qubits,self.qubits)]):

                        noisy_circ.append(self.error_gen(op,h_array[i]))#,self.qubits))
            
        return noisy_circ, h
        
    def gen_noise_instances(self, circ, num_MC, rgen=np.random):
        #num_qubits = len(circ.all_qubits())
        num_moments = len(circ.moments)

        delay = np.maximum(len(self.b), self.corr_time())
        
        vals = [np.reshape(si.lfilter(self.b, self.a, rgen.randn(num_moments + delay))[delay:],
                           (1, num_moments)) for _ in range(num_MC)]

        if num_MC > 1:
            vals = np.squeeze(vals)
        else:
            return np.array([np.squeeze(vals)])

        return vals



class AdditiveSchWARMAFier(SchWARMAFier):
    """

    A SchWARMAFier that combines mulitple SchWARMAFiers whose symbolic arguments can be added together

    Useful for simulating background and injected noises

    """

    def __init__(self, schwarmafiers):
        """

        schwarmafiers: list of SchWARMAFiers whose noise instances can be added together (i.e., use the same op)

        """
        self.schwarmafiers = schwarmafiers

    def psd(self, worN=8192, whole=False):
        w, P = self.schwarmafiers[0].psd(worN, whole)
        for S in self.schwarmafiers[1:]:
            P += S.psd(worN, whole)[1]
        return w,P

    def schwarmafy(self, circ):
        return self.schwarmafiers[0].schwarmafy(circ)

    def gen_noise_instances(self, circ, num_MC, rgen=np.random):
        vals = [s.gen_noise_instances(circ, num_MC,rgen) for s in self.schwarmafiers]
        return np.sum(vals, 0)


class SequentialSchWARMAFier(SchWARMAFier):
    """
    A SchWARMAFier that combines multiple schwarmafiers by applying them sequentially to the base circuit
    """

    def __init__(self, schwarmafiers):
        self.schwarmafiers = schwarmafiers

        syms = [S.sym for S in self.schwarmafiers]

        assert len(self.schwarmafiers)==len(syms), "SchWARMAFier.sym values not unique"

    def schwarmafy(self, circ):
        
        indiv_noisy, h_array = zip(*[S.schwarmafy(circ) for S in self.schwarmafiers])
        h_array = np.concatenate(h_array)

        noisy_circ = cirq.Circuit([])
        
        # Going through moment by moment rather than directly sequentially schwarmafying
        # so we don't add noise to noise
        for i,mom in enumerate(circ):
            noisy_circ.append(mom)
            
            for j, S in enumerate(self.schwarmafiers):
            
                circ_part = circ[:i+1]
                n_part, _ = S.schwarmafy(circ_part)
            
                if mom != n_part[-1]:
                    noisy_circ.append(indiv_noisy[j][len(n_part)-1])
        
        return noisy_circ, list(h_array)

    def gen_noise_instances(self, circ, num_MC, rgen=np.random):

        noises = [S.gen_noise_instances(circ,num_MC) for S in self.schwarmafiers]
        return np.concatenate(noises,1)

class MatrixCorrelatorSchWARMAFier(SchWARMAFier):
    """
    A SchWARMAFier that takes a SchWARMAFiers and multiplies the noise instances by a matrix to correlate them
    """
    
    def __init__(self, schwarmafier, A):
        self.schwarmafier = schwarmafier
        self.A = A
        try:
            self.sym = self.schwarmafier.sym
        except AttributeError:
            pass

    def schwarmafy(self, circ):

        return self.schwarmafier.schwarmafy(circ)

    def gen_noise_instances(self, circ, num_MC, rgen=np.random):

        #num_qubits = len(circ.all_qubits())
        num_moments = len(circ)
        
        noises = self.schwarmafier.gen_noise_instances(circ, num_MC, rgen)
        corr_noises = np.zeros_like(noises)
        
        for i, noise in enumerate(noises):
            corr_noise = self.A@np.reshape(noise[np.newaxis,:], (noise.shape[0]//num_moments,num_moments))
            corr_noises[i,:]=np.reshape(corr_noise,(1, noise.shape[0]))
        return corr_noises

        
    

class SchWARMAFiedCircuitSimulator(object):
    """

    Base class for SchWARMAfying and simulating circuits

    """

    def __init__(self, circ, schwarmafier):
        self.circ = circ
        self.schwarmafier = schwarmafier

        self.noisy_circuit, self.syms = self.schwarmafier.schwarmafy(self.circ)


class CirqSchWARMASim(SchWARMAFiedCircuitSimulator):
    """

    Simulator that uses Cirq backend

    """

    def state_sim(self, num_MC, rgen=np.random):
        vals = self.schwarmafier.gen_noise_instances(self.circ, num_MC, rgen)
        states = []
        sim = cirq.Simulator()
        for row in vals:
            resolver = cirq.ParamResolver({str(h):row[i] for i,h in enumerate(self.syms)})
            states.append(sim.simulate(self.noisy_circuit, resolver).final_state_vector)
        
        return np.array(states)

    def dm_sim(self, num_MC, rgen=np.random):
        vals = self.schwarmafier.gen_noise_instances(self.circ, num_MC, rgen)
        dms = []
        sim = cirq.DensityMatrixSimulator()
        for row in vals:
            resolver = cirq.ParamResolver({str(h):row[i] for i,h in enumerate(self.syms)})
            dms.append(sim.simulate(self.noisy_circuit, resolver).final_density_matrix)

        return np.mean(dms,0)

    def unitary_props_sim(self, num_MC, as_channel = True, rgen=np.random):
        vals = self.schwarmafier.gen_noise_instances(self.circ, num_MC, rgen)
        unitary_props = []
        for row in vals:
            resolver = cirq.ParamResolver({str(h):row[i] for i,h in enumerate(self.syms)})
            prop = cirq.resolve_parameters(self.noisy_circuit,resolver)._unitary_()
            if as_channel:
                prop = ch.QuantumChannel(prop, 'unitary')

            unitary_props.append(prop)
        
        return unitary_props
        


class TensorFlowSchWARMASim(SchWARMAFiedCircuitSimulator):
    """

    Simulator that uses TensorFlow Quantum

    """

    def state_sim(self, num_MC,rgen=np.random):
        vals = tf.Variable(self.schwarmafier.gen_noise_instances(self.circ, num_MC,rgen))
        state = tfq.layers.State()
        return state(self.noisy_circuit, symbol_names=self.syms, symbol_values=vals).to_tensor().numpy()

    def dm_sim(self, num_MC,rgen=np.random):
        out = self.state_sim(num_MC,rgen)
        return tf.tensordot(tf.transpose(out), tf.math.conj(out), axes=[[1], [0]]).numpy() / num_MC

    def output_sample(self, num_MC, obs=cirq.Z, pm_obs=False, rgen=np.random):

        vals = tf.Variable(self.schwarmafier.gen_noise_instances(self.circ, num_MC,rgen))
        expectation = tfq.layers.Expectation()

        ordered_qubits = cirq.ops.QubitOrder.as_qubit_order(cirq.ops.QubitOrder.DEFAULT).order_for(circ.all_qubits())
        
        if not pm_obs:
            ops = [.5 + .5 * obs(q) for q in ordered_qubits]
        else:
            ops = [obs(q) for q in ordered_qubits]

        return expectation(self.noisy_circuit, symbol_names=self.syms, symbol_values=vals, operators=ops).numpy()

    def output_expectation(self, num_MC, obs=cirq.Z, pm_obs=False, rgen=np.random):

        out = self.output_sample(num_MC, obs, pm_obs, rgen)
        return np.mean(out, 0)