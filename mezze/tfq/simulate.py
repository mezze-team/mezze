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


class SimpleDephasingSchWARMAFier(SchWARMAFier):

    def __init__(self, b, a=[1, ], op=cirq.rz):

        self.b = np.array(b)
        self.a = np.array(a)
        self.op = op

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
        h = sympy.symbols(''.join(['h_{0} '.format(i) for i in range(len(circ.moments) * len(circ.all_qubits()))]),
                          positive=True)
        h_array = np.asarray(h).reshape((len(circ.all_qubits()), len(circ.moments)))

        # Symbolicly SchWARMAfy the input circuit
        noisy_circuit = cirq.Circuit()
        for i, moment in enumerate(circ.moments):
            noisy_circuit.append(moment)
            for j, q in enumerate(circ.all_qubits()):
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

    def schwarmafy(self, circ):
        return self.schwarmafiers[0].schwarmafy(circ)

    def gen_noise_instances(self, circ, num_MC, rgen=np.random):
        vals = [s.gen_noise_instances(circ, num_MC,rgen) for s in self.schwarmafiers]
        return np.sum(vals, 0)


class SchWARMAFiedCircuitSimulator(object):
    """

    Base class for SchWARMAfying and simulating circuits

    """

    def __init__(self, circ, schwarmafier):
        self.circ = circ
        self.schwarmafier = schwarmafier

        self.noisy_circuit, self.syms = self.schwarmafier.schwarmafy(self.circ)


class CirqSchWARMASimulator(SchWARMAFiedCircuitSimulator):
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
        if not pm_obs:
            ops = [.5 + .5 * obs(q) for q in list(self.circ.all_qubits())]
        else:
            ops = [obs(q) for q in list(self.circ.all_qubits())]

        return expectation(self.noisy_circuit, symbol_names=self.syms, symbol_values=vals, operators=ops).numpy()

    def output_expectation(self, num_MC, obs=cirq.Z, pm_obs=False, rgen=np.random):

        out = self.output_sample(num_MC, obs, pm_obs, rgen)
        return np.mean(out, 0)