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

import abc
import numpy as np
import logging
import copy
from .noise import *

class HamiltonianIteratorInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_num_qubits(self):
        """Returns the number of qubits"""
        return

    @abc.abstractmethod
    def instantiate(self):
        """Resets the iterator and instantiates a realization of any noise for a monte carlo run"""
        return

    @abc.abstractmethod
    def next(self):
        """Returns the next hamiltonian matrix as 2D numpy array"""
        return

class PrecomputedHamiltonian(HamiltonianIteratorInterface):
    """
    Provides an iterator interface to precomputed hamiltonian matrices
    """
    def __init__(self, pmd, controls, config, hamiltonian_function):
        self.pmd = pmd
        self.controls = controls
        self.config = config
        self.hamiltonian_function = hamiltonian_function
        self.index = -1
        self.hamiltonian_matrices = None

        noise_factory = NoiseFactory(config.time_length, config.num_steps)
        self.noise_sources = [noise_factory.get(x) for x in pmd.noise_hints]

    def get_num_qubits(self):
        return self.pmd.num_qubits

    def get_qubit_dim(self):
        return np.array(self.pmd.basis[0]).shape[0]

    def instantiate(self):
        self.index = -1
        noise_values = np.transpose(np.array([x.generate() for x in self.noise_sources]))
        self.hamiltonian_matrices = self.hamiltonian_function(self.controls, noise_values)

    def __iter__(self):
        self.index = -1
        return self

    def next(self):
        self.index += 1
        return self.hamiltonian_matrices[self.index]

    @classmethod
    def create(cls, pmd, controls, config):
        if isinstance(controls, list):
            controls = np.array(controls)
        controls = np.transpose(controls)
        return PrecomputedHamiltonian(pmd, controls, config, HamiltonianFunction(pmd).get_function())

class HamiltonianFunction(object):
    def __init__(self, pmd):
        self.logger = logging.getLogger(__name__)
        self.pmd = pmd
        self.precomputed = False

    def __getattr__(self, item):
        if hasattr(self.pmd, item):
            return getattr(self.pmd, item)
        else:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, item))

    def get_function(self):
        """
        Get a closure for passing to PrecomputedHamiltonian
        """
        return lambda c, g: self.eval_n_qubit_ham_fast(c, g)

    def eval_n_qubit_ham_fast(self, c, gamma):
        """
        Makes use of repeated calls to ``eval_1_qubit_ham_fast`` and
        ``eval_2_qubit_only_fast`` to evaluate an ``num_qubit`` hamiltonian
        based on one and two qubit operations.  This function works on
        one and two qubit PMDs, as well as higher order systems.

        Currently, this is the only Hamiltonian generation method that computes
        cross talk between the input (control/noise) signals and incorporates
        these into the Hamiltonian.

        :param c: ``n_steps`` by ``num_qubits*control_dim+num_qubits*(num_qubits-1)/2*two_control_dim`` ``numpy.array`` where each column
        represents the control pulses.  The enumeration of the control pulses is to cycle through the single qubit controls for each qubit
        first, followed the two qubit controls for the first qubit on the second, then first on third, and so on.
        :type c: ``numpy.array``
        :param gamma: ``n_steps`` by ``num_qubits*noise_dim+num_qubits*(num_qubits-1)/2*two_noise_dim`` ``numpy.array`` where each column
        represents the noise pulses.  The enumeration mirrors that of ``c``, above.
        :type gamma: ``numpy.array``

        :returns: ``list`` of ``list`` H -- a sequence of one qubit Hamiltonians of length ``c.shape[0]``

        """

        xtalk = False

        if c.shape[0] == 0:
            c = np.zeros((gamma.shape[0], 0))
        if gamma.shape[0] == 0:
            gamma = np.zeros((c.shape[0], 0))

        if np.any(self.xtalk):
            xtalk = True

            cTemp = copy.copy(c)
            gammaTemp = copy.copy(gamma)
            for i in range(0, c.shape[1]):
                idx = self.xtalk[i, 0:c.shape[1]] != 0
                cTemp[:, idx] += (self.xtalk[i, 0:c.shape[1]][idx][:, np.newaxis] * c[:, i]).T
                idx = self.xtalk[i, c.shape[1]:] != 0
                if len(idx) > 0:
                    gammaTemp[:, idx] += (self.xtalk[i, c.shape[1]:][idx][:, np.newaxis] * c[:, i]).T

            for i in range(0, gamma.shape[1]):
                idx = self.xtalk[c.shape[1] + i, 0:c.shape[1]] != 0
                if len(idx) > 0:
                    cTemp[:, idx] += (self.xtalk[c.shape[1] + i, 0:c.shape[1]][idx][:, np.newaxis] * gamma[:, i]).T
                idx = self.xtalk[c.shape[1] + i, c.shape[1]:] != 0
                gammaTemp[:, idx] += (self.xtalk[c.shape[1] + i, c.shape[1]:][idx][:, np.newaxis] * gamma[:, i]).T

            c = cTemp
            gamma = gammaTemp

        # Split c and gamma into one and two qubit controls
        # Some where up here handle crosstalk, too
        Hs = []
        ctrl_ctr = 0
        noise_ctr = 0
        np_basis = [np.array(x, dtype=complex) for x in self.basis]
        dim = np_basis[0].shape[0]
        for lcv in range(self.num_qubits):
            left = np.eye(dim ** (lcv))
            right = np.eye(dim ** (self.num_qubits - lcv - 1))
            H1 = self.eval_1_qubit_ham_fast(c[:, ctrl_ctr:ctrl_ctr + self.control_dim],
                                            gamma[:, noise_ctr:noise_ctr + self.noise_dim])
            Hs.append([np.kron(left, np.kron(h, right)) for h in H1])

            ctrl_ctr += self.control_dim
            noise_ctr += self.noise_dim

        for lcv in range(self.num_qubits):
            for lcv2 in range(lcv + 1, self.num_qubits):
                left = np.eye(dim ** (lcv))
                middle = np.eye(dim ** (lcv2 - lcv - 1))
                right = np.eye(dim ** (self.num_qubits - lcv2 - 1))
                H2 = self.eval_2_qubit_only_fast(c[:, ctrl_ctr:ctrl_ctr + self.two_control_dim],
                                                 gamma[:, noise_ctr:noise_ctr + self.two_noise_dim], left, middle, right)
                Hs.append(H2)

                ctrl_ctr += self.two_control_dim
                noise_ctr += self.two_noise_dim

        # At this point Hs and contain all "kronned" up hamiltonians, only needs to be summed
        H = []
        for t in range(len(Hs[0])):
            Ht = np.zeros((dim ** self.num_qubits, dim ** self.num_qubits), dtype=complex)
            for i in range(len(Hs)):
                Ht += Hs[i][t]

            H.append(Ht)

        return H

    def _precompute(self):
        """

        Performs some computations that are redundant across evaluations
        for the purposes of increasing computation speed

        """

        self.precomputed = True

        self.np_basis = [np.array(x, dtype=complex) for x in self.basis]
        self.single_qubit_drift_mask = [True for x in self.basis]
        self.single_qubit_noise_mask = [True for x in self.basis]
        self.single_qubit_control_mask = [[True for x in self.basis] for c in range(self.control_dim)]

        for i in range(len(self.basis)):
            self.single_qubit_drift_mask[i] = self.single_qubit_drift[i](np.zeros(self.control_dim),
                                                                         np.zeros(self.noise_dim)) is not None
            self.single_qubit_noise_mask[i] = self.single_qubit_noise[i](np.zeros(self.control_dim),
                                                                         np.zeros(self.noise_dim)) is not None

            for j in range(self.control_dim):
                self.single_qubit_control_mask[j][i] = self.single_qubit_control[j][i](np.zeros(self.control_dim),
                                                                                       np.zeros(
                                                                                           self.noise_dim)) is not None

        self.any_drift = any(self.single_qubit_drift_mask)
        self.any_noise = any(self.single_qubit_noise_mask)

        if self.num_qubits >= 2:
            self.two_qubit_control_mask = [[[True for x in self.basis] for y in self.basis] for c in
                                           range(self.two_control_dim)]
            for i in range(self.two_control_dim):
                for j in range(len(self.basis)):
                    for k in range(len(self.basis)):
                        self.two_qubit_control_mask[i][j][k] = self.two_qubit_control[i][j][k](
                            np.zeros(self.two_control_dim),
                            np.zeros(self.two_noise_dim)) is not None

            self.np_basis2 = [[np.kron(self.np_basis[x], self.np_basis[y]) for y in range(len(self.basis))] for x in
                              range(len(self.basis))]

    def eval_1_qubit_ham_fast(self, c, gamma):
        """
        Evaluates a single qubit Hamiltonian over time given a lists of control
        pulses and noise  by iterating and summing over
        ``single_qubit_control``, ``single_qubit_noise``, and
        ``single_qubit_drift`` .  It does not apply the filter specified by
        ``single_qubit_filter_num_coeffs`` and
        ``single_qubit_filter_den_coeffs`` nor does add it in Lindblads.
        Performs the same Hamiltonian computation as ``eval_1_qubit_ham``, but
        in a more optimized manner and does not compute the constraints.

        :param c: a list whose elements are lists corresponding to single qubit controls at a given time
        :type c: list of list
        :param gamma: a list whose elements are lists corresponding to noise instances at a given time, with spectral densities specified by ``S``
        :type gamma: list of list
        :returns: ``list`` of ``list`` H -- a sequence of one qubit Hamiltonians of length ``len(c)``


        Mathematically, denote ``single_qubit_control[i][j]``,
        ``single_qubit_drift[j]``, ``single_qubit_noise[j]``, ``basis[j]``, ``c[t]``
        ``Gamma[t]``, and ``control_dim`` by

        .. math::
            f_{\gamma_{i,j}}, f_{\delta_j}, f_{\\nu_j}, \sigma_j,
            \\vec{c}_t, \\vec{\Gamma}_t, N,  \\text{ respectively}.

        Then for each ``t`` in ``range(len(c))`` this function computes a
        Hamiltonian

        .. math::
            \mathcal{H}(t) =
            \\underbrace{\sum_{i=0}^{N-1}\sum_{j=0}^{3} f_{\gamma_{i,j}}(\\vec{c}_t,\\vec{\Gamma}_t)\sigma_j}_{\\text{control}}
            +\\underbrace{\sum_{j=0}^3 f_{\delta_j}(\\vec{c}_t,\\vec{\Gamma}_t)\sigma_j}_{\\text{drift}}
            +\\underbrace{\sum_{j=0}^3 f_{\\nu_j}(\\vec{c}_t,\\vec{\Gamma}_t)\sigma_j}_{\\text{noise}}

        """

        if not self.precomputed:
            self._precompute()

        H = []
        for t in range(len(c)):
            Ht = 0.0 * self.np_basis[0]
            if self.any_drift:
                for alpha, sigma_alpha in enumerate(self.np_basis):
                    if self.single_qubit_drift_mask[alpha]:
                        Ht += self.single_qubit_drift[alpha](c[t], gamma[t]) * sigma_alpha

            if self.any_noise:
                for alpha, sigma_alpha in enumerate(self.np_basis):
                    if self.single_qubit_noise_mask[alpha]:
                        Ht += self.single_qubit_noise[alpha](c[t], gamma[t]) * sigma_alpha

            for cont in range(self.control_dim):
                for alpha, sigma_alpha in enumerate(self.np_basis):
                    if self.single_qubit_control_mask[cont][alpha]:
                        Ht += self.single_qubit_control[cont][alpha](c[t], gamma[t]) * sigma_alpha

            H.append(Ht)

        return H

    def eval_2_qubit_only_fast(self, cc, gamma, left=1, middle=1, right=1):
        """
        Evaluates a two qubit Hamiltonian over time given a lists of control
        pulses  and noise by iterating and summing over
        the ``single_qubit_`` and ``two_qubit_`` ``control``, ``drift``, and
        ``noise`` functions.  It does not apply either the single qubit or
        two qubit filter functions, nor does add it in Lindblads.
        Performs the same Hamiltonian computation as ``eval_2_qubit_ham``, but
        in a more optimized manner and does not compute the constraints.

        :param c1: a list whose elements are lists corresponding to single qubit 1 controls at a given time
        :type c1: list of list
        :param c2:  a list whose elements are lists corresponding to single qubit 2 controls at a given time
        :type c2: list of list
        :param cc: a list whose elements are lists corresponding to two qubit controls at a given time
        :type cc: list of list
        :param Gamma: a list whose elements are lists corresponding to noise instances at a given time, with spectral densities specified by ``S``
        :type Gamma: list of list
        :returns: ``list`` of ``list`` H -- a sequence of one qubit Hamiltonians of length ``len(c)``


        Mathematically, denote ``two_qubit_control[i][j][k]``,
        ``two_qubit_drift[j][k]``, ``two_qubit_noise[j][k]``, ``basis[j]``, ``cc[t]``,
        ``Gamma[t][2*noise_dim-1:]``, and ``two_control_dim`` by

        .. math::
            g_{\gamma_{i,j,k}}, g_{\delta_{j,k}}, g_{\\nu_{j,k}}, \sigma_j,
            \\vec{c}_t, \\vec{\Gamma}_t, M,  \\text{ respectively}.

        Then for each ``t`` in ``range(len(c))`` this function computes a
        Hamiltonian

        .. math::
            \mathcal{H}(t) =
            \\underbrace{\sum_{i=0}^{M-1}\sum_{j=0}^{3}\sum_{k=0}^3
            g_{\gamma_{i,j,k}}(\\vec{c}_t,\\vec{\Gamma}_t)\sigma_j\otimes\sigma_k}_{\\text{control}}
            +\\underbrace{\sum_{j=0}^3\sum_{k=0}^3 g_{\delta_{j,k}}(\\vec{c}_t,\\vec{\Gamma}_t)\sigma_j\otimes\sigma_k}_{\\text{drift}}

            +\\underbrace{\sum_{j=0}^3\sum_{k=0}^3 g_{\\nu_{j,j}}(\\vec{c}_t,\\vec{\Gamma}_t)\sigma_j\otimes\sigma_k}_{\\text{noise}}
            +\\underbrace{\mathcal{H}_1(t)\otimes I_2+I_2\otimes\mathcal{H}_2(t)}_{\\text{individual qubits}}

        where

        .. math::
            \mathcal{H}_1, \mathcal{H}_2, \\text{ and }I_2

        denote ``eval_1_qubit_ham(c1[t],Gamma[t][0:noise_dim-1])``, ``eval_1_qubit_ham(c2[t],Gamma[t][noise_dim:2*noise_dim-1])``,
        and the 2 dimensional identity matrix, respectively.

        """

        #ToDo update this function documentation

        H = []
        np_basis2 = [
            [np.kron(left, np.kron(self.np_basis[x], np.kron(middle, np.kron(self.np_basis[y], right)))) for y in
             range(len(self.basis))] for x in range(len(self.basis))]
        dim = self.np_basis[0].shape[0]
        for t in range(len(cc)):
            Ht = np.zeros((dim ** self.num_qubits, dim ** self.num_qubits), dtype=complex)
            for cont in range(self.two_control_dim):
                for x in range(len(self.np_basis)):
                    for y in range(len(self.np_basis)):
                        if self.two_qubit_control_mask[cont][x][y]:
                            Ht += self.two_qubit_control[cont][x][y](cc[t, :], gamma[t, :]) * np_basis2[x][y]
            H.append(Ht)

        return H

