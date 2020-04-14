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

import numpy as np
import scipy
import logging
import copy
import multiprocessing
from .lindblad import LindbladFactory
from .channel import *
import scipy.constants

try:
    xrange
except NameError:
    xrange = range


class SimulationConfig(object):
    """
    Configuration object for the simulation
    """
    def __init__(self, time_length=1, num_steps=1000):
        self.set_time(time_length, num_steps)

        self.num_runs = 1
        self.parallel = False
        self.num_cpus = multiprocessing.cpu_count()
        self.run_to_convergence = False
        self.convergence_tolerance = 1.e-6
        self.num_runs_below_tolerance = 1

        self.realization_sampling = False
        self.time_sampling = False
        self.sampling_interval = None

        self.hbar = 1
        
        self.get_unitary = False # NEW

    def set_time(self, time_length, num_steps):
        self.time_length = float(time_length)
        self.num_steps = num_steps
        self.dt = float(self.time_length) / float(self.num_steps)

    def sample_time(self, interval=1):
        """Sample each realization of the superoperator over time at a specified interval"""
        self.time_sampling = True
        self.sampling_interval = interval

    def sample_sup(self):
        """Store the Liouvillian superoperator for all iterations"""
        self.realization_sampling = True

    def use_convergence(self, tolerance=1.e-6, runs_below_tolerance=1):
        self.run_to_convergence = True
        self.convergence_tolerance = tolerance
        self.num_runs_below_tolerance = runs_below_tolerance

    def use_Js(self):
        # Joule seconds
        self.hbar = scipy.constants.hbar # 1.05457148e-34

    def use_eVs(self):
        # eV seconds
        self.hbar = scipy.constants.hbar / scipy.constants.electron_volt # 6.58211814e-16

    def validate(self):
        return True

class SimulationReport(object):
    def __init__(self):
        self.liouvillian = None
        self.liouvillian_samples = None
        self.time_samples = None
        self.num_runs = 0

class RunReport(object):
    def __init__(self, sup, ts):
        self.liouvillian = sup
        self.time_samples = ts

class ChannelReport(object):
    def __init__(self, report):
        self.channel = QuantumChannel(report.liouvillian,'liou')
        
        if report.liouvillian_samples is not None:
            self.liouvillian_samples = [QuantumChannel(liou,'liou') for liou in report.liouvillian_samples]
        else:
            self.liouvillian_samples = None

        if report.time_samples is not None:

            if report.liouvillian_samples is None:

                if report.time_samples[0][0].shape != report.liouvillian.shape:

                    self.time_samples = [QuantumChannel(np.mean([np.kron(U.conj(),U) for U in row],0),'liou') for row in zip(*report.time_samples)]
                
                else:

                    self.time_samples = [QuantumChannel(np.mean(row,0),'liou') for row in zip(*report.time_samples)]
            
            else:

                if report.time_samples[0][0].shape != report.liouvillian.shape:
                    #unitary
                    self.time_samples=[[QuantumChannel(U,'unitary') for U in row] for row in report.time_samples]

                else:
                    self.time_samples=[[QuantumChannel(liou,'liou') for liou in row] for row in report.time_samples]
        else:
            self.time_samples = None

        self.num_runs = report.num_runs

class Simulation(object):
    """
    Monte Carlo simulation

    :param config: SimulationConfig object
    :param pmd: PMD object. Holds all parameters for the physical device
    :param hamiltonian: HamiltonianIteratorInterface object
    :param lindblad: Lindblad operator. If None, it will generate the lindblads from
                     the PMD if applicable
    """
    def __init__(self, config, pmd, hamiltonian, lindblad=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pmd = pmd
        self.lindblad = lindblad
        self.hamiltonian = hamiltonian

        if lindblad is None and pmd.num_lindblad > 0:
            self.lindblad = LindbladFactory().get(self.pmd).get_matrix()

        if not self.config.validate():
            raise RuntimeError

    def run(self):
        """
        Run the simulation
        :return: SimulationReport
        """
        report = SimulationReport()

        if self.config.realization_sampling:
            report.liouvillian_samples = []
        if self.config.time_sampling:
            report.time_samples = []

        if self.config.run_to_convergence:
            report = self._converging_run(report)
        elif self.config.parallel:
            report = self._parallel_run(report)
        else:
            report = self._local_run(report)

        return ChannelReport(report)

    def _run_once(self):
        """Run a single realization of the monte carlo simulation"""
        self.hamiltonian.instantiate()
        if self.lindblad is not None:
            propagator = LindbladPropagator(self.config, self.hamiltonian, self.lindblad)
        else:
            propagator = NonLindbladPropagator(self.config, self.hamiltonian)
        if self.config.time_sampling:
            propagator.set_sampling_interval(self.config.sampling_interval)
        superoperator = propagator.propagate()
        return RunReport(superoperator, propagator.get_samples())

    def _local_run(self, report):
        """Single threaded run"""
        if self.config.get_unitary: # NEw
            superoperator_size = self.hamiltonian.get_qubit_dim() ** self.hamiltonian.get_num_qubits()
        else:
            superoperator_size = (self.hamiltonian.get_qubit_dim()**2) ** self.hamiltonian.get_num_qubits()
        superoperator_sum = np.zeros((superoperator_size, superoperator_size), dtype=np.complex128)
        for count in xrange(self.config.num_runs):
            run_report = self._run_once()
            report = self._update_report(report, run_report)
            superoperator_sum += run_report.liouvillian
        report.liouvillian = superoperator_sum / self.config.num_runs

        return report

    def _converging_run(self, report):
        """Run until convergence"""
        run_report = self._run_once()
        report = self._update_report(report, run_report)
        running_mean = run_report.liouvillian
        times_below_tolerance = 0
        num_iterations = 1
        while times_below_tolerance < self.config.num_runs_below_tolerance:
            if self.config.parallel:
                run_report = self._parallel_run(report)
            else:
                run_report = self._run_once()
            report = self._update_report(report, run_report)
            current = run_report.liouvillian
            new_mean = (num_iterations * running_mean + current) / (num_iterations + 1)
            num_iterations += 1
            error = abs(running_mean - new_mean).max()
            if error < self.config.convergence_tolerance:
                times_below_tolerance += 1
            else:
                times_below_tolerance = 0
            running_mean = new_mean

        report.liouvillian = running_mean
        return report

    def _parallel_run(self, report):
        """Parallel run using multiprocessing"""
        if self.config.get_unitary: # NEw
            superoperator_size = self.hamiltonian.get_qubit_dim() ** self.hamiltonian.get_num_qubits()
        else:
            superoperator_size = (self.hamiltonian.get_qubit_dim()**2) ** self.hamiltonian.get_num_qubits()
        superoperator_sum = np.zeros((superoperator_size, superoperator_size), dtype=np.complex128)

        num_procs = min(self.config.num_runs, self.config.num_cpus)
        q_in = multiprocessing.Queue(1)
        q_out = multiprocessing.Queue()

        # kick off the child processes
        processes = [multiprocessing.Process(target=self._parallel_thread, args=(self._run_once, q_in, q_out)) for _ in range(num_procs)]
        for p in processes:
            # in case main process dies, kill child processes
            p.daemon = True
            p.start()

        # keep the processes busy with new work
        num_results = 0
        [q_in.put(x) for x in xrange(num_procs)]
        num_queued = num_procs
        while num_results < self.config.num_runs:
            run_report = q_out.get()
            superoperator_sum += run_report.liouvillian
            report = self._update_report(report, run_report)
            num_results += 1
            if num_queued < self.config.num_runs:
                q_in.put(num_queued)
                num_queued += 1

        # shut down processes
        [q_in.put(None) for _ in xrange(num_procs)]
        [p.join() for p in processes]

        report.liouvillian = superoperator_sum / self.config.num_runs
        return report

    @classmethod
    def _parallel_thread(cls, func, q_in, q_out):
        while True:
            index = q_in.get()
            if index is None:
                break
            q_out.put(func())

    def _update_report(self, report, run_report):
        if self.config.realization_sampling:
            report.liouvillian_samples.append(run_report.liouvillian)
        if self.config.time_sampling:
            report.time_samples.append(run_report.time_samples)
        report.num_runs += 1
        return report

class Propagator(object):
    def __init__(self, config, hamiltonian):
        self.hamiltonian = hamiltonian
        self.dt = config.dt
        self.num_steps = config.num_steps
        self.num_qubits = hamiltonian.get_num_qubits()
        self.qubit_dim = hamiltonian.get_qubit_dim()
        self.hbar = config.hbar
        self.sample_interval = None
        self.samples = []
        self.get_unitary = config.get_unitary # NEW

    def set_sampling_interval(self, interval):
        self.sample_interval = interval

    def get_samples(self):
        return self.samples

class NonLindbladPropagator(Propagator):
    def __init__(self, config, hamiltonian):
        Propagator.__init__(self, config, hamiltonian)

    def _next(self):
        return np.mat(scipy.linalg.expm(-1.j * self.hamiltonian.next() * self.dt / self.hbar))

    def propagate(self):
        unitary_matrix = self._get_initial_state()
        for index in xrange(self.num_steps):
            unitary_matrix = np.dot(self._next(), unitary_matrix)
            if self.sample_interval and (index+1) % self.sample_interval == 0:
                self.samples.append(copy.copy(unitary_matrix))
        if self.get_unitary: # NEW
            superoperator_matrix = unitary_matrix
        else:
            superoperator_matrix = scipy.kron(unitary_matrix.conjugate(), unitary_matrix)
        return superoperator_matrix

    def _get_initial_state(self):
        return np.mat(np.eye(self.qubit_dim ** self.num_qubits, dtype=np.complex128))

class LindbladPropagator(Propagator):
    def __init__(self, config, hamiltonian, lindblad):
        Propagator.__init__(self, config, hamiltonian)
        self.lindblad = lindblad
        self.identity = np.mat(np.eye(self.qubit_dim  ** self.num_qubits, dtype=np.complex128))

    def _next(self):
        ham = self.hamiltonian.next()
        intermediate = -1.j * (np.kron(self.identity, ham) - np.kron(ham.T, self.identity)) / self.hbar + self.lindblad
        return np.mat(scipy.linalg.expm(intermediate * self.dt))

    def propagate(self):
        superoperator_matrix = self._get_initial_state()
        for index in xrange(self.num_steps):
            superoperator_matrix = np.dot(self._next(), superoperator_matrix)
            if self.sample_interval and (index+1) % self.sample_interval == 0:
                self.samples.append(copy.copy(superoperator_matrix))
        return superoperator_matrix

    def _get_initial_state(self):
        return np.mat(np.kron(self.identity, self.identity))
