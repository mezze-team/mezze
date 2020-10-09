# Copyright: 2020 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
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

import sys
import mezze
import mezze.metrics
import mezze.random.SchWARMA
from mezze.channel import _sigmaI, _sigmaX, _sigmaY, _sigmaZ
import numpy as np
from MultiAxisDephasingPMD import MultiAxisDephasing
from Gates import Gates
import pickle
import qecc
import qutip as qt
import time
import os
import scipy.signal as si
import scipy.linalg as la
from qutip_gates import *
import scipy.optimize as op
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

######################################################################################################################
# This is a script that is intended to be called from a cluster based submission script. It accepts three arguments: #
# The first argument is an integer between [0,..,5] inclusively. The second is an integer between [0,..,7]           #
# inclusively. These two arguments select the noise amplitude and noise correlation time respectively from a         #
# predefined set of values. These are the values used in arXiv:xxxx.xxxxx. The final argument is a string indicating #
# the directory to save the output files to.                                                                         #
######################################################################################################################

start = time.time()

directory = str(sys.argv[3])+"/"
print(directory)

# Parameters that strongly impact runtime. dcv=1000 is typically used. This makes each trotter step 1/1000 of a gate.
# num_realz is the number of monte carlo trials used for both the Trotter and SchWARMA simulations. num_realz=1000
# provides good convergance.
dcv = 1000 #down conversion in time from mezze to schwarma
num_realz = 1000 # Number of monte carlo trials for both Trotter and SchWARMA simulation

# Noise spectra parameters to be run
s_pow_list = [1.0e-12, 1.0e-10, 1e-8, 1e-6, 1e-4, 1e-2]
s_pow = s_pow_list[int(sys.argv[1])]
s_amp = np.sqrt(s_pow)
NN_spec = 24 # Number of discrete time steps in the quantum circuit. This is number of SchWARMA steps
corr_time_list = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
corr_time = corr_time_list[int(sys.argv[2])]

# Other setup parameters
NN = NN_spec * dcv # Total number of steps in the Trotter circuit
T_max = float(NN_spec) # Sets gates to 1 unit of time
dt = T_max / NN
num_qubits = 5

# Setup mezze configuration. This is the full Trotter based simulation of the circuit
config = mezze.SimulationConfig()  # Get the config structure (class)
config.num_steps = NN  # Number of time steps to use in the simulation (larger numbers are more accurate)
config.time_length = T_max  # Set the time length of the integration
config.dt = float(config.time_length) / float(config.num_steps)  # Set the time-step size
config.parallel = False  # Set to true for parallel Monte Carlo, False otherwise
config.num_runs = num_realz  # This sets the number of Monte Carlo trials to use to compute the average superoperator
config.get_unitary = False  # Report unitary as opposed to Liouvillian
time_axis = np.linspace(0., config.time_length, config.num_steps)

# Initialize Physical Machine Description
pmd = MultiAxisDephasing(num_qubits, [s_pow, s_pow, s_pow], [corr_time, corr_time, corr_time])
# Surface Syndome Circuit
surf_xsyndrome = qecc.SurfaceSyndrome()
# Build controls
surf_xsyndrome.generate_xsyndrome_controls()
sys.stdout.flush()

# Build control pulses for the surface code syndrome circuit
circuit = Gates(pmd, config)
sq_ctrls = []
for n in range(num_qubits):
    ctrls = circuit.generate_circuit_controls(surf_xsyndrome.ctrl_list[n])
    for cc in ctrls:
        sq_ctrls.append(cc)
tq_ctrls = []
for n in range(5, len(surf_xsyndrome.ctrl_list)):
    ctrls = circuit.generate_circuit_controls(surf_xsyndrome.ctrl_list[n])
    tq_ctrls.append(ctrls[2])

controls = np.transpose(np.concatenate((sq_ctrls, tq_ctrls), axis=0))

# Run Trotter dynamics for syndrome circuit using mezze simulation methodology
mezze_start = time.time()

ham_func = mezze.HamiltonianFunction(pmd)
ham = mezze.PrecomputedHamiltonian(pmd, controls, config, ham_func.get_function())
sim = mezze.Simulation(config, pmd, ham)
report = sim.run()
Ut = report.channel.liouvillian()

mezze_end = time.time()
print('mezze runtime: %d' % (mezze_end - mezze_start))

# Get Schwarma model parameters using build in helper functions. These model paramaters are obtained from the
# high-fideltiy simulation correlation function and fit to give equivalent results.
a = [1,]
bx = mezze.random.SchWARMA.extract_and_downconvert_from_sim(sim,1.0)[0]
by = mezze.random.SchWARMA.extract_and_downconvert_from_sim(sim,1.0)[1]
bz = mezze.random.SchWARMA.extract_and_downconvert_from_sim(sim,1.0)[2]

# Generate perfect circuit for syndrome circuit using qutip
qx = qt.qip.QubitCircuit(5, reverse_states=False)
qx.add_gate("SNOT", targets=[0])
qx.add_gate("CNOT", controls=[0], targets=[2])
qx.add_gate("CNOT", controls=[0], targets=[1])
qx.add_gate("CNOT", controls=[0], targets=[4])
qx.add_gate("CNOT", controls=[0], targets=[3])
qx.add_gate("SNOT", targets=[0])
perfect_xcircuit = qt.gate_sequence_product(qx.propagators())
xcircuit_infidelity = mezze.metrics.process_infidelity(report.channel, mezze.channel.QuantumChannel(perfect_xcircuit.full(), 'unitary'))

# Now simulate syndrome circuit with SchWARMA noise model and qutip circuit simulator
schwarma_xcircuit_infidelities = [0.0]*num_realz
schwarma_start = time.time()
for k in range(num_realz):
    schwarma_xangles = np.array([mezze.random.SchWARMA.schwarma_trajectory(NN_spec, a, bx, amp=1.0) for i in range(5)])
    schwarma_yangles = np.array([mezze.random.SchWARMA.schwarma_trajectory(NN_spec, a, by, amp=1.0) for i in range(5)])
    schwarma_zangles = np.array([mezze.random.SchWARMA.schwarma_trajectory(NN_spec, a, bz, amp=1.0) for i in range(5)])
    qx_err = qt.qip.QubitCircuit(5, reverse_states=False)
    qx_err.user_gates = {"CZZ": zz_gate, "XYZ": xyz_gate}
    qx_err.add_gate("SNOT", targets=[0])
    add_qtcnot(qx_err, 0, 2, xerr=schwarma_xangles[:, 0:NN_spec//4], yerr=schwarma_yangles[:, 0:NN_spec//4], zerr=schwarma_zangles[:, 0:NN_spec//4])
    add_qtcnot(qx_err, 0, 1, xerr=schwarma_xangles[:, NN_spec//4:2*NN_spec//4], yerr=schwarma_yangles[:, NN_spec//4:2*NN_spec//4], zerr=schwarma_zangles[:, NN_spec//4:2*NN_spec//4])
    add_qtcnot(qx_err, 0, 4, xerr=schwarma_xangles[:, 2*NN_spec//4:3*NN_spec//4], yerr=schwarma_yangles[:, 2*NN_spec//4:3*NN_spec//4], zerr=schwarma_zangles[:, 2*NN_spec//4:3*NN_spec//4])
    add_qtcnot(qx_err, 0, 3, xerr=schwarma_xangles[:, 3*NN_spec//4:NN_spec], yerr=schwarma_yangles[:, 3*NN_spec//4:NN_spec], zerr=schwarma_zangles[:, 3*NN_spec//4:NN_spec])
    qx_err.add_gate("SNOT", targets=[0])
    noisy_schwarma_xcircuit = qt.gate_sequence_product(qx_err.propagators())
    schwarma_xcircuit_infidelities[k] = mezze.metrics.process_infidelity(mezze.channel.QuantumChannel(noisy_schwarma_xcircuit.full(), 'unitary'), mezze.channel.QuantumChannel(perfect_xcircuit.full(), 'unitary'))

schwarma_end = time.time()
print('schwarma runtime: %d' % (schwarma_end - schwarma_start))

schwarma_xcircuit_infidelity = np.mean(schwarma_xcircuit_infidelities)
print("X Syndrome Circuit Infidelity: " + str(xcircuit_infidelity))
print("Schwarma X Syndrome Circuit Infidelity: " + str(schwarma_xcircuit_infidelity))

output = {"SPower": s_pow, "CorrelationTime":corr_time, "Config": config, "FullSimInfidelity":xcircuit_infidelity, "SchwARMAInfidelity":schwarma_xcircuit_infidelity, "bx":bx, "by":by, "bz":bz, "Noise Type": "XYZ Equal"}
pickle.dump(output, open(directory + 'surface_xsyndrome_xyznoise-%d-%d.p' % (int(sys.argv[1]), int(sys.argv[2])), 'wb'))

end = time.time()
print('total runtime: %d' % (end - start))
