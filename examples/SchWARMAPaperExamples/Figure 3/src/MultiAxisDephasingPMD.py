from mezze.pmd import PMD
import mezze.core as uar
import scipy.special
import numpy as np


class MultiAxisDephasing(PMD):
    def __init__(self, num_qs, amp, corr_time):
        num_qubits = num_qs
        single_control_dim = 3
        single_noise_dim = 3
        two_control_dim = 1
        two_noise_dim = 0
        num_lindblad = 0
        PMD.__init__(self, single_control_dim, single_noise_dim, num_lindblad, num_qubits, two_control_dim, two_noise_dim)
        
        self.single_qubit_control[0][1] = lambda c, gamma: c[0]
        self.single_qubit_control[0][2] = lambda c, gamma: c[1]
        #self.single_qubit_control[0][3] = lambda c, gamma: c[2]
        
        self.single_qubit_noise[1] = lambda c, gamma: gamma[0] / 2.0
        self.single_qubit_noise[2] = lambda c, gamma: gamma[1] / 2.0
        self.single_qubit_noise[3] = lambda c, gamma: gamma[2] / 2.0
        #self.noise_hints[0]={'type':'gauss', 'amp': noise_amp, 'ham_value': 1, 'corr_time': corr_time, 'ctrl': False}
        #self.noise_hints = [{'type': 'gauss', 'amp': noise_amp, 'ham_value': 1, 'corr_time': corr_time, 'ctrl': False}]*num_qubits
        for i in range(num_qubits*3):
            self.noise_hints[i]['type'] = 'gauss'
        if len(amp) == 1:
            for i in range(num_qubits * 3):
                self.noise_hints[i]['amp'] = amp[0]
                self.noise_hints[i]['corr_time'] = corr_time[0]
        else:
            for i in range(num_qubits):
                self.noise_hints[3*i]['amp'] = amp[0]
                self.noise_hints[1 + 3*i]['amp'] = amp[1]
                self.noise_hints[2 + 3*i]['amp'] = amp[2]
                self.noise_hints[3 * i]['corr_time'] = corr_time[0]
                self.noise_hints[1 + 3 * i]['corr_time'] = corr_time[1]
                self.noise_hints[2 + 3 * i]['corr_time'] = corr_time[2]

        self.two_qubit_control[0][3][3] = lambda c, gamma: c[0]


class SingleQubitDephasing(PMD):
    def __init__(self, noise_amp, corr_time):
        num_qubits = 1
        single_control_dim = 3
        single_noise_dim = 1
        two_control_dim = 0
        two_noise_dim = 0
        num_lindblad = 0
        PMD.__init__(self, single_control_dim, single_noise_dim, num_lindblad, num_qubits, two_control_dim, two_noise_dim)
        
        self.single_qubit_control[0][1] = lambda c, gamma: c[0]
        self.single_qubit_control[0][2] = lambda c, gamma: c[1]
        #self.single_qubit_control[0][3] = lambda c, gamma: c[2]
        
        self.single_qubit_noise[1] = lambda c, gamma: gamma[0]
        self.noise_hints[0]={'type':'gauss', 'amp': noise_amp, 'ham_value': 1, 'corr_time': corr_time, 'ctrl': False}
        