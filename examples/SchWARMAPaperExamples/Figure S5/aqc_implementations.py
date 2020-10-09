from mezze.pmd import PMD
import mezze.core as uar
import scipy.special
import numpy as np


class LZSTransverse(PMD):
    def __init__(self, noise_amp=0, corr_time=1):
        num_qubits = 1
        self.num_qubits = num_qubits
        single_control_dim = 2
        single_noise_dim = 1
        num_lindblad = 0
        two_control_dim = 0
        two_noise_dim = 0
        PMD.__init__(self, single_control_dim, single_noise_dim, num_lindblad, num_qubits, two_control_dim, two_noise_dim)        
        self.single_qubit_control[0][0] = lambda c, gamma: 0
        self.single_qubit_control[0][1] = lambda c, gamma: c[0]/2
        self.single_qubit_control[0][3] = lambda c, gamma: c[1]/2
        
        self.single_qubit_noise[1] = lambda c,gamma: gamma[0]
        
        self.noise_hints[0] = {'type':'gauss', 'amp': noise_amp, 'ham_value': 1, 'corr_time': corr_time, 'ctrl': False}


class LZNTransverse(PMD):
    def __init__(self, basis):
        num_qubits = 1
        self.num_qubits = num_qubits
        single_control_dim = 2
        single_noise_dim = 1
        num_lindblad = 0
        two_control_dim = 0
        two_noise_dim = 0
        PMD.__init__(self, single_control_dim, single_noise_dim, num_lindblad, num_qubits, two_control_dim, two_noise_dim)
        self.basis = basis
        self.single_qubit_control[0][0] = lambda c, gamma: c[0]
        self.single_qubit_control[0][1] = lambda c, gamma: c[1]
        
        # Noise
        self.single_qubit_noise[2] = lambda c,gamma: gamma[0]
        self.noise_hints[0] = {'type':'gauss', 'amp': 0.01, 'ham_value': 1, 'corr_time': 2.0, 'ctrl': False}

        

class DephasingGrover(PMD):
    def __init__(self, num_qubits, init_state, marked_state, O_noise):
        self.real_num_qubits = 1
        control_dim = 2
        noise_dim = 1
        num_lindblad = 0

        PMD.__init__(self, control_dim, noise_dim, num_lindblad, self.real_num_qubits)

        Iden = np.identity(2**num_qubits)
        init_projector = init_state @ np.conjugate(np.transpose(init_state))
        H0 = Iden - init_projector
        marked_projector = marked_state @ np.conjugate(np.transpose(marked_state))
        Hp = Iden - marked_projector
        self.basis = [H0, Hp, O_noise]

        # c = [A(t), B(t)]
        self.single_qubit_control[0][0] = lambda c, gamma: c[0]
        self.single_qubit_control[0][1] = lambda c, gamma: c[1]
        
        # Noise
        self.single_qubit_noise[2] = lambda c,gamma: gamma[0]
        self.noise_hints[0] = {'type':'gauss', 'amp': 0.01, 'ham_value': 1, 'corr_time': 2.0, 'ctrl': False}