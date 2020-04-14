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

from .pmd import PMD
from .core import *
import scipy.special
import numpy as np



class ExamplePMD(PMD):
    """
    Example PMD used to illustrate code without FOUO restrictions.
    """
    def __init__(self, num_qubits=1):
        control_dim = 3
        noise_dim = 5
        num_lindblad = 0

        if is_multi_qubit(num_qubits):
            two_control_dim = 1
            two_noise_dim = 2
            PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits, two_control_dim, two_noise_dim)
        else:
            PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        #This is f_x, f_y, f_z in Eq 1 in terms of control and noise
        self.single_qubit_control[0][1]=lambda c, gamma: c[0]*(1+gamma[0]+gamma[1])*np.cos(c[1])/2.0
        self.single_qubit_control[0][2]=lambda c, gamma: c[0]*(1+gamma[0]+gamma[1])*np.sin(c[1])/2.0
        self.single_qubit_control[0][3]=lambda c, gamma: 0

        self.single_qubit_control[2][1]=lambda c, gamma: 0
        self.single_qubit_control[2][2]=lambda c, gamma: 0
        self.single_qubit_control[2][3]=lambda c, gamma: c[2]*(1+gamma[3]+gamma[4])/2.0

        self.single_qubit_drift[1]=lambda c, gamma: 0
        self.single_qubit_drift[2]=lambda c, gamma: 0
        self.single_qubit_drift[3]=lambda c, gamma: 2*np.pi*1e9/2.0

        self.single_qubit_noise[1]=lambda c, gamma: 0
        self.single_qubit_noise[2]=lambda c, gamma: 0
        self.single_qubit_noise[3]=lambda c, gamma: gamma[2]/2.0

        #These are the constraints on control and its derivatives
        self.c_const=[0 for x in range(3)]
        self.c_const[0]=lambda c, dc: 0<=c[0] and c[0]<=1e6
        self.c_const[1]=lambda c, dc: 0<=c[1] and c[1]<=2*np.pi
        self.c_const[2]=lambda c, dc: 0<=c[2] and c[2]<=1e6

        #Replace the None with 0
        for x in range(len(self.noise_spectral_density)):
            for y in range(len(self.noise_spectral_density)):
                self.noise_spectral_density[x][y]=lambda w: 0

        #Each entry S[i][j] is the Fourier transform of the autocorrelation between noise i and j
        self.noise_spectral_density[0][0]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-18
        self.noise_spectral_density[1][1]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-9/abs(w)
        self.noise_spectral_density[2][2]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*2*np.pi*1e-4
        self.noise_spectral_density[3][3]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-18
        self.noise_spectral_density[4][4]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-9/abs(w)

        self.noise_hints[0]={'type':'white','amp':1e-18,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
        self.noise_hints[1]={'type':'pink','amp':1e-9,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
        self.noise_hints[2]={'type':'white','amp':2*np.pi*1e-4,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
        self.noise_hints[3]={'type':'white','amp':1e-18,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
        self.noise_hints[4]={'type':'pink','amp':1e-9,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}

        if is_multi_qubit(self.num_qubits):
            #g_zz control function
            self.two_qubit_control[0][0][0] = lambda c, gamma: c[0]*(1+gamma[0]+gamma[1])
            self.two_qubit_control[0][3][3] = lambda c, gamma: -c[0]*(1+gamma[0]+gamma[1])

            #constraints on two qubit control
            self.two_c_const=[0 for x in range(1)]
            self.two_c_const[0] = lambda c, dc: 0<= c[0] and c[0]<= 1.5e4

            #Total system noise first noise_dim are for qubit one, second noise_dim for
            #qubit 2, last two_noise_dim for two qubit noise
            self.noise_spectral_density[5][5]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-18
            self.noise_spectral_density[6][6]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-9/abs(w)
            self.noise_spectral_density[7][7]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*8*np.pi*1e-4
            self.noise_spectral_density[8][8]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-18
            self.noise_spectral_density[9][9]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-9/abs(w)
            self.noise_spectral_density[10][10]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-18
            self.noise_spectral_density[11][11]=lambda w: (2*np.pi<abs(w) and abs(w)<2*np.pi*1e9)*1e-9/abs(w)

            self.noise_hints[5]={'type':'white','amp':1e-18,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
            self.noise_hints[6]={'type':'pink','amp':1e-9,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
            self.noise_hints[7]={'type':'white','amp':2*np.pi*1e-4,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
            self.noise_hints[8]={'type':'white','amp':1e-18,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
            self.noise_hints[9]={'type':'pink','amp':1e-9,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
            self.noise_hints[10]={'type':'white','amp':1e-18,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}
            self.noise_hints[11]={'type':'pink','amp':1e-9,'omega_min':2*np.pi,'omega_max':2*np.pi*1e9,'ham_value':1}



class BlattPMDComp(PMD):
    """

    """
    def __init__(self, num_qubits=1):
        control_dim = 0
        noise_dim = 2
        num_lindblad = 0

        if is_multi_qubit(num_qubits):
            two_control_dim = 0
            two_noise_dim = 0
            PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits, two_control_dim, two_noise_dim)
        else:
            PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        self.basis = [s_z]
        self.single_qubit_noise[0] = lambda c, gamma: (gamma[0]+gamma[1])/2.0
        self.noise_hints[0] = {'type':'exp', 'amp' : 1.0, 'corr_time' : 1.0, 'ctrl': False}
        #This is the noise source that we will add in to all others through the cross talk
        #matrix -- will need to change amplitude at that point
        self.noise_hints[1] = {'type':'exp', 'amp' : 0.0, 'corr_time' : 1.0, 'ctrl': False}
        for j in range(1, self.num_qubits):
            self.xtalk[1,j*2+1] = 1.0

        if is_multi_qubit(num_qubits):
            self.noise_hints[2] = {'type':'exp', 'amp' : 1.0, 'corr_time' : 1.0, 'ctrl': False}
            self.noise_hints[3] = {'type':'exp', 'amp' : 0.0, 'corr_time' : 1.0, 'ctrl': False}



class Z_noise_X_control_qubit(PMD):
    """
    Single qubit system with dephasing and X control
    
    x-control: c[0]
    dephasing: gamma[0]
    """

    def __init__(self):
        num_qubits = 1
        control_dim = 1
        noise_dim = 1
        num_lindblad = 0
        PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        self.single_qubit_control[0][1] = lambda c, gamma : c[0]/2.0

        self.single_qubit_noise[3] = lambda c, gamma: gamma[0]/2.0

        self.noise_hints[0]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': False}
        self.noise_basis = [2] # must be channel# - 1, e.g. Z=3-1=2



class Z_noise_XY_control_qubit(PMD):
    """
    Single qubit system with dephasing and x,y control
    
    x-control: c[0]
    y-control: c[1]
    
    dephasing: gamma[0]
    """

    def __init__(self):
        num_qubits = 1
        control_dim = 2
        noise_dim = 1
        num_lindblad = 0
        PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        self.single_qubit_control[0][1] = lambda c, gamma : c[0]/2.0
        self.single_qubit_control[0][2] = lambda c, gamma : c[1]/2.0

        self.single_qubit_noise[3] = lambda c, gamma: gamma[0]/2.0

        self.noise_hints[0]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': False}
        self.noise_basis = [2] # must be channel# - 1, e.g. Z=3-1=2



class XYZ_noise_XY_control_Qubit(PMD):
    """
    Single qubit system with additive noise along x,y,z channels with x,y controls
    
    x-control: c[0]
    y-control: c[1]
    
    x-noise: gamma[0]
    y-noise: gamma[1]
    z-noise: gamma[2]
    """

    def __init__(self):
        num_qubits = 1
        control_dim = 2
        noise_dim = 3
        num_lindblad = 0
        PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        self.single_qubit_control[0][1] = lambda c, gamma : c[0]/2.0
        self.single_qubit_control[0][2] = lambda c, gamma : c[1]/2.0

        self.single_qubit_noise[1] = lambda c, gamma: gamma[0]/2.0
        self.single_qubit_noise[2] = lambda c, gamma: gamma[1]/2.0
        self.single_qubit_noise[3] = lambda c, gamma: gamma[2]/2.0

        self.noise_hints[0]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': False}
        self.noise_hints[1]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': False}
        self.noise_hints[2]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': False}
        self.noise_basis = [0,1,2] # must be channel# - 1, e.g. Z=3-1=2



class UniversalDephasingQubit(PMD):
    """
    Single qubit system with dephasing noise and x,y control
    
    Control is expressed in polar coordinates: rabi rate and phase angle
    
    rabi rate: c[0]
    phase angle: c[1]
    
    dephasing: gamma[0]
    """
    def __init__(self):
        num_qubits = 1
        control_dim = 2
        noise_dim = 1
        num_lindblad = 0
        PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        self.single_qubit_control[0][1] = lambda c, gamma : c[0]*np.cos(c[1])/2.0
        self.single_qubit_control[0][2] = lambda c, gamma : c[0]*np.sin(c[1])/2.0

        self.single_qubit_noise[3] = lambda c, gamma: gamma[0]/2.0

        self.noise_time_correlation[0][0] = lambda t: np.exp(-t**2/2.0)
        self.noise_spectral_density[0][0] = lambda w: np.sqrt(2*np.pi)*np.exp(-w**2/2)
        self.noise_hints[0]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': False}



class UniversalControlNoiseQubit(PMD):
    """
    Single qubit system with multiplicative control noise
    
    Control is expressed in polar coordinates: rabi rate and phase angle

    rabi rate: c[0]
    phase angle: c[1]
    
    control noise: gamma[0]
    """
    def __init__(self):
        num_qubits = 1
        control_dim = 2
        noise_dim = 1
        num_lindblad = 0
        PMD.__init__(self, control_dim, noise_dim, num_lindblad, num_qubits)

        self.single_qubit_control[0][1] = lambda c, gamma : (1+gamma[0])*c[0]*np.cos(c[1])/2.0
        self.single_qubit_control[0][2] = lambda c, gamma : (1+gamma[0])*c[0]*np.sin(c[1])/2.0

        self.noise_time_correlation[0][0] = lambda t: np.exp(-t**2/2.0)
        self.noise_spectral_density[0][0] = lambda w: np.sqrt(2*np.pi)*np.exp(-w**2/2)
        self.noise_hints[0]={'type':'gauss', 'amp': 1.0, 'ham_value': 1, 'corr_time': 1.0, 'ctrl': True}


