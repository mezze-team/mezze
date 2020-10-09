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

import scipy
import scipy.special
import numpy as np
import logging


"""
Noise generators are created using NoiseFactory:

 noise_factory = NoiseFactory(time_final=1., num_steps=100)
 white_generator = noise_factory.get({'type':'white', 'amp':0.5})
 noise = white_generator.generate()

A noise generator is specified through a dictionary of parameters.
See NoiseFactory for a list of parameters per noise type and see the individual
noise classes (like WhiteNoise) for more detailed information on those parameters.

To add a new type of noise, define a class that inherits from FFTNoise for
Gaussian noise processes or Noise for non-Gaussian noise sources.
Then add your noise type to NoiseFactory.

Gaussian noise sources are generated through a Fourier transform method that
requires an autocorrelation function and the power spectral density function.
See ExponentialNoise for an example.

Non-Gaussian noise generation requires the object to directly generate the samples.
See TelegraphNoise for an example.

"""



# not used for anything yet - maybe replace use of strings in PMDs
class NoiseLabels(object):
    null = 0
    white = 1
    white_cutoff = 2
    pink = 3
    pink_cutoff = 4
    gaussian = 5
    exponential = 6
    telegraph = 7
    user_defined = 100


class Noise(object):
    """
    Base noise class
    """

    _seed = 0

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @classmethod
    def seed(cls, seed):
        """
        Set the seed for the random generator for repeatable experiments.
        Only works for single threaded simulations
        """
        cls._seed = seed
        np.random.seed(seed)

    @classmethod
    def get_next_seed(cls):
        """
        Get a random sequence of seeds for the numpy RandomState object for
        repeatable experiments. Should be used by Noise objects only.
        """
        return np.random.randint(0, 2**31)

    @classmethod
    def has_seed(cls):
        """
        """
        return cls._seed != 0


class FFTNoise(Noise):
    """
    FFT-based noise generation using a correlation function
    """
    def __init__(self, time_final, num_steps, corrfn, over_write_zero=False):
        Noise.__init__(self)
        self.num_steps = num_steps
        self.new_num_steps = int(2**int(np.ceil(np.log(num_steps)/np.log(2))) + 1)
        self.dt = time_final / float(num_steps)
        times = np.arange(self.new_num_steps) * self.dt
        if over_write_zero:
            times[0] = times[1]/10000.  # avoid divide by zeros warnings
        corr_values = map(corrfn, times)
        self.phi = self._calculate_phi(corr_values)

    def _calculate_phi(self, corr_values):
        corr_values = list(corr_values)
        rho = np.real(np.concatenate((corr_values, np.flipud(corr_values[1:-1]))))
        q = np.real(scipy.ifft(rho))*len(rho)
        q[q<0] = 0
        return np.sqrt(q)

    def generate(self):
        n = self.new_num_steps
        N = 2 * (n-1)
        rgen = np.random.RandomState()
        # for repeatable experiments
        if Noise.has_seed():
            rgen.seed(Noise.get_next_seed())
        z = np.zeros(len(self.phi), dtype=complex)
        z[0] = rgen.randn(1) * np.sqrt(N)
        z[n-1] = rgen.randn(1) * np.sqrt(N)
        z[1:n-1] = (rgen.randn(n-2) +1j * rgen.randn(n-2))*np.sqrt(N/2)
        z[n:] = np.conj(np.flipud(z[1:n-1]))
        x = scipy.fft(self.phi*z)/N
        return np.real(x[0:self.num_steps])

    def _estimate_psd(self, x):
        """
        Returns frequency (Hz), corresponding spectral densities.
        """
        f, Pxx =  scipy.signal.periodogram(x, 1./self.dt)
        return f, Pxx


class NullNoise(Noise):
    """
    No noise. Can be used for unit tests or debugging
    """
    def __init__(self, num_steps):
        Noise.__init__(self)
        self.num_steps = num_steps
        self.corrfn = lambda t: 0.
        self.psd = lambda omega: 0.

    def generate(self):
        return scipy.zeros(self.num_steps)


class WhiteNoise(Noise):
    """
    Make white noise with power spectrum S(omega) = magnitude
    """
    def __init__(self, time_final, num_steps, magnitude):
        Noise.__init__(self)
        self.num_steps = num_steps
        dt = float(time_final) / num_steps
        self.sigma = scipy.sqrt(magnitude / dt)
        self.corrfn = lambda t: 0.
        self.psd = lambda omega: magnitude

    def generate(self):
        data = scipy.zeros(self.num_steps)
        for index in xrange(self.num_steps):
            data[index] = np.random.randn()*self.sigma
        return data


class WhiteCutoffNoise(FFTNoise):
    """
    Make white noise with power spectrum
    S(omega) = noise_power for omega_min < omega < omega_max, and
    S(omega) = 0 for omega < omega_min and omega > omega_max.
    """
    def __init__(self, time_final, num_steps, amplitude, omega_min, omega_max):
        self.amplitude = amplitude
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.corrfn = lambda t: (self.amplitude/(np.pi*t)) * (scipy.sin(self.omega_max*t) - scipy.sin(self.omega_min*t))
        self.psd = lambda omega: amplitude if np.abs(omega) < omega_max and np.abs(omega) > omega_min else 0.
        FFTNoise.__init__(self, time_final, num_steps, self.corrfn, True)


class PinkCutoffNoise(FFTNoise):
    """
    Make 1/f (pink) noise with hard cutoffs.  The power spectrum is
    S(omega) = amplitude / omega for omega_min < omega < omega_max,
    and S(omega) = 0 for omega < omega_min and omega > omega_max.
    And integrating the 1/f power spectrum between these values, to get a correlation
    function:
      C(t) = (ampl/pi)*[  ci(omega_max*t) - ci(omega_min*t) ]
    where ci is the cosine integral function.
    Note: Here, S(omega) = 0 for omega < omega_min
    """
    def __init__(self, time_final, num_steps, amplitude, omega_min, omega_max):
        self.amplitude = amplitude
        self.omega_min = omega_min
        self.omega_max = omega_max
        ci = lambda x: scipy.special.sici(x)[1]
        self.corrfn = lambda t: (amplitude/np.pi) * (ci(omega_max*t) - ci(omega_min*t))
        self.psd = lambda omega: amplitude/np.abs(omega) if np.abs(omega) < omega_max and np.abs(omega) > omega_min else 0.
        FFTNoise.__init__(self, time_final, num_steps, self.corrfn, True)


class GaussianNoise(FFTNoise):
    """
    Defines a noise source with a spectrum that is the same as the pdf of a
    Gaussian random variable.
    """
    def __init__(self, time_final, num_steps, amplitude, corr_time):
        self.amplitude = amplitude
        self.corr_time = corr_time
        self.corrfn = lambda t: amplitude/2/np.sqrt(np.pi)/corr_time*scipy.exp(-(t/2/corr_time)**2)
        self.psd = lambda omega: amplitude*scipy.exp(-(corr_time*omega)**2)
        # For QFI expressions
        self.partial_amp = lambda omega: scipy.exp(-(corr_time*omega)**2)
        self.partial_corr_time = lambda omega: -2.*corr_time*omega**2*amplitude*scipy.exp(-(corr_time*omega)**2)
        FFTNoise.__init__(self, time_final, num_steps, self.corrfn)

class BiModalGaussianNoise(Noise):
    """
    Defines a noise source with a spectrum that is the same as the pdf...
    """
    def __init__(self, time_final, num_steps, amp_list, corr_list):
        #self.amplitude = amplitude
        #self.corr_time = corr_time
        A1, A2 = amp_list
        t1, t2, w0 = corr_list
        #self.corrfn = lambda t: amplitude/2/np.sqrt(np.pi)/corr_time*scipy.exp(-(t/2/corr_time)**2)
        self.psd = lambda omega: A1*scipy.exp(-(t1*omega)**2) + A2*scipy.exp(-(t2*(omega-w0))**2)
        # For QFI expressions
        #self.partial_amp = lambda omega: scipy.exp(-(corr_time*omega)**2)
        #self.partial_corr_time = lambda omega: -2.*corr_time*omega**2*amplitude*scipy.exp(-(corr_time*omega)**2)
        #FFTNoise.__init__(self, time_final, num_steps, self.corrfn)


class ExponentialNoise(FFTNoise):
    """
    Defines a noise source with an exponentially decaying correlation function
    """
    def __init__(self, time_final, num_steps, amplitude, corr_time):
        self.amplitude = amplitude
        self.corr_time = corr_time
        self.corrfn = lambda t: amplitude * scipy.exp(-t/corr_time)
        self.psd = lambda omega: 2*amplitude/corr_time/(corr_time**-2+(omega)**2)
        FFTNoise.__init__(self, time_final, num_steps, self.corrfn)


class TelegraphNoise(Noise):
    """
    Defines telegraph noise by the two telegraph values and the two scale
    parameters of the exponential distribution that drive the continuous
    time Markov chain.  If one scale parameter (taus) is defined, it is
    used for both switching processes.
    """
    def __init__(self, time_final, num_steps, vals, taus):
        Noise.__init__(self)
        self.telegraph_vals = vals
        try:
            if len(taus) == 1:
                taus = list(taus) + list(taus)
        except TypeError:
                taus = [taus, taus]
        self.telegraph_taus = taus
        self.times = scipy.linspace(0., time_final, int(num_steps) + 1)
        self.corrfn = lambda t: (time_final-t)/time_final* \
        ( vals[0]**2*(taus[0]**2/(sum(taus)**2) + taus[0]*taus[1]/(sum(taus)**2)*np.exp(-(1./taus[0]+1./taus[1])*t)) \
        + vals[0]*vals[1]*(taus[0]*taus[1]/(sum(taus)**2) - taus[0]*taus[1]/(sum(taus)**2)*np.exp(-(1./taus[0]+1./taus[1])*t)) \
        + vals[1]*vals[0]*(taus[1]*taus[0]/(sum(taus)**2) - taus[0]*taus[1]/(sum(taus)**2)*np.exp(-(1./taus[0]+1./taus[1])*t)) \
        + vals[1]**2*(taus[1]**2/(sum(taus)**2) + taus[0]*taus[1]/(sum(taus)**2)*np.exp(-(1./taus[0]+1./taus[1])*t)))
        self.psd = lambda omega: 2*((vals[0]-vals[1])**2)/(taus[0]+taus[1])/((1./taus[0]+1./taus[1])**2 + omega**2)

    def generate(self):
        tau_1 = float(self.telegraph_taus[0])
        tau_2 = float(self.telegraph_taus[1])
        # Draw sample from stationary distribution
        idx = np.random.rand() > tau_1 / (tau_1 + tau_2)
        t = 0
        x = np.zeros(len(self.times)-1)
        while t < self.times[-1]:
            if idx == 0:
                t_next = t + np.random.exponential(scale=tau_1)
                ts = (self.times[:-1] >= t) & (self.times[:-1] < t_next)
                x[ts] = self.telegraph_vals[idx]
                idx = 1
            elif idx == 1:
                t_next = t + np.random.exponential(scale=tau_2)
                ts = (self.times[:-1] >= t) & (self.times[:-1] < t_next)
                x[ts] = self.telegraph_vals[idx]
                idx = 0
            t = t_next
        return x


class UserDefinedNoise(FFTNoise):
    """
    User defined correlation function
    """
    def __init__(self, time_final, num_steps, corrfn):
        self.corrfn = corrfn
        FFTNoise.__init__(self, time_final, num_steps, corrfn)


class NoiseFactory(object):
    """
    Factory for producing noise generators given a dictionary of parameters

    Types: white, pink, gauss, exp, time, telegraph, null

    Parameters:
    null: none
    white: amp
    white (cutoff): amp, omega_min, omega_max
    pink (cutoff): amp, omega_min, omega_max
    gauss: amp, corr_time
    exp: amp, corr_time
    time: corrfn
    telegraph: vals, taus
    """
    def __init__(self, time_final, num_steps):
        self.time_final = time_final
        self.num_steps = num_steps

    def get(self, params):
        if params['type'] == 'white':
            if 'omega_min' in params.keys() or 'omega_max' in params.keys():
                return WhiteCutoffNoise(self.time_final, self.num_steps, params['amp'], params['omega_min'],
                                        params['omega_max'])
            else:
                return WhiteNoise(self.time_final, self.num_steps, params['amp'])
        elif params['type'] == 'pink':
            if 'omega_min' in params.keys() or 'omega_max' in params.keys():
                return PinkCutoffNoise(self.time_final, self.num_steps, params['amp'], params['omega_min'],
                                       params['omega_max'])
            else:
                raise NotImplementedError
        elif params['type'] == 'gauss':
            return GaussianNoise(self.time_final, self.num_steps, params['amp'], params['corr_time'])
        elif params['type'] == 'exp':
            return ExponentialNoise(self.time_final, self.num_steps, params['amp'], params['corr_time'])
        elif params['type'] == 'time':
            return UserDefinedNoise(self.time_final, self.num_steps, params['corrfn'])
        elif params['type'] == 'telegraph':
            return TelegraphNoise(self.time_final, self.num_steps, params['vals'], params['taus'])
        elif params['type'] == 'null':
            return NullNoise(self.num_steps)
        else:
            raise NotImplementedError("No noise class of type '%s'" % params['type'])
