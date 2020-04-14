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

import copy
import numpy as np
from .core import *

# metaclass that ensures that build_out_multi_qubit is called
class MultiQubitPreBuilder(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if is_multi_qubit(obj.num_qubits):
            obj.build_out_multi_qubit()
        return obj

class PMD(object):
    """
    Physical Machine Description base class. Creates structures to hold all terms requires to
    implement Hamiltonian and non-Hamiltonian based PMDs.

    :param control_dim: Number of single qubit control dimensions
    :type control_dim: int
    :param noise_dim: Number of single noise sources
    :type noise_dim: int
    :param num_lindblad: Number of Lindblad operators
    :type num_lindblad: int
    :param num_qubits: Number of qubits (1 or 2)
    :type num_qubits: int
    :param two_control_dim: Number of two-qubit control dimensions
    :type two_control_dim: int
    :param two_noise_dim: Number of two-qubit noise sources
    :type two_noise_dim: int
    :param num_two_lindblad: Number of two qubit Lindblad operators
    :type num_two_lindblad: int

    Has public member variables that can be overwritten in inherited classes

    .. py:attribute:: basis

        ``list`` of basis elements, each a ``list`` of ``list``

        Default values are scaled Pauli matrices

        .. math::

            \sigma_\\alpha/\sqrt{2} \\text{    for } \\alpha=\{I,x,y,z\}

        So, e.g., by default ``basis[0]=[[1/sqrt(2),0],[0,1/sqrt(2)]]``

    .. py:attribute:: c_const

       ``list`` of ``lambda c,dc`` that are functions of control
       and that define the constraints on the control

    .. py:attribute:: control_dim

        Integer number of control inputs for single qubit control

    .. py:attribute:: error_rate

        Error rate for photonics PMDs



    .. py:attribute:: lindblad_operators

        ``list`` of ``list`` where the inner list indexes coefficients on ``basis``
        elements for each Lindblad operator.  Denote ``lindblad_operators[i][j]``,
        ``lindblad_rates[i]``, and ``basis[j]`` by

        .. math::
            \ell_{i,j}\,,\lambda_{i}\,,\\text{ and }\sigma_{j}

        respectively.  Then, let

        .. math::

            L_i = \sum_{j=0}^3\ell_{i,j}\sigma_j

        and the Lindblads interact with the state via

        .. math::

            \\frac{d\\rho}{dt}=-\\frac{i}{\hbar}\left[\mathcal{H},\\rho\\right]
            + \sum_{j=1}^{K} \lambda_j\left(L_j\\rho
            L_j^{H}-\\frac{1}{2}L_j^{H}L_j\\rho-\\frac{1}{2}\\rho
            L_j^HL_j\\right)\,.

        where *K* = ``len(lindblad_rates)``

    .. py:attribute:: lindblad_rates

       ``list`` of constants that scale the Lindblad operators, see
       ``lindblad_operators``


    .. py:attribute:: loss_rate

        Photon loss rate for photonics PMDs

    .. py:attribute:: noise_dim

        Integer number of noise sources in the ``single_qubit_control``,
        ``single_qubit_drift``, and ``single_qubit_noise`` functions

    .. py:attribute:: noise_hints

        An array of of size ``noise_dim`` containing dictionaries that define
        the type, amplitude, and cutoffs of each noise, intended to be used
        to contain information about the noises that would be otherwise
        cumbersome to parse from the spectral densities or autocorrelations.
        The following keys are currently used:

            - ``type`` - the noise type/color.  Currently, only the following are
              supported in ``pmd_convert.py``

              - ``pink`` (1/f) spectra
              - ``white`` (constant) spectra
              - ``gauss`` spectrum defined by quadratic exponential
              - ``time`` autocorrelation specified.  Requires ``corrfn``

              See also ``amp``, ``corrfn``, ``omega_min``, and ``omega_max``
            - ``amp`` - the amplitude (scalar multiplier) of the noise source
            - ``corrfn`` - ``string`` defining a ``lambda`` function of a single
              variable that defines the autocorrelation function of the noise
              when ``eval``-ed.
            - ``omega_min`` - the lower cutoff frequency (in rad/s) of the
              noise spectrum
            - ``omega_max`` - the upper cutoff frequency (in rad/s) of the
              noise spectrum
            - ``ham_value`` - a value that specifies a linear or nonlinear
              interaction with the noise source.  Currently, this is only used
              in ``APL_to_BBN`` class during cumulant simulation.A value of
              ``0`` means that the interaction is nonlinear, and the key
              ``linear_ham`` is used to define the linearization for the noise
              interaction.  All other values serve as multipliers to the noise
              Hamiltonian, and will generally be one.  This is really best
              explained by the example in ``implemenations.SCQ`` which has phase
              noise that goes through a nonlinearity, as well as linear noise
            - ``linear_ham`` - if ``ham_value`` is 0, the user needs to specify
              a method for linearizing the noise so that it is linear in ``basis``.
              This is accomplished by passing a ``string`` that when ``eval``-ed
              defines a ``lambda`` function that is a function of control and
              noise, and returns a list of ``len(basis)`` that represent the
              coefficients to ``basis`` to determine the linear response.  Note
              that these coefficients generally need to be scaled by ``1/sqrt(2)``
              to be consistent with the default basis, and that you can use
              functions from ``numpy`` by using ``np``. For examples, the
              superconducting qubit phase noise is linearized by
              ``lambda c, gamma: [0,c[1]/np.sqrt(2), -c[0]/np.sqrt(2), 0]``.



    .. py:attribute:: noise_spectral_density

        ``list`` of ``list`` of ``lambda w`` that define the spectral densities
        of the various one qubit (and two qubit, if applicable) noise sources,
        in the frequency domain.  The enumeration of the single qubit noise
        sources is the same as specified in ``single_qubit_control`` and
        ``single_qubit_noise`` .  For the two qubit case, the enumeration is
        over the first single-qubit noise (``0:noise_dim-1``), then the second
        single-qubit noise (``noise_dim:2*noise_dim-1``), and then finally the
        two qubit noises, as enumerated by ``two_qubit_control`` and
        ``two_qubit_noise``, (``2*noise_dim:2*noise_dim+two_noise_dim-1``). If
        a spectral density is not defined, it will be denoted as ``None``.
        Typically, only ``noise_spectral_density`` or
        ``noise_time_correlation`` will be defined for a given PMD
        implementation.

        In other words,

            ``noise_spectral_density[i][j]``

        is a function of a single variable that defines the cross-spectral
        density between noise sources ``i`` and ``j`` , and

            ``noise_spectral_density[i][j](w)``

        evaluates it at the frequency ``w``, in Hz.

        As an example, consider a ``PMD`` instance with ``num_qubits=2``
        where the single qubit control has three noise sources and the two
        qubit control requires two.  Thus ``noise_spectral_density`` will be
        an 8 by 8 array of function in ``w`` (i.e., frequency).  So, e.g.,

            ``noise_spectral_density[4][4]`` is a function handle for the
            spectral density of the second noise source on the second qubit's
            single qubit Hamiltonian.  This handle could be passed to other
            functions (e.g., ``map`` ) to perform other useful operations.

            ``noise_spectral_density[0][2](1e9)`` is the cross-spectral density
            between the first qubit's first and third noise sources evaluated
            at 1 GHz

            ``noise_spectral_density[3][7](1e6)`` is the cross-spectral density
            between the second qubit's first noise source and the second noise
            source on the two qubit control evaluated at 1 MHz


    .. py:attribute:: noise_time_correlation

        Identical in structure to ``noise_spectral_density``, except that
        ``noise_time_correlation[i][j]`` is a ``lambda`` function in ``t`` that
        represents the cross-correlation (in seconds) between noise processes
        ``i`` and ``j``.  If a time correlation is not defined, it will be
        denoted as ``None``.  Typically only ``noise_spectral_density`` or
        ``noise_time_correlation`` will be defined for a given PMD implementation.


    .. py:attribute:: processmatrices

       ``dict`` whose elements are ``list`` of ``list`` that define process
       matrices specified by the dictionary's key


    .. py:attribute:: single_qubit_control

        ``list`` of ``list`` of ``lambda c, Gamma`` that define scalar
        multiples for control to ``basis`` first index iterates over control
        dimensions, second index over ``basis`` elements.  Multiplicative noise
        sources are specified by the ``lambda`` argument ``Gamma``.  See
        ``eval_1_qubit_hamiltonian``


    .. py:attribute:: single_qubit_drift

        ``list`` of ``list`` of ``lambda c,Gamma`` that define scalar multiples
        for drift to ``basis`` first index iterates over control dimensions,
        second index over ``basis`` elements. See ``eval_1_qubit_hamiltonian``

    .. py:attribute:: single_qubit_noise

        ``list`` of ``list`` of ``lambda c,Gamma`` that define scalar multiples
        for additive noise to ``basis`` first index iterates over control
        dimensions, second index over ``basis`` elements.  See
        ``eval_1_qubit_hamiltonian``

    .. py:attribute:: single_qubit_filter_den_coeffs

        Denote ``single_qubit_filter_den_coeffs`` by *a* and
        ``single_qubit_filter_num_coeffs`` by *b*.  *a* and *b* are ``lists``
        of ``lists`` of real numbers where the first index iterates over
        control input, and the second index defines coefficients for a rational
        polynomial in 1/*z* that defines a filter function in the *z* transform
        domain to be convolved with the control input prior to evaluation of
        the Hamiltonian.


        Mathematically,


        Denote ``single_qubit_filter_den_coeffs[i][j]`` and
        ``single_qubit_filter_num_coeffs[i][j]`` by

        .. math::
            b_{i,j}\\text{ and }a_{i,j}

        respectively.  Then the *i* th control input is filtered
        by a filter with the transfer function

        .. math::

            G_i(z)=\\frac{b_{i,0}+b_{i,1}z^{-1}+
            \cdots+b_{i,N-1}z^{N-1}}{a_{i,0}+a_{i,1}z^{-1}+\cdots+a_{i,M-1}z^{M-1}}

        where *N* and *M* are the lengths of
        ``single_qubit_filter_den_coeffs[i]`` and
        ``single_qubit_filter_den_coeffs[i]``, respectively.  The default
        filter is

        .. math::
            G_i(z)=G(z)=1.


    .. py:attribute:: single_qubit_filter_num_coeffs.

        see ``single_qubit_filter_den_coeffs``


    .. py:attribute:: time_step

        Defines the time step of control (assumed identical for all control
        dimensions).  Default value is 1 ns.

    .. py:attribute:: two_c_const

        like ``c_const``, but for the two qubit controls

    .. py:attribute:: two_control_dim

        Integer number of control inputs in ``two_qubit_control``

    .. py:attribute:: two_noise_dim

        Integer number of noise sources in the ``two_qubit_control``,
        ``two_qubit_drift``, and ``two_qubit_noise`` functions



    .. py:attribute:: two_qubit_control

        ``list`` of ``list``  of ``list`` of ``lambda c, Gamma`` that define
        scalar multiples for two-qubit control over Kronecker products of
        ``basis`` elements.  The first index iterates over control dimensions,
        second index over ``basis`` elements, and third over ``basis``
        elements.  See ``eval_2_qubit_hamiltonian``

    .. py:attribute:: two_qubit_drift

        ``list`` of ``list``  of ``list`` of ``lambda c, Gamma`` that define
        scalar multiples for two-qubit drift  over Kronecker products of
        ``basis`` elements.  The first index iterates over control dimensions,
        second index over ``basis`` elements, and third over ``basis``
        elements. See ``eval_2_qubit_hamiltonian``


    .. py:attribute:: two_qubit_noise

        ``list`` of ``list``  of ``list`` of ``lambda c, Gamma`` that define
        scalar multiples for two-qubit noise  over Kronecker products of
        ``basis`` elements.  The first index iterates over control dimensions,
        second index over ``basis`` elements, and third over ``basis``
        elements. See ``eval_2_qubit_hamiltonian``

    .. py:attribute:: two_qubit_filter_den_coeffs

        as ``single_qubit_filter_den_coeffs`` but for the two qubit controls

    .. py:attribute:: two_qubit_filter_num_coeffs

        as ``single_qubit_filter_num_coeffs`` but for the two qubit controls


    .. py:attribute:: two_qubit_lindblad_operators

        ``list`` of ``list`` of ``list`` where the second and third lists
        index coefficients on Kronecker products of ``basis`` elements for
        each Lindblad oprrator.  Denote
        ``two_qubit_lindblad_operators[i][j][k]``,
        ``two_qubit_lindblad_rates[i]``, and ``basis[j]`` by

        .. math::

            \ell_{i,j,k}^{(2)}, \lambda_{i}^{(2)}, \\text{ and }
            \sigma_j

        respectively.  Then, let

        .. math::

            L^{(2)}_i = \sum_{j=0}^3\sum_{k=0}^3\ell^{(2)}_{i,j,k}\sigma_j
            \otimes\sigma_k


        and the Lindblads interact with the state via

        .. math::

            \\frac{d\\rho}{dt} = -\\frac{i}{\hbar}\left[\mathcal{H},\\rho\\right]
            +\left( \sum_{j=1}^{K} \lambda_j\left(L_j\\rho
            L_j^{H}-\\frac{1}{2}L_j^{H}L_j\\rho-\\frac{1}{2}\\rho
            L_j^HL_j\\right)\\right)\otimes I_2

            +
            I_2\otimes\left( \sum_{j=1}^{K} \lambda_j\left(L_j\\rho
            L_j^{H}-\\frac{1}{2}L_j^{H}L_j\\rho-\\frac{1}{2}\\rho
            L_j^HL_j\\right)\\right)

            + \sum_{j=1}^{K_2} \lambda^{(2)}_j\left(L_j^{(2)}\\rho
            L_j^{(2)H}-\\frac{1}{2}L_j^{(2)H}L_j^{(2)}\\rho-
            \\frac{1}{2}\\rho L_j^{(2)H}L_j^{(2)}\\right)

        where

        .. math::

            K, L_j, \lambda_j

        are as in the documentation for ``lindblad_operators``,

        .. math::

            I_2

        is the two-dimensional identity matrix, and

        .. math::

            K_2


        is the length of ``two_lindblad_rates``.



    .. py:attribute:: two_qubit_lindblad_rates

        as ``lindblad_rates`` but scale the operators defined by
        ``two_qubit_lindblad_operators`` on Kronecker products of ``basis``
        elements.  See ``two_qubit_lindblad_operators``


    """

    __metaclass__ = MultiQubitPreBuilder

    def __init__(self, control_dim, noise_dim, num_lindblad, num_qubits=1, two_control_dim=0, two_noise_dim=0, num_two_lindblad=0):

        self.num_qubits = num_qubits
        self.control_dim = control_dim
        self.noise_dim = noise_dim
        self.num_lindblad = num_lindblad

        self.two_control_dim = two_control_dim
        self.two_noise_dim = two_noise_dim
        self.num_two_lindblad = num_two_lindblad

        # default basis
        self.basis = [s_i, s_x, s_y, s_z]

        #Initialize 'f' portion of Hamiltonian
        self.single_qubit_control = [[0 for x in range(len(self.basis))] for y in range(self.control_dim)]
        for x in range(self.control_dim):
            for y in range(len(self.basis)):
                self.single_qubit_control[x][y] = lambda c, gamma: None

        self.single_qubit_noise=[0 for x in range(len(self.basis))]
        self.single_qubit_drift=[0 for x in range(len(self.basis))]
        for x in range(len(self.basis)):
            self.single_qubit_noise[x] = lambda c, gamma: None
            self.single_qubit_drift[x] = lambda c, gamma: None

        self.single_qubit_filter_num_coeffs=[[1.0] for y in range(self.control_dim)]
        self.single_qubit_filter_den_coeffs=[[1.0] for y in range(self.control_dim)]

        #Initialize spectral density matrix
        self.noise_spectral_density=[[0 for x in range(self.noise_dim)] for y in range(self.noise_dim)]
        for x in range(self.noise_dim):
            for y in range(self.noise_dim):
                self.noise_spectral_density[x][y] = None

        self.noise_time_correlation=[[0 for x in range(self.noise_dim)] for y in range(self.noise_dim)]
        for x in range(self.noise_dim):
            for y in range(self.noise_dim):
                self.noise_time_correlation[x][y] = None

        #Initialize an empty list of noise hints
        self.noise_hints=[{} for x in range(self.noise_dim)]

        #Generate an empty list of control constraints
        self.c_const=list()

        #Generate a zero crosstalk matrix
        self.xtalk_dim = self.num_qubits*(self.control_dim+self.noise_dim)
        self.xtalk = np.zeros((self.xtalk_dim,self.xtalk_dim))

        #Lindblads
        self.lindblad_rates = list()
        self.lindblad_operators = list()
        for x in range(self.num_lindblad):
            self.lindblad_rates.append(0.0)
            self.lindblad_operators.append([0.0, 0.0, 0.0, 0.0])

        if is_multi_qubit(self.num_qubits):
            self._init_multi_qubit()

    def _init_multi_qubit(self):

        self.tot_noise_dim = self.num_qubits*self.noise_dim + int(self.num_qubits*(self.num_qubits - 1)/2)*(self.two_noise_dim)

        #redo cross talk for multiqubit case (eventually could merge both parts)
        self.xtalk_dim = self.num_qubits*(self.control_dim+self.noise_dim)
        self.xtalk_dim += int(self.num_qubits*(self.num_qubits - 1)/2)*(self.two_control_dim+self.two_noise_dim)

        self.xtalk = np.zeros((self.xtalk_dim,self.xtalk_dim))

        #Two qubit interaction function
        self.two_qubit_control =[[[0 for x in range(4)] for y in range(4)] for z in range(self.two_control_dim)]
        for x in range(self.two_control_dim):
            for y in range(len(self.basis)):
                for z in range(len(self.basis)):
                    self.two_qubit_control[x][y][z]=lambda c, gamma: None

        self.two_qubit_filter_num_coeffs=[[1.0] for y in range(self.two_control_dim)]
        self.two_qubit_filter_den_coeffs=[[1.0] for y in range(self.two_control_dim)]

        #Make noise_spectral_density include noise from both qubits individually and interaction noise
        self.noise_spectral_density=[[0 for x in range(self.tot_noise_dim)] for y in range(self.tot_noise_dim)]
        for x in range(self.tot_noise_dim):
            for y in range(self.tot_noise_dim):
                self.noise_spectral_density[x][y]=None

        self.noise_time_correlation=[[0 for x in range(self.tot_noise_dim)] for y in range(self.tot_noise_dim)]
        for x in range(self.tot_noise_dim):
            for y in range(self.tot_noise_dim):
                self.noise_time_correlation[x][y]=None

        self.noise_hints=[{} for x in range(self.tot_noise_dim)]

        self.two_c_const=list()

        self.two_lindblad_rates = list()
        self.two_lindblad_operators = list()

        for x in range(self.num_two_lindblad):
            self.two_lindblad_rates.append(0.0)
            self.two_lindblad_operators.append(np.zeros([4,4]).tolist())

    def build_out_multi_qubit(self):
        """

        Takes the noise specifications for a two qubit PMD and expands them to
        an n-qubit PMD.  PMD implementations originally specified the noise
        characteristics for two single qubits and the two qubit interaction
        noises.  This function takes a PMD with those already specified and
        "krons" them up to the number of qubits specified in ``num_qubits``.
        This is consistent with the PMD specification defining only one and two
        qubit actions.

        Specifically, ``noise_spectral_density``, ``noise_time_correlation``,
        and ``noise_hints`` are extended, but variables related to cross talk
        are not, as these are intended to be specified by the user to conform
        to some geometric configuration.

        """
        #TODO, this will not work with the general cross spectral density case

        if not is_multi_qubit(self.num_qubits):
            raise RuntimeError("Illegal call to build_out_multi_qubit() on single qubit system")

        #Not sure if need to copy, but safer this way...
        orig_spectral = copy.copy(self.noise_spectral_density)
        orig_time = copy.copy(self.noise_time_correlation)
        orig_hints = copy.copy(self.noise_hints)

        #Expand single qubit noise correlation matrix and noise hints
        for i in range(1,self.num_qubits):
            for j in range(self.noise_dim):
                for k in range(self.noise_dim):
                    self.noise_spectral_density[i*self.noise_dim+j][i*self.noise_dim+k] = copy.copy(orig_spectral[j][k])
                    self.noise_time_correlation[i*self.noise_dim+j][i*self.noise_dim+k] = copy.copy(orig_time[j][k])

                self.noise_hints[i*self.noise_dim+j] = copy.copy(orig_hints[j])

        noise_idx = self.num_qubits*self.noise_dim
        #Assume noise coming in is fully specified for two qubits for backwards compatibility
        noise_orig = 2*self.noise_dim#self.num_qubits*self.noise_dim
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                for k in range(self.two_noise_dim):
                    for ell in range(self.two_noise_dim):
                        self.noise_spectral_density[noise_idx+k][noise_idx+ell] = copy.copy(orig_spectral[noise_orig+k][noise_orig+ell])
                        self.noise_time_correlation[noise_idx+k][noise_idx+ell] = copy.copy(orig_time[noise_orig+k][noise_orig+ell])
                    self.noise_hints[noise_idx+k] = copy.copy(orig_hints[noise_orig+k])
                noise_idx+=self.two_noise_dim

    def noise_off(self):
        """Replace noise hints with null noise (zeros)"""
        self.noise_hints = [{'type': 'null'} for x in range(len(self.noise_hints))]