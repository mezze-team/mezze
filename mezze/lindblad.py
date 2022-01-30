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
import logging

# Define unnormalized Pauli Matrices
sigI = np.array([[1., 0], [0, 1.]],dtype=complex)
sigX = np.array([[0, 1.], [1., 0]],dtype=complex)
sigY = np.array([[0, -1.j], [1.j, 0]],dtype=complex)
sigZ = np.array([[1., 0], [0, -1.]],dtype=complex)

class LindbladFactory(object):
    """
    Factory for getting Lindblad calculator based on a PMD
    """
    @classmethod
    def get(cls, pmd):
        lindblad = Lindblad(pmd.num_qubits)
        if len(pmd.lindblad_rates) >= 1:
            lindblad.add_with_t1(1.0 / pmd.lindblad_rates[0])
        if len(pmd.lindblad_rates) >= 2:
            lindblad.add_with_t2(1.0 / pmd.lindblad_rates[1])
        return lindblad

class Lindblad(object):
    """
    Lindblad calculator.
    """
    def __init__(self, num_qubits):
        self.logger = logging.getLogger(__name__)
        self.num_qubits = num_qubits
        self.operator = np.zeros([4**num_qubits, 4**num_qubits])
        self.identity = np.array(np.eye(2**num_qubits,dtype=complex))

    def add_with_t1(self, t1_time):
        rate = 1.0 / t1_time
        sigma_minus = np.array([[0, 1.0], [0, 0]])
        if self.num_qubits == 1:
            self.add_lindblad(sigma_minus, rate)
        if self.num_qubits == 2:
            self.add_lindblad(np.kron(sigI, sigma_minus), rate)
            self.add_lindblad(np.kron(sigma_minus, sigI), rate)

    def add_with_t2(self, t2_time):
        rate = 1.0 / t2_time
        if self.num_qubits == 1:
            lindblad_operator = sigZ
            self.add_lindblad(sigZ, rate)
        if self.num_qubits == 2:
            self.add_lindblad(np.kron(sigI, sigZ), rate)
            self.add_lindblad(np.kron(sigZ, sigI), rate)

    def add_lindblad(self, operator, scale_factor):
        """
        Update the lindblad operator
        """
        lindblad_operator = np.array(operator)
        update = np.kron(lindblad_operator.conj(), lindblad_operator) \
                                - 0.5 * np.kron(self.identity, np.dot(lindblad_operator.conj().T, lindblad_operator)) \
                                - 0.5 * np.kron(np.dot(lindblad_operator.conj().T, lindblad_operator).T, self.identity)
        self.operator = self.operator + scale_factor * update

    def get_matrix(self):
        """
        Get the matrix for the Lindblad superoperator
        """
        return self.operator

    def set_matrix(self, operator):
        """
        Set the matrix for the Lindblad operator
        """
        self.operator = operator