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

#import tensorflow_quantum as tfq
#import tensorflow as tf
import numpy as np
import scipy.signal as si
import sympy
import cirq  
import mezze.channel as ch

def cirq_to_total_channel(circ: cirq.Circuit):
    
    return ch.QuantumChannel(circ.unitary(), 'unitary')

def cirq_moments_to_channel_list(circ: cirq.Circuit):

    channel_list = []

    for m in circ:
        m_circ = cirq.Circuit(m)
        for qbit in circ.all_qubits():
            if qbit not in m_circ.all_qubits():
                m_circ.append(cirq.I.on(qbit))
            
        channel_list.append(cirq_to_total_channel(m_circ))
    
    return channel_list

def channel_to_circuit(C: ch.QuantumChannel, qbits = None):

    if qbits is None:
        num_qubits = int(np.log2(C.kraus()[0].shape[0]))
        qbits = cirq.GridQubit.rect(num_qubits,1)
        
    return cirq.Circuit(cirq.MatrixGate(C.kraus()[0]).on(*qbits))

def channel_list_to_circuit(Clist, qbits = None):

    if qbits is None:
        num_qubits = int(np.log2(Clist[0].kraus()[0].shape[0]))
        qbits = cirq.GridQubit.rect(num_qubits,1)

    return cirq.Circuit([channel_to_circuit(C, qbits) for C in Clist])

    
def compute_PTM_prop(Clist):
    PTMProp = [Clist[0].ptm()[1:,1:]]
    for i in range(1,len(Clist)):
        PTMProp.append(Clist[i].ptm()[1:,1:]@PTMProp[-1])

    return np.array(PTMProp)
    