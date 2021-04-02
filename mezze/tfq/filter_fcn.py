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

import tensorflow as tf
import numpy as np
import scipy.signal as si
import cirq
from mezze.tfq.helpers import *

def get_FTTS_FFs(N=128,worN=8192):
    freqs = np.linspace(0, np.pi, N // 2 + 1)[:-1]

    mod_funs = []
    for i, ww in enumerate(freqs):
        mod_fun = np.sign(np.cos(ww * np.arange(1, N + 1)))
        mod_fun[mod_fun == 0] = 1
        mod_funs.append(mod_fun)

    # This is the filter function matrix for predictions
    Phi = np.array([np.abs(np.fft.fft(mf, n=worN)) ** 2 / (.5 * worN) for mf in mod_funs])

    # For reconstruction
    PhiRecon = np.array([np.abs(np.fft.fft(mf, N)) ** 2 / (.5 * N) for mf in mod_funs])
    PhiRecon = PhiRecon[:, :N // 2]
    PhiRecon[1:, :] = PhiRecon[1:, :] * 2

    num_gates = np.array([np.sum(np.abs((mf[1:] + 1) / 2 - (mf[:-1] + 1) / 2)) for mf in mod_funs])[:, np.newaxis]

    return Phi, PhiRecon, num_gates

def get_FTTS_circuits(N):
    freqs = np.linspace(0, np.pi, N // 2 + 1)[:-1]

    q = cirq.GridQubit.rect(1, 1)
    mod_funs = []
    for i, ww in enumerate(freqs):
        mod_fun = np.sign(np.cos(ww * np.arange(1, N + 1)))
        mod_fun[mod_fun == 0] = 1
        mod_fun = (mod_fun + 1) / 2
        mod_funs.append(mod_fun)

    circuits = []
    for mod_fun in mod_funs:
        circuit = cirq.Circuit()
        circuit.append(cirq.rx(np.pi / 2).on(q[0]))

        flip = np.concatenate(([0], np.abs(mod_fun[1:] - mod_fun[:-1])))
        for f in flip:
            if f == 1:
                circuit.append(cirq.rx(np.pi).on(q[0]))
            else:
                circuit.append(cirq.I(q[0]))

        if np.mod(np.sum(flip), 2) == 1:
            circuit.append(cirq.rx(np.pi / 2).on(q[0]))
        else:
            circuit.append(cirq.rx(-np.pi / 2).on(q[0]))
        circuits.append(circuit)
    return circuits



class SPAM_constraint(tf.keras.constraints.Constraint):

    def __call__(self, w):
        return w * tf.cast(tf.math.logical_and(tf.math.greater_equal(w, 0.0), tf.math.less_equal(w, 0.5)), tf.float32)


class SchWARMAZDephasingFF(tf.keras.layers.Layer):
    def __init__(self, b_dim, a_dim_less_first, w, A_var=False, B_var=False, c1_var=False,
                 c2_var=False, b_back=[0, ], a_back=[1, ], **kwargs):
        self.b_dim = b_dim
        self.a_dim_less_first = a_dim_less_first
        self.a_dim = a_dim_less_first + 1
        self.A_var = A_var
        self.B_var = B_var
        self.c1_var = c1_var
        self.c2_var = c2_var
        self.w = w

        if b_back is None:
            self.b_back = [0, ]
        else:
            self.b_back = b_back

        if a_back is None:
            self.a_back = [1, ]
        else:
            self.a_back = a_back

        if type(w) == int:
            w, _ = si.freqz([.1], [1], worN=w, whole=True)
        self.iw = tf.constant(-1j * w, dtype=tf.complex64)
        rb = tf.constant(np.arange(self.b_dim), dtype=tf.complex64)
        ra = tf.constant(np.arange(self.a_dim), dtype=tf.complex64)
        ebw = tf.exp(tf.tensordot(self.iw, rb, axes=0))
        eaw = tf.exp(tf.tensordot(self.iw, ra, axes=0))
        self.rbw = tf.math.real(ebw)
        self.ibw = tf.math.imag(ebw)
        self.raw = tf.math.real(eaw)
        self.iaw = tf.math.imag(eaw)

        super(SchWARMAZDephasingFF, self).__init__(**kwargs)
        self.build((1,))

    def background_psd(self):

        # aa = tf.concat([[1], tf.multiply(-1.0, self.a_back_less_first)],axis=0).numpy()

        _, h = si.freqz(self.b_back, self.a_back, worN=self.w, whole=True)

        return tf.constant(np.abs(h) ** 2, dtype=tf.float32)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.b = self.add_weight(name='b', shape=(self.b_dim,), dtype=tf.float32,
                                 initializer='normal', trainable=True)
        self.a_less_first = self.add_weight(name='a_less_first', dtype=tf.float32,
                                            shape=(self.a_dim_less_first,),
                                            initializer='normal', trainable=True)

        self.A = self.add_weight(name='A', shape=(1,),
                                 initializer=tf.keras.initializers.Constant(value=.5), trainable=self.A_var,
                                 constraint=SPAM_constraint(), dtype=tf.float32)
        self.B = self.add_weight(name='B', shape=(1,),
                                 initializer=tf.keras.initializers.Constant(value=.5), trainable=self.B_var,
                                 constraint=SPAM_constraint(), dtype=tf.float32)
        self.c1 = self.add_weight(name='c1', shape=(1,),
                                  initializer=tf.keras.initializers.Constant(value=0), trainable=self.c1_var,
                                  constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)
        self.c2 = self.add_weight(name='c2', shape=(1,),
                                  initializer=tf.keras.initializers.Constant(value=0), trainable=self.c2_var,
                                  constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)

        self.bgound = self.background_psd()

        super(SchWARMAZDephasingFF, self).build(input_shape)

    def call(self, inputs):
        return self.A + self.B * tf.exp(
            -tf.tensordot(inputs[0], tf.expand_dims(self.bgound + self.psd(), 1), axes=[[1], [0]])
            - tf.multiply(self.c1, inputs[1])
            - tf.multiply(self.c2, tf.square(inputs[1])))

    def psd(self):
        aa = tf.concat([[1], tf.multiply(-1.0, self.a_less_first)], axis=0)
        # rb = tf.range(self.b_dim,dtype=tf.float64)
        # ra = tf.range(self.a_dim,dtype=tf.float64)

        num = tf.add(tf.square(tf.abs(tf.reduce_sum(tf.multiply(self.b, self.rbw), axis=1))),
                     tf.square(tf.abs(tf.reduce_sum(tf.multiply(self.b, self.ibw), axis=1))))
        den = tf.add(tf.square(tf.abs(tf.reduce_sum(tf.multiply(aa, self.raw), axis=1))),
                     tf.square(tf.abs(tf.reduce_sum(tf.multiply(aa, self.iaw), axis=1))))
        # den = tf.square(tf.abs(tf.reduce_sum(tf.multiply(aa,tf.exp(aw)),axis=1)))
        return num / den

    def convert_to_lfilter_form(self):
        a = tf.concat([[1.0], tf.multiply([-1.0], self.a_less_first)], axis=0)
        b = self.b
        return b.numpy(), a.numpy()

    def set_coeffs(self, b=None,a=None,A=None,B=None,c1=None,c2=None):
        weights = self.get_weights()

        if b is not None:
            weights[0] = np.array(b)
        if a is not None:
            weights[1] = np.array(-a[1:])
        if A is not None:
            weights[2] = np.array(A)
        if B is not None:
            weights[3] = np.array(B)
        if c1 is not None:
            weights[4] = np.array(c1)
        if c2 is not None:
            weights[5] = np.array(c2)
        self.set_weights(weights)

def compute_gen_dephasing_Fourier(circ: cirq.Circuit, worN):

    Clist = cirq_moments_to_channel_list(circ)
    PTMProp = compute_PTM_prop(Clist)

    FFs = np.fft.fft(PTMProp, worN, axis=0)
    return FFs

def compute_gen_dephasing_FF(circ: cirq.Circuit, worN):

    FFs = compute_gen_dephasing_Fourier(circ, worN)
    
    F_zx = np.abs(FFs[:worN,2,0])**2
    F_zy = np.abs(FFs[:worN,2,1])**2

    return (F_zx+F_zy)/worN/2

def compute_Z_dephasing_prob(circ: cirq.Circuit, S):

    w, P = S.psd()
    worN = len(w)
    FFs = compute_gen_dephasing_Fourier(circ, worN=worN*2)

    F_zx = np.abs(FFs[:worN,2,0])**2
    F_zy = np.abs(FFs[:worN,2,1])**2

    oi = np.sum(P*(F_zx+F_zy))/worN

    return .5+.5*np.exp(-oi/2)