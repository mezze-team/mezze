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
import mezze.tfq.filter_fcn as ff
import scipy.optimize as op
import numpy as np
import scipy.linalg as la

class ZDephasingFFLearner(object):
    def __init__(self, b_len, a_dim_less_first,worN=8192,A_var=False, B_var=False, c1_var=False,
                 c2_var=False, b_back=[0, ], a_back=[1, ]):
        self.b_len = b_len
        self.a_dim_less_first = a_dim_less_first

        ff_input = tf.keras.Input(shape=(None,), dtype=tf.float32, name='ff_input')
        count_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='count_input')

        schwarma = ff.SchWARMAZDephasingFF(b_len, a_dim_less_first, worN, A_var, B_var, c1_var, c2_var,b_back, a_back)(
            [ff_input, count_input])

        self.model = tf.keras.Model(inputs=[ff_input, count_input], outputs=schwarma)

    def NNLS_recon(self, ps, PhiRecon):
        chis = -np.log(2 * (ps - .5))
        return op.nnls(PhiRecon, np.squeeze(chis))[0]

    def YW_est(self, ps, PhiRecon, a_dim_less_first, overdetermined=False):
        S = self.NNLS_recon(ps, PhiRecon)
        Sfull = np.concatenate([S, S[1:][::-1]])
        R = np.real(np.fft.fft(Sfull)[:len(ps)])

        if not overdetermined:
            ahat = np.concatenate([[1], -la.pinv(la.toeplitz(R[:a_dim_less_first])) @ R[1:1+a_dim_less_first]])
        else:
            ahat = np.concatenate([[1], -la.pinv(la.toeplitz(R)[:-1, :a_dim_less_first]) @ R[1:]])
        bhat = np.array([R[0] - np.sum(-ahat[1:] * R[1:1+a_dim_less_first])])

        return bhat, ahat

    def MUSIC_est(self, ps, PhiRecon, dim):
        S = self.NNLS_recon(ps, PhiRecon)
        Sfull = np.concatenate([S, S[1:][::-1]])
        R = np.real(np.fft.fft(Sfull)[:len(ps)])

        vals,vecs = la.eigh(la.toeplitz(R))

        P = lambda w, v: 1. / np.sum(
            [np.abs(np.exp(1j * np.arange(len(ps)) * w) @ v[:, i]) ** 2 for i in range(v.shape[1])])
        return lambda w: P(w, vecs[:,:-dim])

    def set_coeffs(self, **kwargs):
        self.model.layers[-1].set_coeffs(**kwargs)

    def fit(self, ps, Phi, num_gates, learning_rates=[0.01,.001, .0001], epochs=1000, optim=tf.optimizers.Adam):
        learning_rates = np.array(learning_rates)

        if len(learning_rates.shape) == 0:
            learning_rates = np.array([learning_rates])

        earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=epochs, restore_best_weights=True)

        for learning_rate in learning_rates:
            self.model.compile(optimizer=optim(learning_rate), loss='mean_squared_error')
            self.model.fit(x=[Phi, num_gates], y=ps, epochs=epochs, verbose=0, callbacks=[earlyStop])
            self.model.set_weights(earlyStop.best_weights)

        #weights = self.model.get_weights()
        #if weights[0][0] < 0:
        #    weights[0][0] *= -1
        #    self.model.set_weights(weights)
        return earlyStop.best

    def convert_to_lfilter_form(self):
        return self.model.layers[-1].convert_to_lfilter_form()
    
    def akaike_info_criterion(self, ps, Phi, num_gates):

        if len(ps.shape)==1:
            ps=ps[:,np.newaxis]

        k = np.sum([np.prod(w.shape) for w in self.model.layers[-1].trainable_weights])
        RSS = np.sum((ps-self.model([[Phi, num_gates]]))**2)

        return 2*k+len(ps)*np.log(RSS)