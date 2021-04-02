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

import numpy as np
import scipy.linalg as la
from ...channel import *
from ...channel import _sigmaI, _sigmaX, _sigmaY, _sigmaZ
import copy

import scipy.optimize as opt
from scipy.special import factorial
import scipy.signal as si

"""
 This is experimental code. Use at your own risk.
"""


class ARMA(object):
    
    def __init__(self,a=[],b=[1]):
        self.a = np.array(a)
        self.b = np.array(b)
        self.x = np.zeros(len(b))
        self.y = np.zeros(len(a)+1)
        
    def step(self,x_in):
        
        try:
            self.x = np.concatenate((x_in,self.x[:-1]))
        except ValueError:
            self.x = np.concatenate(([x_in],self.x[:-1]))

        y_out = np.sum(self.a*self.y[:-1])+np.sum(self.b*self.x)
        
        self.y = np.concatenate(([y_out],self.y[:-1]))
        return y_out


class VARMA(ARMA):

    def __init__(self, a = [[]], b=[[1]]):
        self.armas = [ARMA(aa, bb) for aa, bb in zip(a, b)]

    def step(self,x_in):
        return [self.armas.step(xx) for xx in x_in]


class correlated_error(object):

    def __init__(self, noise_type, corr_time=1., fidelity=.99, rgen=np.random):

        if noise_type in ['amp_damping', 'depolarizing', 'zdephasing', 'zrot', 'xyzrot']:
            # Can easily add more types later -- or recognize that zdephasing and zrot should
            # have essentially the same inputs to the simulator
            self.noise_type = noise_type

        else:
            raise NotImplementedError

        self.corr_time = corr_time
        self.fidelity = fidelity
        self.rgen = rgen

        b0 = self._fidelity_to_variance()
        b = si.gaussian(6 * int(np.ceil(corr_time)) + 1, std=corr_time / 4.)
        b = b / np.sqrt(np.sum(b ** 2)) * b0

        if noise_type == 'xyzrot' :
            self.model = VARMA([[],[],[]],[b,b,b])
            for i in range(3):
                self.model.armas[i].x=self.rgen.randn(len(self.model.armas[i].x))
        else:

            self.model = ARMA(a=[], b=b)
            self.model.x = self.rgen.randn(len(self.model.x))
        self.reset()

    def next_err(self, as_channel=True):

        ys = np.array([self.model.step(self.rgen.randn())])
        return self._error_stream_common(ys, as_channel)[0]

    def error_stream(self, n, as_channel=True):
        # This generates a stream of errors quickly using si.lfilter
        if self.noise_type == 'xyzrot':
            ys= [si.lfilter(m.b, np.concatenate([[1],-m.a]),self.rgen.randn(n+len(m.b)))[len(m.b):] for m in self.model.armas]

        else:
            ys = si.lfilter(self.model.b, np.concatenate([[1],-self.model.a]), self.rgen.randn(n + len(self.model.b)))[len(self.model.b):]

        return self._error_stream_common(ys, as_channel)

    def reset(self):
        
        if self.noise_type == 'xyzrot':
            for i in range(3):
                self.model.armas[i].x=self.rgen.randn(len(self.model.armas[i].x))
                self.model.armas[i].y = 0.0*self.model.armas[i].y
        else:
            self.model.x = self.rgen.randn(len(self.model.x))
            self.model.y = 0.0 * self.model.y

    def _error_stream_common(self, ys, as_channel=False):

        if self.noise_type == 'zrot':
            return self._ARMA_to_zrot(ys, as_channel)
        elif self.noise_type == 'zdephasing':
            return self._ARMA_to_zdephasing(ys, as_channel)
        elif self.noise_type == 'depolarizing':
            return self._ARMA_to_depolarizing(ys, as_channel)
        elif self.noise_type == 'amp_damping':
            return self._ARMA_to_amp_damping(ys, as_channel)
        elif self.noise_type == 'xyzrot':
            return self._ARMA_to_xyzrot(ys, as_channel)
        else:
            raise NotImplementedError

    def _ARMA_to_zrot(self, ys, as_channel=False):

        thetas = np.exp(1j * ys)

        if as_channel:
            return [QuantumChannel(np.diag([theta, np.conj(theta)]), 'unitary') for theta in thetas]
        else:
            return thetas

    def _ARMA_to_zdephasing(self, ys, as_channel=False):

        thetas = np.cos(ys) ** 2

        if as_channel:
            return [QuantumChannel(np.diag([theta, 0, 0, 1 - theta]), 'chi') for theta in thetas]
        else:
            return thetas

    def _ARMA_to_depolarizing(self, ys, as_channel=False):

        thetas = np.cos(np.sqrt(3) * ys) ** 2

        if as_channel:
            nonIs = (1. - thetas) / 3.
            return [QuantumChannel(np.diag([theta, nonI, nonI, nonI]), 'chi') for theta, nonI in zip(thetas, nonIs)]
        else:
            return thetas

    def _ARMA_to_amp_damping(self, ys, as_channel=False):

        if as_channel:
            return [QuantumChannel([[[1., 0.], [0, np.cos(y)]], [[0., np.sin(y)], [0., 0.]]], 'kraus') for y in ys]
        else:
            return np.sin(ys) ** 2  # gamma

    def _ARMA_to_xyzrot(self, ys, as_channel=False):

        if as_channel:
            return [QuantumChannel(la.expm(1j*(yy[0]*_sigmaX+yy[1]*_sigmaY+yy[2]*_sigmaZ)),'unitary') for yy in zip(*ys)]
        else:
            raise NotImplementedError


    def _fidelity_to_variance(self):

        if self.noise_type == 'zrot' or self.noise_type == 'zdephasing':

            #f =E[.5+.5cos(2x)] = .5+.5exp(-2*sigma^2)

            return np.sqrt(np.log(2*self.fidelity-1)/-2)

        elif self.noise_type == 'depolarizing':

            return np.sqrt(np.log(2*self.fidelity-1)/-6)

        elif self.noise_type == 'amp_damping':

            fcn = lambda x: .375+.5*np.exp(-.5*x**2)+.125*np.exp(-2*x**2)-self.fidelity
            return opt.bisect(fcn, 0, .5)

        elif self.noise_type == 'xyzrot':

            return np.sqrt(np.log(2*self.fidelity-1)/-2)/np.sqrt(3)

        else:
            raise NotImplementedError

def schwarma_trajectory(n_steps, a, b, amp = 1.0, rgen=np.random):
    xs = rgen.randn(n_steps + len(b))
    ys = amp * si.lfilter(b, a, xs)[len(b):]
    return ys

def gaussian_ma(corr_time):
    b = si.gaussian(6 * int(np.ceil(np.sqrt(2.0)*corr_time)) + 1, std=np.sqrt(2.0)*corr_time)
    b = b / (np.sqrt(2.) * corr_time)
    return b


def acv_2_var(acv,ks):
    var = np.zeros(len(ks))
    var +=ks*acv[0]
    for i in range(1,int(np.max(ks))):
        if i<len(acv)-1:
            var+=2*np.maximum(0,ks-i)*acv[i]
    return var


def var_2_fid(var, A=.5, B=.5):
    return A*np.exp(-2*var)+B


def acv_from_b(b, kmax=None, use_fft=False):
    if kmax == None:
        kmax = len(b)

    if use_fft:

        Y = np.fft.fft(b, len(b)*2)
        acv = np.real(np.fft.ifft(Y.conj()*Y))/len(b)
        acv=acv[:len(b)]

    else:

        acv = np.zeros(len(b))
        acv[0] = np.sum(b**2)
        for k in range(1,len(b)):
            acv[k] = np.sum(b[k:]*b[:-k])

    if kmax<len(b):
        return acv[:kmax]
    else:
        return np.pad(acv, (0, kmax-len(b)),'constant', constant_values=(0,0))


def extract_and_downconvert_from_sim(sim, Tgate):
    tfinal = int(np.ceil(sim.config.time_length))

    fits = []

    for rx in extract_sim_autocorrs(sim, 1. / sim.config.dt):  # mezze_noise in sim.hamiltonian.noise_sources:
        # rx = mezze_noise.corrfn(times)/dcv**2
        b_fit = down_sample_SchWMA_model(rx, int(np.ceil(Tgate / sim.config.dt)), int(np.ceil(tfinal / Tgate)))
        fits.append(b_fit)

    return fits


def make_gauss_approximation(sim, Tgauss):
    def obj_fun(x, fids, dcv):
        halfwin = int(np.maximum(x[1], int(np.floor(len(fids) / dcv))))
        b = x[0] * si.gaussian(8 * halfwin, x[1])
        fids2 = var_2_fid(acv_2_var(acv_from_b(b), np.arange(len(fids) / dcv)))
        return np.sum((fids2 - fids[::dcv]) ** 2)

    b_gausses = []

    T_max = sim.config.time_length
    dcv = Tgauss / sim.config.dt

    for ii, rx in enumerate(extract_sim_autocorrs(sim, 1. / sim.config.dt)):

        if sim.pmd.noise_hints[ii]['type'] == 'gauss':
            amp0 = sim.pmd.noise_hints[ii]['amp']
            corr_time0 = sim.pmd.noise_hints[ii]['corr_time']
        else:
            raise NotImplementedError

        x0 = [amp0 / dcv, corr_time0 / Tgauss]
        fids_fast = var_2_fid(acv_2_var(rx, ks=np.arange(T_max / sim.config.dt)))
        out = opt.least_squares(lambda x: obj_fun(x, fids_fast, int(Tgauss / sim.config.dt)), x0, max_nfev=10000)
        halfwin = int(np.maximum(out['x'][-1], int(np.floor(len(fids_fast) / dcv))))
        b_gauss = np.abs(out['x'][0]) * si.gaussian(8 * halfwin, out['x'][1])

        b_gausses.append(b_gauss)

    return b_gausses


def down_sample_SchWMA_model(rx, dcv, num_slow_samps):
    ry = np.zeros(2 * num_slow_samps)

    A = np.zeros((len(ry), len(ry)))
    C = np.zeros(len(ry))

    for L in range(1, len(ry) + 1):
        vec = np.concatenate([[L], 2 * (L - np.arange(1, L))])
        A[L - 1, :len(vec)] = vec

        vec = np.concatenate([[L * dcv], 2 * (L * dcv - np.arange(1, L * dcv))])
        vec = vec[:np.minimum(len(vec), len(rx))]
        C[L - 1] = np.sum(vec * rx[:len(vec)])

    ry = opt.lsq_linear(A, C)['x']
    Ry = np.real(np.fft.fft(np.concatenate([ry, ry[1::][::-1]])))
    Ry[Ry <= 0] = np.min(Ry[Ry > 0]) / 10.
    alpha = .5 * np.log(Ry)
    phi = np.imag(si.hilbert(alpha))
    Bmp = np.exp(alpha + 1j * phi)
    bmp = np.fft.fftshift(np.fft.ifft(Bmp))
    return np.abs(bmp)


def extract_sim_autocorrs(sim, dcv):
    tfinal = int(np.ceil(sim.config.time_length))
    times = np.arange(tfinal * dcv) / dcv

    return [mn.corrfn(times) / dcv ** 2 for mn in sim.hamiltonian.noise_sources]


def downconvert_from_SchWARMA(b, dcv, tfinal=None, a=None, outN=None):
    if tfinal is None:
        tfinal = len(b) * 2

    if a is not None:
        raise NotImplementedError

    if outN is None:
        outN = len(b)

    rx = acv_from_b(b, kmax=int(tfinal * dcv))

    return down_sample_SchWMA_model(rx, dcv, int(outN))


"""
Everything below here is pretty much outdated and is of limited utility
"""


class SchWARMA(object):
    
    def __init__(self, ARMAs, basis, D, rgen = np.random):
        
        self.models = copy.copy(ARMAs)
        self.basis = copy.copy(basis)
        self.D = copy.copy(D)
        self.rgen = rgen

        self.basis = [np.matrix(b) for b in self.basis]
        shapes = np.matrix(np.concatenate([np.matrix(b.shape) for b in self.basis],0))

        rmin = np.min(shapes[:,0])
        rmax = np.max(shapes[:,0])
        cmin = np.min(shapes[:,1])
        cmax = np.min(shapes[:,1])

        assert(cmin == cmax) #make exception eventually
        self.N = cmin

        for i in range(len(self.basis)):
            if shapes[i,0] == shapes[i,1]:
                # square matrix, check if hermitian and turn to skew-hermitian
                if la.norm(self.basis[i].H-self.basis[i])/(la.norm(self.basis[i])+ np.finfo(float).eps) < 1e-10:
                    self.basis[i] = self.basis[i]*1j

                if shapes[i,0] < rmax:
                    b = np.matrix(np.zeros((rmax,rmax),dtype=np.complex))
                    b[:shapes[i,0],:shapes[i,1]] = self.basis[i]

                    self.basis[i] = b
            else:
                # is a rectangular stiefel manifold element => make skew hermitian
                b = np.matrix(np.zeros((rmax,rmax),dtype=np.complex))
                b[:shapes[i,0],:shapes[i,1]] = self.basis[i]
                b[:self.N,self.N:shapes[i,0]] = -self.basis[i][self.N:,:].H

                self.basis[i]=b

    def init_ARMAs(self):
        for model in self.models:
            model.x = self.rgen.randn(len(model.x))
            model.y = 0.0*model.y
            
    def next_err(self):
        
        out = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            out[i] = model.step(self.rgen.randn())
        
        alg = np.sum([o*b for o,b in zip(out,self.basis)], axis=0)
        exp = la.expm(alg)
        
        return QuantumChannel(exp[:,:self.N], 'stiefel')#np.kron(np.conj(exp),exp)
    
    def __getitem__(self,index):
        if index == 'del_t':
            return self.D['del_t']
        else:
            err = self.next_err()
            gate = self.D[index]
            return err*gate

class SchWMA(SchWARMA):

    def __init__(self, MAs, basis, D, rgen = np.random):


        for MA in MAs:
            assert len(MA.a) == 0

        super(SchWMA,self).__init__(MAs, basis, D, rgen = rgen)

    def fit(self, corrs, qs, mode = 'sequential', apply_xform = True):
        qs = np.ones(len(self.basis), dtype=int)*np.array(qs)

        if apply_xform:
            corrs = [np.arcsin(np.sqrt(x)) ** 2 for x in corrs]

        assert len(qs) == len(corrs)

        self.models = []

        for corr, q in zip(corrs, qs):

            Ts = []
            for i in range(int(q)):
                vec = np.concatenate((np.arange(i+1,0,-1),np.zeros(q-i-1)))
                T = np.matrix(la.toeplitz(vec))
                Ts.append(T)
                #print T

            def fun(x, Ts):
                x = np.matrix(x)
                #print x

                #print x.shape
                if x.shape[0]<x.shape[1]:
                    x = x.T

                #print x

                out = [x.T*T*x for T in Ts]
                return np.matrix(np.squeeze(np.array(out))).T

            def min_fun(x, des):
                return la.norm(x/des-des/des)**2
                #return la.norm(x-des)**2
                


            if mode == 'sequential':

                x0 = np.matrix(np.random.rand(1)).T
                x0 = x0/la.norm(x0)

                for i in range(1,q+1):
                    
                    des = np.array(corr)[1:i+1,np.newaxis]
                    
                    if i == 1:
                        xopt = np.array([np.sqrt(corr[1])])
                    else:
                        xopt = opt.fmin(lambda x: min_fun(fun(x,[T[:i,:i] for T in Ts[:i]]),des), np.array(x0),
                                    disp= False)
                    
                    x0 = np.concatenate((xopt,[0]))

            elif mode == 'seqfull':
                Ts=[]
                for i in range(len(corr)-1):# range(int(q)):
                    vec = np.concatenate((np.arange(i+1,0,-1),np.zeros(len(corr)-1-i-1)))
                    T = np.matrix(la.toeplitz(vec))
                    Ts.append(T[:q,:q])
                    #print T

                def fun(x, Ts):
                    x = np.matrix(x)
                    #print x

                    #print x.shape
                    if x.shape[0]<x.shape[1]:
                        x = x.T

                    #print x

                    out = [x.T*T*x for T in Ts]
                    return np.matrix(np.squeeze(np.array(out))).T

                def min_fun(x, des):
                    return la.norm(x/des-des/des)**2
                    #return la.norm(x-des)**2

                    
                x0 = np.matrix(np.random.rand(1)).T
                x0 = x0/la.norm(x0)

                for i in range(1,q+1):
                    des = np.array(corr)[1:,np.newaxis]

                    if i == 1:
                        xopt = np.array([np.sqrt(corr[1])])
                    else:
                        xopt = opt.fmin(lambda x: min_fun(fun(x,[T[:i,:i] for T in Ts]),des), np.array(x0),
                                    disp= False)

                    x0 = np.concatenate((xopt,[0]))

            elif mode == 'random':
                x0 = np.matrix(np.random.rand(int(q))).T
                x0 = x0/la.norm(x0)
                des = np.array(corr)[1:int(q)+1,np.newaxis]
                xopt = opt.fmin(lambda x: min_fun(fun(x,Ts),des), np.array(x0),
                                disp= False)

            elif mode == 'stoch':
                x0 = np.matrix(np.random.rand(int(q))).T
                x0 = x0/la.norm(x0)
                des = np.array(corr)[1:int(q)+1,np.newaxis]
                optfun = lambda x: min_fun(fun(x,Ts),des)
                res = opt.basinhopping(optfun,np.array(x0),1000)
                xopt = res['x']

            elif mode == 'stoch2':
                des = np.array(corr)[1:int(q)+1,np.newaxis]
                optfun = lambda x: min_fun(fun(x,Ts),des)
                res = opt.differential_evolution(optfun, bounds=[(-1,1)]*q,
                                                 maxiter=1000, popsize=100,
                                                 polish=True)
                xopt = res['x']
            else:
                raise NotImplementedError

            model = ARMA([],xopt)

            self.models.append(model)

        return min_fun(fun(xopt,Ts),des)

    def closed_form_variance(self,ts):

        out = np.zeros((len(self.models),len(ts)))

        for j, model in enumerate(self.models):
            bb = np.matrix(model.b)
            if bb.shape[1] > bb.shape[0]:
                bb = bb.T



            q = bb.shape[0]
            for i, t in enumerate(ts):
                if q > t:
                    vec = np.concatenate((np.arange(int(t), 0, -1), np.zeros(q - int(t))))
                else:
                    vec = np.arange(int(t), 0, -1)[:q]

                R = np.matrix(la.toeplitz(vec))
                out[j,i] = (bb.T * R * bb)[0, 0]

        return out

    def avg(self, ts,choi_basis):

        singlet = False
        try:
            ts.__getitem__(0)
        except AttributeError:
            singlet = True
            ts = [ts]

        def appx_avg_sin_2(s):
            return s - s ** 2 + 2 * s ** 3 / 3 - s ** 4 / 3

        pow = appx_avg_sin_2(self.closed_form_variance(ts))

        chans = []
        #choi_basis = [QuantumChannel(la.expm(b)[:,:self.N],'stiefel').choi() for b in self.basis]
        Ichoi = QuantumChannel(np.eye(self.N),'unitary').choi()

        for i in range(len(ts)):
            mat_form = (1.-np.sum(pow[:,i]))*Ichoi+np.sum([p*C for p,C in zip(pow[:,i],choi_basis)],0)
            chans.append(QuantumChannel(mat_form,'choi'))

        if singlet:
            return chans[0]
        else:
            return chans

    # def avg(self, step, gate = None, tol = 1e-12):
    #
    #     U = np.zeros(self.basis[0].shape)
    #
    #     L = np.zeros(np.array(self.basis[0].shape)**2,dtype=np.complex)
    #
    #     I = np.eye(self.basis[0].shape[0])
    #
    #     for i, model in enumerate(self.models):
    #         vec = np.arange(step,0,-1)
    #         if step < len(model.b):
    #             vec = np.concatenate((vec, np.zeros(len(model.b)-step)))
    #
    #         T = np.matrix(la.toeplitz(vec))
    #
    #         x = np.matrix(model.b)
    #
    #         if x.shape[1]!=1:
    #             x = x.T
    #
    #         avg = x.T*T[:len(model.b),:len(model.b)]*x
    #         avg = avg[0,0]
    #
    #         b = np.matrix(self.basis[i])
    #
    #         #U1 = np.sum([1j**(2*k)/(2**k*fac(k))*avg**k*b**(2*k) for k in range(1,8)],0)
    #
    #         #L+= np.kron(I,I)+np.kron(I,U1)+np.kron(-U1,I)
    #
    #         conv = False
    #
    #         tot = 0
    #         norms = []
    #         while not conv:
    #
    #             Linc =  np.zeros(np.array(self.basis[0].shape)**2,dtype=np.complex)
    #
    #             for k in range(0,tot+1):
    #                 for j in range(0, tot-k+1):
    #
    #                     if j+k == tot:
    #                         coeff = (-1)**(k)*1j**(j+k)/(fac(k)*fac(j))
    #                         coeff *= fac(2*(j+k))/(2**(j+k)*fac(j+k))
    #
    #                         Linc += coeff*avg**((j+k)/2.)*np.kron(b**k,b**j)
    #             norms.append(la.norm(Linc))
    #             if la.norm((Linc+L)-L) ==0.0 and tot >12:
    #                 conv = True
    #
    #
    #             L+=Linc
    #             tot+=2
    #
    #         #alg+=np.sqrt(avg[0,0])*self.basis[i]
    #
    #
    #     #U += np.eye(U.shape[0])
    #
    #     #L = np.kron(np.conj(U),U)
    #     #assert False
    #     if gate is None:
    #         return QuantumChannel(L,'liou')
    #     else:
    #         return QuantumChannel(L,'liou')*D[gate]

class SchWAR(SchWARMA):

    def __init__(self, ARs, basis, D, rgen = np.random):

        for AR in ARs:
            assert len(AR.b) == 1 

        super(SchWAR,self).__init__(ARs, basis, D, rgen = rgen)

    def fit(self, corrs, ps, mode='full'):

        ps = np.ones(len(self.basis))*np.array(ps)

        assert len(ps) == len(corrs)

        self.models = []

        def qf(a,b,c):
            x = (-b+np.sqrt(b**2-4*a*c))/(2*a), (-b-np.sqrt(b**2-4*a*c)/2*a)
            if np.abs(x[0])<np.abs(x[1]):
                return x[0],x[1]
            else:
                return x[1],x[0]

        for corr, p in zip(corrs, ps):
            p = int(p)

            #des = np.array(corr)[0:p+1,np.newaxis]

            #YW = la.toeplitz(des[:-1])
            #a = np.dot(la.pinv(YW),des[1:])
            #b = np.sqrt(des[0])
            

            #print des, b
            #A = np.zeros((p+1,p+1))
            #A[:,0] = 1.
            #for i in range(1,p+1):
            #    A[i:,i] = corr[1:p+2-i]

            ##if mode == 'nnls':
            #    a,res = opt.nnls(A,corr[1:p+2][:np.newaxis])
            #    a = np.sqrt(a)
            #elif mode == 'ls':
            #    out = opt.lsq_linear(A,corr[1:p+2][:np.newaxis])
            #    a = out['x']
            #    res = out['cost']
            #    a = np.sqrt(a+0.j)
            #
            #a[1] -= 1
            #AR = ARMA(a[1:],[a[0]])
            
            #Delta = corr[2:]-corr[1:-1]
            #if mode == 'full':
            #    pp = len(Delta)
            #elif mode == 'first':
            #    pp = p
            #else:
            #    raise NotImplementedError

            #A = np.eye(pp)
            #for i in range(0,pp-1):
            #    A[i+1:,i] = Delta[:pp-i-1]

            #a,res = opt.nnls(A[:,:p],Delta[:pp])

            #a = np.sqrt(a)
            #b = np.sqrt(corr[1])
            corr = np.array(corr)
            Delta = corr[2:]-corr[1:-1]
            a = np.zeros(p)
            b = np.sqrt(corr[1])
            for i in range(p):
                if i == 0:
                    a[i] = np.sqrt(Delta[i]/corr[1])
                else:                    
                    part = np.sqrt(Delta[:i][::-1]/corr[1])
                    q = np.matrix(np.concatenate((part,[1])))
                    Q = q.T*q
                    amat = np.matrix(a[:i])
                    
                    qfb = 2*amat*Q[:-1,-1]
                    qfc = amat*Q[:-1,:-1]*amat.T-Delta[i]/corr[1]
                    
                    out = qf(1,qfb[0,0],qfc[0,0])
                    a[i]=out[0]
                
            res = 0.0
            #a[0]-=1.
            AR = ARMA(a,[b])

            self.models.append(AR)

        return res








#class SchWARMA(object):
#    
#    def __init__(self, b, sd, D, rgen = np.random, normalize = True):
#        self.b = np.array(b)
#        if normalize:
#            self.b = self.b/np.sqrt(np.sum(self.b**2))
#        self.sd = sd
#        self.rgen = rgen
#        self.x = sd*self.rgen.randn(len(b))
#        self.y = sd*self.rgen.randn(len(b))
#        self.z = sd*self.rgen.randn(len(b))
#        self.D = D
#    def next_err(self):
#        self.x = np.concatenate((self.sd*self.rgen.randn(1),self.x[:-1]))
#        self.y = np.concatenate((self.sd*self.rgen.randn(1),self.y[:-1]))
#        self.z = np.concatenate((self.sd*self.rgen.randn(1),self.z[:-1]))
#        
#        alg = np.sum(self.b*self.x)*qr.sigmaX + np.sum(self.b*self.y)*qr.sigmaY + np.sum(self.b*self.z)*qr.sigmaZ
#        exp = la.expm(1j*alg)
#        return np.kron(np.conj(exp),exp)
#    
#    def __getitem__(self,index):
#        if index == 'del_t':
#            return self.D['del_t']
#        else:
#            err = self.next_err()
#            gate = self.D[index]
#            return np.dot(err, gate)
#
#        
#class dephasingSchWARMA(SchWARMA):
#    
#    def next_err(self):
#        #self.x = np.concatenate((self.sd*self.rgen.randn(1),self.x[:-1]))
#        #self.y = np.concatenate((self.sd*self.rgen.randn(1),self.y[:-1]))
#        self.z = np.concatenate((self.sd*self.rgen.randn(1),self.z[:-1]))
#        
#        #alg = np.sum(self.b*self.x)*qr.sigmaX + np.sum(self.b*self.y)*qr.sigmaY + np.sum(self.b*self.z)*qr.sigmaZ
#        alg = np.sum(self.b*self.z)*qr.sigmaZ
#        exp = la.expm(1j*alg)
#        return np.kron(np.conj(exp),exp)
