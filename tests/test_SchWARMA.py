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

import unittest
import mezze.random.SchWARMA as SchWARMA
import mezze.channel as ch
import numpy as np
import scipy.linalg as la
import copy


s_i = np.matrix([[1,0],[0,1]])
s_x = np.matrix([[0,1],[1,0]])
s_y = np.matrix([[0,-1j],[1j,0]])
s_z = np.matrix([[1,0],[0,-1]])



class TestSchWARMA(unittest.TestCase):

    def test_init(self):

        ARMAs = [SchWARMA.ARMA([.5,.25],[1])]
        ARMAs.append(SchWARMA.ARMA([],[1]))
        ARMAs.append(SchWARMA.ARMA([],[1,1,1]))

        D = {'I': ch.QuantumChannel(np.matrix(np.eye(4)))}

        sch = SchWARMA.SchWARMA(ARMAs, [s_x,s_y,s_z], D)

        xs = [copy.copy(model.x) for model in sch.models]
        ys = [copy.copy(model.y) for model in sch.models]

        sch.init_ARMAs()

        xs2 = [copy.copy(model.x) for model in sch.models]
        ys2 = [copy.copy(model.y) for model in sch.models]

        xdiff = np.array([la.norm(x-xx) for x,xx in zip(xs,xs2)])
        ydiff = np.array([la.norm(yy) for y,yy in zip(ys,ys2)])

        self.assertTrue(np.all(xdiff>0))
        self.assertTrue(np.all(ydiff==0))


    def test_no_SchWARMA(self):

        ARMAs = [SchWARMA.ARMA([],[0])]
        ARMAs.append(SchWARMA.ARMA([],[0]))
        ARMAs.append(SchWARMA.ARMA([],[0]))

        D = {'I': ch.QuantumChannel(np.matrix(np.eye(4))),
             'X': ch.QuantumChannel(np.kron(s_x.conj(),s_x))}

        sch = SchWARMA.SchWARMA(ARMAs, [s_x,s_y,s_z], D)

        sch.init_ARMAs

        self.assertEqual(la.norm(D['I'].choi()-sch['I'].choi()),0)
        self.assertEqual(la.norm(D['X'].choi()-sch['X'].choi()),0)

    def test_hamiltonian_basis(self):

        ARMAs = [SchWARMA.ARMA([.5,.25],[1])]
        ARMAs.append(SchWARMA.ARMA([],[1]))
        ARMAs.append(SchWARMA.ARMA([],[1,1,1]))

        D = {'I': ch.QuantumChannel(np.matrix(np.eye(4)))}

        sch = SchWARMA.SchWARMA(ARMAs, [s_x,s_y,s_z], D)

        self.assertTrue(all([la.norm(b+b.H<1e-9)for b in sch.basis]))

    def test_zero_basis(self):

        ARMAs = [SchWARMA.ARMA([.5,.25],[1])]
        ARMAs.append(SchWARMA.ARMA([],[1]))

        D = {'I': ch.QuantumChannel(np.matrix(np.eye(4)))}

        sch = SchWARMA.SchWARMA(ARMAs, [np.zeros((8,2)),np.zeros((2,2))],D)

        self.assertEqual(sch.N,2)
        self.assertTrue(all([b.shape==(8,8) for b in sch.basis]))

        self.assertEqual(la.norm(sch['I'].liouvillian()-np.eye(4)),0)

    def test_mixed_basis(self):
        ARMAs = [SchWARMA.ARMA([.5,.25],[1])]
        ARMAs.append(SchWARMA.ARMA([],[1]))
        ARMAs.append(SchWARMA.ARMA([],[1,1,1]))

        D = {'I': ch.QuantumChannel(np.matrix(np.eye(4))),
             'X': ch.QuantumChannel(np.kron(s_x.conj(),s_x))}

        ad = np.zeros((4,2),dtype=np.complex)
        ad[2,1]=.1
        sch = SchWARMA.SchWARMA(ARMAs, [s_x,1j*s_y,ad], D)

        self.assertEqual(sch.N, 2)
        self.assertTrue(all([la.norm(b + b.H < 1e-9) for b in sch.basis]))

    def test_zero_basis(self):

        ARMAs = [SchWARMA.ARMA([.5,.25],[1])]
        ARMAs.append(SchWARMA.ARMA([],[1]))

        D = {'I': ch.QuantumChannel(np.matrix(np.eye(4)))}

        sch = SchWARMA.SchWARMA(ARMAs, [s_z, np.zeros((8,2))],D)

        self.assertEqual(sch.N,2)
        self.assertTrue(all([b.shape==(8,8) for b in sch.basis]))
        self.assertEqual(la.norm(1j*s_z-sch.basis[0][:2,:2]),0)
        self.assertEqual(la.norm(sch.basis[0][2:,:]),0)
        self.assertEqual(la.norm(sch.basis[0][:,2:]),0)

        out = sch['I']
        self.assertEqual(out.rank(),1)
        self.assertEqual(out.chi()[1,1],0)
        self.assertEqual(out.chi()[2,2],0)

        avg = ch.QuantumChannel(np.mean([sch['I'].choi() for _ in range(3)],0),'choi')
        self.assertEqual(avg.rank(),2)
        self.assertEqual(avg.chi()[1,1],0)
        self.assertEqual(avg.chi()[2,2],0)

if __name__ == '__main__':
    unittest.main()
