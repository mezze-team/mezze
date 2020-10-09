import numpy as np
import scipy.linalg as la
import qutip as qt
from mezze.channel import _sigmaI, _sigmaX, _sigmaY, _sigmaZ

def zz_gate(arg_value):
    # ZZ interaction
    mat = np.zeros((4, 4), dtype=np.complex)
    mat[0, 0] = mat[3, 3] = np.exp(-0.5j*arg_value)
    mat[1, 1] = mat[2, 2] = np.exp(0.5j*arg_value)
    return qt.Qobj(mat, dims=[[2, 2], [2, 2]])

def xyz_gate(arg_value):
    mat = la.expm(-0.5j * (arg_value[0] * _sigmaX + arg_value[1] * _sigmaY +arg_value[2] * _sigmaZ))
    return qt.Qobj(mat, dims=[[2], [2]])

def add_qtcnot(qcirc, ctrl, targ, xerr=None, yerr=None, zerr=None):
    qcirc.add_gate("SNOT", targets=[targ])
    for i in range(qcirc.N):
        xerrval = xerr[i][0] if xerr is not None else 0.0
        yerrval = yerr[i][0] if yerr is not None else 0.0
        zerrval = zerr[i][0] if zerr is not None else 0.0
        qcirc.add_gate("XYZ", targets=[i], arg_value=[xerrval, yerrval, zerrval])

    qcirc.add_gate("CZZ", arg_value=-np.pi / 2.0, targets=[ctrl, targ], arg_label=r'\frac{\pi}{2}')
    for i in range(qcirc.N):
        for j in range(4):
            xerrval = xerr[i][1+j] if xerr is not None else 0.0
            yerrval = yerr[i][1+j] if yerr is not None else 0.0
            zerrval = zerr[i][1+j] if zerr is not None else 0.0
            qcirc.add_gate("XYZ", targets=[i], arg_value=[xerrval, yerrval, zerrval])

    qcirc.add_gate("RY", targets=[targ], arg_value=np.pi / 2.0, arg_label=r'\frac{\pi}{2}')
    qcirc.add_gate("RX", targets=[targ], arg_value=-np.pi / 2.0, arg_label=r'\frac{\pi}{2}')
    for i in range(qcirc.N):
        xerrval = xerr[i][5] if xerr is not None else 0.0
        yerrval = yerr[i][5] if yerr is not None else 0.0
        zerrval = zerr[i][5] if zerr is not None else 0.0
        qcirc.add_gate("XYZ", targets=[i], arg_value=[xerrval, yerrval, zerrval])

def add_qtzz(qcirc, ctrl, targ, xerr=None, yerr=None, zerr=None):
    qcirc.add_gate("CZZ", arg_value=-np.pi / 2.0, targets=[ctrl, targ], arg_label=r'\frac{\pi}{2}')
    for i in range(qcirc.N):
        for j in range(4):
            xerrval = xerr[i][j] if xerr is not None else 0.0
            yerrval = yerr[i][j] if yerr is not None else 0.0
            zerrval = zerr[i][j] if zerr is not None else 0.0
            qcirc.add_gate("XYZ", targets=[i], arg_value=[xerrval, yerrval, zerrval])

def add_qth(qcirc, targ, xerr=None, yerr=None, zerr=None):
    qcirc.add_gate("SNOT", targets=[targ])
    for i in range(qcirc.N):
        xerrval = xerr[i] if xerr is not None else 0.0
        yerrval = yerr[i] if yerr is not None else 0.0
        zerrval = zerr[i] if zerr is not None else 0.0
        qcirc.add_gate("XYZ", targets=[i], arg_value=[xerrval, yerrval, zerrval])