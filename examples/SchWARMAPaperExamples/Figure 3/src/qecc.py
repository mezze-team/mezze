import numpy as np
from scipy import stats
import qutip as qt

class SurfaceSyndrome():
    def __init__(self, num_qubits=5):
        self.num_qubits = num_qubits
        self.build_ctrl_list(num_qubits)

    def build_ctrl_list(self, num_qubits):
        self.ctrl_list = [[] for i in range(int(num_qubits + 1. / 2 * num_qubits * (num_qubits - 1)))]

    def generate_xsyndrome_controls(self):
        self.add_h2(exceptions=list(range(1, 15)))
        self.cnot_gate(0, 2)
        self.cnot_gate(0, 1)
        self.cnot_gate(0, 4)
        self.cnot_gate(0, 3)
        self.add_h(exceptions=list(range(1, 15)))
        # Hack to push the first and last Hadamards into the first and last CNOT gates
        del (self.ctrl_list[0][1])
        del (self.ctrl_list[0][-2])
        return self.ctrl_list

    def generate_zsyndrome_controls(self):
        self.add_h2(exceptions=list(range(1, 15)))
        self.add_identity(exceptions=[0])
        self.zz_gate(2, 0)
        self.zz_gate(1, 0)
        self.zz_gate(4, 0)
        self.zz_gate(3, 0)
        self.add_h(exceptions=list(range(1, 15)))
        self.add_identity(exceptions=[0])
        return self.ctrl_list

    def cnot_gate(self, ctrl, target):
        if ctrl > target:
            temp = ctrl
            tqctrl = target
            tqtarget = temp
        else:
            tqctrl = ctrl
            tqtarget = target

        sq_ctrl = self.ctrl_list[ctrl]
        sq_target = self.ctrl_list[target]
        tq_idx = self.num_qubits + self.num_qubits * tqctrl + tqtarget - (tqctrl * (tqctrl + 1) * 1 // 2 + tqctrl + 1)
        tq_ctrl = self.ctrl_list[tq_idx]
        # U_CNOT = H2.Exp(i*Pi/4).Z1(pi/4).Z2(pi/4).ZZ(-pi/4).H2
        ctrl_gates = ['I', 'I4', 'I']
        targ_gates = ['H2', 'I4', 'Z90H']
        tq_gates = ['I', 'ZZ90', 'I']
        for i in range(len(ctrl_gates)):
            sq_ctrl.append(ctrl_gates[i])
            sq_target.append(targ_gates[i])
            tq_ctrl.append(tq_gates[i])
        self.ctrl_list[ctrl] = sq_ctrl
        self.ctrl_list[target] = sq_target
        self.ctrl_list[tq_idx] = tq_ctrl

        self.add_identity(exceptions=[ctrl, target, tq_idx])
        self.add_identity(exceptions=[ctrl, target, tq_idx], key="I4")
        #self.add_xy4(exceptions=[ctrl, target, tq_idx])
        #self.add_identity(exceptions=[ctrl, target, tq_idx])
        self.add_identity(exceptions=[ctrl, target, tq_idx])

    def zz_gate(self, ctrl, target):
        if ctrl > target:
            temp = ctrl
            tqctrl = target
            tqtarget = temp
        else:
            tqctrl = ctrl
            tqtarget = target

        sq_ctrl = self.ctrl_list[ctrl]
        sq_target = self.ctrl_list[target]
        tq_idx = self.num_qubits + self.num_qubits * tqctrl + tqtarget - (tqctrl * (tqctrl + 1) * 1 // 2 + tqctrl + 1)
        tq_ctrl = self.ctrl_list[tq_idx]
        # U_CNOT = H2.Exp(i*Pi/4).Z1(pi/4).Z2(pi/4).ZZ(-pi/4).H2
        ctrl_gates = ['I4']
        targ_gates = ['I4']
        tq_gates = ['ZZ90']
        for i in range(len(ctrl_gates)):
            sq_ctrl.append(ctrl_gates[i])
            sq_target.append(targ_gates[i])
            tq_ctrl.append(tq_gates[i])
        self.ctrl_list[ctrl] = sq_ctrl
        self.ctrl_list[target] = sq_target
        self.ctrl_list[tq_idx] = tq_ctrl

        self.add_identity(exceptions=[ctrl, target, tq_idx], key="I4")

    def add_identity(self, exceptions=[], key="I"):
        for i in range(len(self.ctrl_list)):
            ctrl = self.ctrl_list[i]
            if i not in exceptions:
                ctrl.append(key)
                self.ctrl_list[i] = ctrl

    def add_xy4(self, exceptions=[]):
        for i in range(self.num_qubits):
            ctrl = self.ctrl_list[i]
            if i not in exceptions:
                ctrl.append("X")
                ctrl.append("Y")
                ctrl.append("X")
                ctrl.append("Y")
                self.ctrl_list[i] = ctrl

    def add_xi(self, exceptions=[]):
        for i in range(len(self.ctrl_list)):
            ctrl = self.ctrl_list[i]
            if i not in exceptions:
                ctrl.append('XI')
                self.ctrl_list[i] = ctrl

    def add_x(self, exceptions=[]):
        for i in range(len(self.ctrl_list)):
            ctrl = self.ctrl_list[i]
            if i not in exceptions:
                ctrl.append('X')
                self.ctrl_list[i] = ctrl

    def add_h(self, exceptions=[]):
        for i in range(len(self.ctrl_list)):
            ctrl = self.ctrl_list[i]
            if i not in exceptions:
                ctrl.append('H')
                self.ctrl_list[i] = ctrl

    def add_h2(self, exceptions=[]):
        for i in range(len(self.ctrl_list)):
            ctrl = self.ctrl_list[i]
            if i not in exceptions:
                ctrl.append('H2')
                self.ctrl_list[i] = ctrl

    def reset_ctrl_list(self):
        self.build_ctrl_list(5)

    def add_random_bitflip(self, debug=False):
        rand_q = np.random.randint(5)
        sq_list = self.ctrl_list[rand_q]
        rand_gate = np.random.randint(len(sq_list))
        while ((sq_list[rand_gate] != 'I') or self.check_for_two_qubit_gate(rand_q, rand_gate)):
            rand_gate = np.random.randint(len(sq_list))
        sq_list[rand_gate] = 'X'
        if debug == True:
            print(sq_list)
        self.ctrl_list[rand_q] = sq_list
        self.error_location = [rand_q, rand_gate]

    def check_for_two_qubit_gate(self, q_idx, gate_idx):
        # Checks for two qubit gate in ctrl_list
        # True: Gate present
        gates = []
        for i in range(self.num_qubits):
            if q_idx != i:
                ctrl, target = np.sort([q_idx, i])
                tq_idx = self.num_qubits + self.num_qubits * ctrl + target - (ctrl * (ctrl + 1) * 1 // 2 + ctrl + 1)
                tq_list = self.ctrl_list[tq_idx]
                gates.append(tq_list[gate_idx])
        gate_present = False
        for gate in gates:
            if gate != 'I':
                gate_present = True
                break
        return gate_present
