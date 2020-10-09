# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:32:35 2015

@author: dclader
"""

import numpy as np
import mezze

class UARCircuitSimulator(object):
    """
    Class to build a circuit and simulate it using the UAR simulator code
    """
    def __init__(self, pmd, config, gate_defs=None):
        self._pmd=pmd
        self._config=config
        self._gate_defs=gate_defs
        
        # Example of how to define gate
        #if self._gate_defs is None :
        #self._gate_defs = {'I': {'Control': ((0,),), 'Time': (1.0,), 'Theta': ((0.0,),)},
                            #'X': {'Control': ((0,),), 'Time': (1.0,), 'Theta': ((np.pi/2.0,),)},
                            #'Y': {'Control': ((1,),), 'Time': (1.0,), 'Theta': ((np.pi/2.0,),)},
                            #'Z': {'Control': ((2,),), 'Time': (1.0,), 'Theta': ((np.pi/2.0,),)},
                            #'Ypiover4': {'Control': ((1,),), 'Time': (1.0,), 'Theta': ((np.pi/4.0,),)},
        #'H': {'Control': ((1,),(0,)), 'Time': (0.5,0.5), 'Theta': ((np.pi/4.0,),(np.pi/2.0,))}}
            
    
    def define_gates(self, gate_defs):
        if isinstance(gate_defs,dict):
            self._gate_defs.update(gate_defs)
        else:
            raise ValueError("The argument must be a dictionary.")
            
    def get_defined_gates(self):
        return self._gate_defs
            

    def generate_circuit_controls(self, circuit):
        
        # First check that all gates are defined. If not raise an exception.
        for gate in circuit :
            if gate not in self._gate_defs :
                raise ValueError("Circuit has undefined gates.")

        # Check that there are enough time steps to integrate
        num_gates = len(circuit)
        num_steps = self._config.num_steps # for typing ease
        if num_gates > num_steps :
            raise ValueError("The number of integration steps must be greater than the number of gates.")
            
        # Get total circuit time
        total_circuit_time = 0
        for gate in circuit :
            for ctrl_time in self._gate_defs[gate]['Time'] :
                total_circuit_time += ctrl_time
        
        #zero_field = mezze.controls.get_zero_control(zero_field = mezze.controls.get_zero_control(gate_length))
        controls = [[] for _ in range(self._pmd.control_dim)]
        # Run through the circuit and generate primitive controls
        for gate_idx, gate in enumerate(circuit) :
            # Loop over the number of sequential controls in the gate
            for i_ctrls in range(len(self._gate_defs[gate]['Control'])) :
                controls_not_used = [x for x in range(self._pmd.control_dim) if x not in self._gate_defs[gate]['Control'][i_ctrls]]
                gate_length=0                
                # Now loop over the individual controls being used at this time sequence
                for ctrl in range(len(self._gate_defs[gate]['Control'][i_ctrls])) :
                    # Set control length
                    control_length=int(np.round(self._config.num_steps*self._gate_defs[gate]['Time'][i_ctrls]/total_circuit_time))
                    control_time=self._config.time_length*self._gate_defs[gate]['Time'][i_ctrls]/total_circuit_time
                    
                    # If the gate times cannot be broken up correctly into equal size chunks, correct the time argument
                    # to make sure the integral works correctly
                    ratio = float(control_length)/(self._config.num_steps*self._gate_defs[gate]['Time'][i_ctrls]/total_circuit_time)
                    
                    # Check if we need to interpolate the last control to extend to the number of integration steps
                    control_length_new = control_length
                    if (gate_idx == len(circuit)-1) and (i_ctrls == len(self._gate_defs[gate]['Control'])-1) :
                        total_circuit_steps = len(controls[self._gate_defs[gate]['Control'][i_ctrls][ctrl]]) + control_length
                        if total_circuit_steps is not self._config.num_steps :
                           extension_size = self._config.num_steps - total_circuit_steps
                           control_length_new = control_length + extension_size
                           
                    # Get the field for the current control
                    field = mezze.controls.get_theta_pulse_control(control_length_new, control_time*ratio, self._gate_defs[gate]["Theta"][i_ctrls][ctrl])
                           
                    # Add this control onto the control list
                    controls[self._gate_defs[gate]['Control'][i_ctrls][ctrl]].extend(field.tolist())

                    
                # Now set the controls not used to zero field
                zero_field = mezze.controls.get_zero_control(control_length_new)
                for i in controls_not_used :
                    #controls_init[i]=np.concatenate((controls_init[i],zero_field))  
                    controls[i].extend(zero_field.tolist())                
        
        
        return controls      
            

        
