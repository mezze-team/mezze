# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:32:35 2015

@author: dclader
"""

from circuit import *

class Gates(UARCircuitSimulator):
    """
    Class to build a circuit and simulate it using the UAR simulator code
    """
    def __init__(self, pmd, config):
        
        self._gate_defs = {'I': {'Control': ((0,),), 'Time': (1.0,), 'Theta': ((0.0,),)},
                           'I2': {'Control': ((0,),), 'Time': (2.0,), 'Theta': ((0.0,),)},
                           'I3': {'Control': ((0,),), 'Time': (3.0,), 'Theta': ((0.0,),)},
                           'I4': {'Control': ((0,),), 'Time': (4.0,), 'Theta': ((0.0,),)},
                           'XI': {'Control': ((0,),), 'Time': (1.0,), 'Theta': ((np.pi,),)},
                      'X': {'Control': ((0,),), 'Time': (1.0,), 'Theta': ((np.pi/2.0,),)},
                      'Y': {'Control': ((1,),), 'Time': (1.0,), 'Theta': ((np.pi/2.0,),)},
                      'ZZ90': {'Control': ((2,),), 'Time': (4.0,), 'Theta': ((-np.pi/4.0,),)},
                      'S': {'Control': ((2,),), 'Time': (1.0,), 'Theta': ((np.pi/4.0,),)},
                      'H': {'Control': ((1,),(0,)), 'Time': (0.5,0.5), 'Theta': ((np.pi/4.0,),(np.pi/2.0,))},
                      'Z90H': {'Control': ((1,), (0,)), 'Time': (0.5, 0.5), 'Theta': ((np.pi / 4.0,), (-np.pi / 4.0,))},
                      'H2': {'Control': ((0,),(1,)), 'Time': (0.5,0.5), 'Theta': ((np.pi/2.0,),(-np.pi/4.0,))}}

        #self._gate_defs = {'I': {'Control': ((0,),), 'Time': (1.0,), 'Theta': ((0.0,),)},
        #                   'P': {'Control': ((2,),), 'Time': (1.0,), 'Theta': ((np.pi/4.0,),)},
        #                   'H': {'Control': ((1,),(0,)), 'Time': (0.5,0.5), 'Theta': ((np.pi/4.0,),(np.pi/2.0,))}}

        UARCircuitSimulator.__init__(self, pmd, config, self._gate_defs)


    def get_gate_names(self):
        return self._gate_defs.keys()
    
      
            

        