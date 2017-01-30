
# coding: utf-8

# In[1]:

import numpy as np


# In[54]:

class UmaaRhomaa():
    def __init__(self):
        self.qtz_rhoma = 2.6500
        self.cal_rhoma = 2.7100
        self.dol_rhoma = 2.8700
        self.qtz_uma = 4.8000
        self.cal_uma = 13.8000
        self.dol_uma = 9.0000
        self.unity = 1.0000
        self.invert()
        
    def invert(self):
        self.A = np.matrix([[self.qtz_rhoma, self.cal_rhoma, self.dol_rhoma],
                       [self.qtz_uma, self.cal_uma, self.dol_uma],
                       [self.unity, self.unity, self.unity]])
        self.I = self.A.I
        
    def compute_lith(self, umaa, rhomaa):
        self.invert()
        self._qtz = self.I[0,0] * rhomaa + self.I[0,1] * umaa + self.I[0,2]
        self._cal = self.I[1,0] * rhomaa + self.I[1,1] * umaa + self.I[1,2]
        self._dol = self.I[2,0] * rhomaa + self.I[2,1] * umaa + self.I[2,2]
        if self._qtz < 0.0:
            self._qtz = 0.0
        if self._qtz > 1.0:
            self._qtz = 1.0
        if self._cal < 0.0:
            self._cal = 0.0
        if self._cal > 1.0:
            self._cal = 1.0
        if self._dol < 0.0:
            self._dol = 0.0
        if self._dol > 1.0:
            self._dol = 1.0        
        self.qtz = self._qtz / (self._qtz + self._cal + self._dol)
        self.cal = self._cal / (self._qtz + self._cal + self._dol)
        self.dol = self._dol / (self._qtz + self._cal + self._dol)
        
    def set_qtz_rhoma(self, x):
        self.qtz_rhoma = x
        
    def set_cal_rhoma(self, x):
        self.cal_rhoma = x

    def set_dol_rhoma(self, x):
        self.dol_rhoma = x
        
    def set_qtz_uma(self, x):
        self.qtz_uma = x
        
    def set_cal_uma(self, x):
        self.cal_uma = x
        
    def set_dol_uma(self, x):
        self.dol_uma = x
        
    def get_qtz(self, umaa, rhomaa):
        self.compute_lith(umaa, rhomaa)
        return self.qtz
    
    def get_cal(self, umaa, rhomaa):
        self.compute_lith(umaa, rhomaa)
        return self.cal
    
    def get_dol(self, umaa, rhomaa):
        self.compute_lith(umaa, rhomaa)
        return self.dol
    
    def print_equations(self):
        self.invert()
        print('qtz = ' + str(self.I[0,0]) + ' * rhomaa + ' + str(self.I[0,1]) + ' * umaa + ' + str(self.I[0,2]))
        print('cal = ' + str(self.I[1,0]) + ' * rhomaa + ' + str(self.I[1,1]) + ' * umaa + ' + str(self.I[1,2]))
        print('dol = ' + str(self.I[2,0]) + ' * rhomaa + ' + str(self.I[2,1]) + ' * umaa + ' + str(self.I[2,2]))