# -*- coding: utf-8 -*-
"""
    lambert module
    ==============
    
    Contains the CR3BP class, that allows to create a cr3bp problem for a given value
    of the µ parameter. CR3BP object contains methods to compute the flow, and some
    dynamical properties such as the STM.
    
    Contains the Lambert_Problem class, that allows to create a lambert problem in a 
    given CR3BP problem. Solving is made using a shooting method. 
    @author: Simon Dupourqué 
"""

import numpy as np
import pykep as pk
from cr3bp import CR3BP

class Lambert_Problem:

    def __init__(self,cr3bp,R1,R2,t):
        
        self._cr3bp = cr3bp
        self._R1 = R1
        self._R2 = R2
        self._t = t
        self.sol = np.zeros((3,))
        self._convergence = float("inf")
        self._nb_of_iterations = 0
        self._initialized = False
        self._converged = False

    def _iteration_lambert(self):
        
        if not self._initialized:
            print("Warning : No first guess was set\nConsider using .set_initial_2BP()")
            self._initialized = True
        
        phi,M = self._cr3bp.propagate_STM(np.hstack((self._R1,self.sol)),self._t)
        DG = np.matrix(M[0:3,3:6])
     
        G = np.reshape(phi[0:3]-self._R2,(3,1))
        new_sol =  np.reshape(np.array(np.reshape(self.sol,(3,1))-DG**(-1)*G),(3,))
                
        self._convergence = np.linalg.norm(self.sol-new_sol)
        self._nb_of_iterations += 1
        self.sol = new_sol
        
    def solve(self):
        
        while self._convergence > 1e-8 and self._nb_of_iterations < 100:
            self._iteration_lambert()
            
        self._converged = self._convergence < 1e-8
        
        return {"solution":self.sol,"success":self._converged,"iterations":self._nb_of_iterations}
                
    def set_initial_2BP(self):
    
        def rotation(t) : 
            rotation = np.zeros((3,3))
            rotation[0,:] = np.array([np.cos(t),-np.sin(t),0])
            rotation[1,:] = np.array([np.sin(t),np.cos(t),0])
            rotation[2,:] = np.array([0,0,1])
            return(rotation)
        
        R1_abs = np.matmul(rotation(0),self._R1)
        R2_abs = np.matmul(rotation(self._t),self._R2)
        
        l = pk.lambert_problem(R1_abs,R2_abs,self._t)
        V1_abs = l.get_v1()[0]
        V1_rot = V1_abs - np.cross(np.array([0,0,1]),R1_abs)
        
        self.sol = V1_rot
        self._initialized = True
        
    def set_initial_guess(self,V):
        
        self.sol = V
        self._initialized = True

#%% Example script
        
if __name__ == '__main__':
    
    mu = 1/(81.3005617547232*(1+1/81.3005617547232))
    R1 = np.array([1.119950858486894e+00,-9.763556537105871e-02,0]) 
    R2 = np.array([1.167819506401286e+00,5.497640206149425e-02,-5.741831377097451e-02])
    cr3bp = CR3BP(mu)
    problem = Lambert_Problem(cr3bp,R1,R2,2)
    problem.set_initial_2BP()
    result = problem.solve()
