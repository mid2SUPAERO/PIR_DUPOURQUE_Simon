# -*- coding: utf-8 -*-
"""
    CR3BP module
    ============
    
    Contains the CR3BP class, that allow to create a cr3bp problem for a given value
    of the µ parameter. CR3BP object contains methods to compute the flow, and some
    dynamical properties such as the STM.

    @author: Simon Dupourqué 
"""
import numpy as np
from ode import ode87

class CR3BP:
    
    def __init__(self,mu):
        self.mu = mu
        
    def ode_function(self,t,Y):
        """ 
            Compute the dynamical equations of the CR3BP problem
            
            :param t: Dummy parameter, to interface with ode functions
            :param Y: State vector of the system
            :type t: float
            :type Y: numpy.ndarray
            
        """
    
        dF = np.zeros((6))    
        dF[0] = Y[3]
        dF[1] = Y[4]
        dF[2] = Y[5]
        dF[3] = - self._dOmega_dx(Y) + 2*Y[4]
        dF[4] = - self._dOmega_dy(Y) - 2*Y[3]
        dF[5] = - self._dOmega_dz(Y)
      
        return(dF)
        
    def propagate(self,Y,t,t0=0):
        """
            Compute the flow at time t with the state vector Y as initial value
            
            :param Y: State vector of the system
            :param t: Final time
            :param t0: Initial time (default : 0)
            :type Y: numpy.ndarray
            :type t: float
            :type t0: float
                    
        """
        return ode87(self.ode_function,0,t,Y,propagate=True)
        
    def trajectory(self,Y,t,t0=0,points=100):
        """
        Compute several points to display the trajectory 
        """
        return ode87(self.ode_function,0,t,Y,minimal_nb_of_points=points)
    
    def jacobi_integral(self,Y):
        """
        Compute the Jacobi Integral of the state vector Y
        """
        return(-(Y[3]**2+Y[4]**2+Y[5]**2) - 2*self._Omega(Y[0],Y[1],Y[2],self.mu))
        
    def jacobian_matrix(self,Y):
        """
        Compute the Jacobian Matrix of the system for the state vector Y
        """
    
        x = Y[0]
        y = Y[1]
        z = Y[2]
        
        A = np.matrix(np.zeros((6,6)))
        
        A[0,3] = 1
        A[1,4] = 1
        A[2,5] = 1
        A[3,4] = 2
        A[4,3] = -2
        A[3,0] = -self._d2Omega_dx2(x,y,z,self.mu)
        A[3,1] = -self._d2Omega_dxdy(x,y,z,self.mu)
        A[3,2] = -self._d2Omega_dxdz(x,y,z,self.mu)
        A[4,0] = -self._d2Omega_dxdy(x,y,z,self.mu)
        A[4,1] = -self._d2Omega_dy2(x,y,z,self.mu)
        A[4,2] = -self._d2Omega_dydz(x,y,z,self.mu)
        A[5,0] = -self._d2Omega_dxdz(x,y,z,self.mu)
        A[5,1] = -self._d2Omega_dydz(x,y,z,self.mu)
        A[5,2] = -self._d2Omega_dz2(x,y,z,self.mu)
        
        return(A)
        
    def _Omega(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                -(1-mu)*((x+mu)**2+y**2+z**2)**(-1/2)
                -(mu)*((x-(1-mu))**2+y**2+z**2)**(-1/2)
                - 0.5*(x**2+y**2)
                )
        
    def _dOmega_dx(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                +(1-mu)*(x+mu)*((x+mu)**2+y**2+z**2)**(-3/2)
                + mu*(x-(1-mu))*((x-(1-mu))**2+y**2+z**2)**(-3/2)
                -x
                )
        
    def _dOmega_dy(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                +(1-mu)*y*((x+mu)**2+y**2+z**2)**(-3/2)
                + mu*y*((x-(1-mu))**2+y**2+z**2)**(-3/2)
                -y
                )
    
    def _dOmega_dz(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                +(1-mu)*z*((x+mu)**2+y**2+z**2)**(-3/2)
                + mu*z*((x-(1-mu))**2+y**2+z**2)**(-3/2)
                )
    
    def _d2Omega_dx2(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                -3*(1-mu)*(x+mu)**2*((x+mu)**2+y**2+z**2)**(-5/2)
                -3*mu*(x-(1-mu))**2*((x-(1-mu))**2+y**2+z**2)**(-5/2)
                + mu*((x-(1-mu))**2+y**2+z**2)**(-3/2)
                + (1-mu)*((x+mu)**2+y**2+z**2)**(-3/2)
                -1)
    
    def _d2Omega_dy2(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                -3*(1-mu)*y**2*((x+mu)**2+y**2+z**2)**(-5/2)
                -3*mu*y**2*((x-(1-mu))**2+y**2+z**2)**(-5/2)
                + mu*((x-(1-mu))**2+y**2+z**2)**(-3/2)
                + (1-mu)*((x+mu)**2+y**2+z**2)**(-3/2)
                -1        
                )
        
    def _d2Omega_dz2(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                -3*(1-mu)*z**2*((x+mu)**2+y**2+z**2)**(-5/2)
                -3*mu*z**2*((x-(1-mu))**2+y**2+z**2)**(-5/2)
                + mu*((x-(1-mu))**2+y**2+z**2)**(-3/2)
                + (1-mu)*((x+mu)**2+y**2+z**2)**(-3/2)       
                )
    
    def _d2Omega_dxdy(self,Y):
        
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
    
        return(
                -3*(1-mu)*(x+mu)*y*((x+mu)**2+y**2+z**2)**(-5/2)
                -3*mu*(x-(1-mu))*y*((x-(1-mu))**2+y**2+z**2)**(-5/2)
                )
    
    def _d2Omega_dxdz(self,Y):
    
        x = Y[0]
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                -3*(1-mu)*(x+mu)*z*((x+mu)**2+y**2+z**2)**(-5/2)
                -3*mu*(x-(1-mu))*z*((x-(1-mu))**2+y**2+z**2)**(-5/2)
                )
    
    def _d2Omega_dydz(self,Y):
        
        x = Y[0] 
        y = Y[1]
        z = Y[2]
        mu = self.mu
        
        return(
                -3*(1-mu)*y*z*((x+mu)**2+y**2+z**2)**(-5/2)
                -3*mu*y*z*((x-(1-mu))**2+y**2+z**2)**(-5/2)
                )