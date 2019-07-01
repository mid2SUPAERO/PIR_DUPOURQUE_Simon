# -*- coding: utf-8 -*-
 
 
"""
    ode module
    ==========
    
    Contains a set of functions to solve ode problem, the most elaborate being ode87
    
    @author : Simon Dupourqué 
"""

import numpy as np

def ode87(f,t_ini,t_end,Y0,tol=1e-9,propagate=False,minimal_nb_of_points=10):
    """
        8-7thth order Dorman-Prince solver with adaptative step. 
        
        Requires 13 function evaluations per iteration. Compares 8th and 7th order 
        solutions and actively change step size to match a given precision. 
        Inspired by Vasiliy Govorukhin's ode87 for Matlab.
        
        :param f: The differential equation's function
        :param t_ini : The initial time of integration
        :param t_end: The final time of integration
        :param Y0: Initial state vector
        :param tol: Precision to match
        :param propagate: If True, only returns the final state vector
        :param minimal_nb_of_points: Minimal number of points for the computation
        :type f: function
        :type t_ini: float
        :type t_end: float
        :type Y0: numpy.ndarray
        :type tol: float
        :type propagate: bool
        :type minimal_nb_of_points: int
        
        ..warning:: No loop-exit implemented if the convergence can't be achieved.
    """
    c_i=  np.array([0,1/18,1/12,1/8,5/16,3/8,59/400,93/200,5490023248/9719169821,13/20,1201146811/1299019798,1,1])
    a_i_j = np.transpose(np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0], 
                 [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0],
                 [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0, 0, 0, 0],
                 [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0],
                 [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0],
                 [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0],
                 [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0],
                 [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0]]))      
    b_8 = np.array([ 14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731,   561292985/797845732,   -1041891430/1371343529,  760417239/1151165299, 118820643/751138087, -528747749/2220607170,  1/4])
    b_7 = np.array([ 13451932/455176623, 0, 0, 0, 0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186,   -3867574721/1518517206,   465885868/322736535,  53011238/667516719,                  2/45,    0])

    step_pow = 1/8
    sign = np.sign(t_end - t_ini)
    h = (t_end-t_ini)/minimal_nb_of_points
    n_reject = 0
    updated = False
    
    if not propagate:
        t = np.array([t_ini])      
        Y = np.reshape(np.copy(Y0),(1,)+np.shape(Y0))
  
    else:
        t = t_ini
        Y = np.copy(Y0)
    
    #Minimal step size
    h_min = 16*np.spacing(1)
    h_max = abs((t_end - t_ini)/minimal_nb_of_points)
    
    c_time = t_ini
    
    while 0 < (t_end-c_time)*sign:
        
        #If next step bring the solution beyond the final time
        if ((c_time+h)-t_end)*sign > 0 :
            h = t_end - c_time
        
        updated = False
        
        while not updated:
            
            if not propagate:
                c_Y = Y[-1]
            else :
                c_Y = Y
            
            p_n_i = np.zeros(np.shape(b_8)+np.shape(Y0))
            for i in range(len(b_8)):
                
                t_n_i = c_time + c_i[i]*h               
                Y_n_i = c_Y  + h*np.tensordot(a_i_j[:,i],p_n_i,1)
                p_n_i[i] = f(t_n_i,Y_n_i) 
            
            Y_8 = c_Y + h*np.dot(b_8,p_n_i)
            Y_7 = c_Y + h*np.dot(b_7,p_n_i)
            error_step = np.sqrt(np.dot(Y_8-Y_7,Y_8-Y_7))
            tau = tol*max(np.max(abs(c_Y)),1.0)
            
            #update if solution is precise enough
            if error_step<tau:
                
                if not propagate:
                    t = np.append(t,t[-1]+h)
                    Y = np.vstack((Y,Y_8))
                    c_time = t[-1]
                else:
                    t = t+h
                    Y = np.copy(Y_8)
                    c_time = t
                updated = True
            
            else:
                
                n_reject += 1 
                
            #STEP CONTROL
            if error_step == 0.:
                error_step = 10*np.spacing(1)
            
            h = sign*min(h_max,abs(0.9*h*(tau/error_step)**step_pow))
            h = sign*max(h_min,abs(h))
    
    if not propagate:
        return(t,Y)
    else :
        return(Y)