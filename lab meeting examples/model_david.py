# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

import math
from math import floor, exp, sqrt, pi
import cmath
import numpy
from numpy import e, cos, zeros, arange, roll, where, random, ones, mean, reshape, dot, array, flipud, pi, exp, dot, angle, degrees, shape, linspace
import matplotlib.pyplot as plt
from itertools import chain
import scipy
from scipy import special
import numpy as np 
import seaborn as sns
import time
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.signal
from scipy.optimize import curve_fit 
from scipy import stats



## accesory functions
def decode_rE(rE, a_ini=0, a_fin=360, N=512):
    #Population vector for a given rE
    # return ( angle in radians, absolut angle in radians, abs angle in degrees )
    N=len(rE)
    Angles = np.linspace(a_ini, a_fin, N) 
    angles=np.radians(Angles)
    rE = np.reshape(rE, (1,N))
    R = numpy.sum(np.dot(rE,exp(1j*angles)))/numpy.sum(rE) ## finding it with imagianry numbers
    angle_decoded = np.degrees(np.angle(R))
    if angle_decoded<0:
        angle_decoded = 360+angle_decoded
    
    return angle_decoded



def closest(lst, K):       
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
      


def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)


def most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return(itm) 


def Interference_effects(target, response, reference):
    #input list of target, list of responses and list of references
    #Error_interference; positive for attraction and negative for repulsion
    #######
    #Decimals to get
    decimals=2
    ####
    interferences=[]
    for i in range(0, len(target)):
        angle_err_abs=abs(target[i] - response[i])
        if circ_dist(np.array(response)[i], np.array(reference)[i])<=circ_dist(np.array(target)[i], np.array(reference)[i]):
            Err_interference=round( angle_err_abs, decimals) 
        else:
            Err_interference=round( -angle_err_abs, decimals)
        interferences.append(Err_interference)
    
    return interferences



def viz_polymonial(X, y, poly_reg, pol_reg):
    plt.figure()
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Fit Bump')
    plt.xlabel('Neuron')
    plt.ylabel('rate')
    plt.show(block=False)
    return



def model_I0E(center_angle, size_windows, N=512):
    inf, sup = np.radians(center_angle) - np.radians(size_windows/2), np.radians(center_angle) + np.radians(size_windows/2)
    new_I0E=[]
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)];
    for i in theta:
        if i < closest(theta, inf):
            new_I0E.append(0)
        elif i>= closest(theta, inf) and i <=closest(theta, sup):
            new_I0E.append(1)
        else:
            new_I0E.append(0)
    
    return np.reshape(np.array(new_I0E), (N,1))



def model_I0E_guass(center_angle, N=512):
    mu = 0
    variance = 0.6
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 8*sigma, mu + 8*sigma, 512)
    y = 2*stats.norm.pdf(x, mu, sigma)
    center_pdf = 180 #deg 
    rolling_angle = center_angle - center_pdf #deg
    rolling_angle_neur  = int(rolling_angle*N/360)
    new_I0E = np.roll(y, rolling_angle_neur )
    return np.reshape(np.array(new_I0E), (N,1))
    

    
def model_I0E_constant(value, N=512):
    y=[value for x in range(N)]
    return np.reshape(np.array(y), (N,1))





def model_I0E_flat(center_angle, size_windows=100, n_ramping=10, N=512):
    inf, sup = np.radians(center_angle) - np.radians(size_windows/2), np.radians(center_angle) + np.radians(size_windows/2)
    inf_r, sup_r = inf  - np.radians(n_ramping), sup  + np.radians(n_ramping)
    #print(inf,sup)
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)];
    len_ramping = np.where(np.array(theta) == closest(theta, inf))[0][0] - np.where(np.array(theta) == closest(theta, inf_r))[0][0]
    line = np.linspace(0, 1, len_ramping)
    new_I0E=[]
    a=-1
    for i in theta:
        if i < closest(theta, inf_r):
            new_I0E.append(0)
        elif i>= closest(theta, inf_r) and i <closest(theta, inf):
            a=a+1
            new_I0E.append(line[a])
            #print(a)
        elif i>= closest(theta, inf) and i <=closest(theta, sup):
            new_I0E.append(1)
            a=0
        elif i>= closest(theta, sup) and i <closest(theta, sup_r):
            a=a-1
            new_I0E.append(line[a])
            
        elif i>=closest(theta, sup_r):
            new_I0E.append(0)
    
    return np.reshape(np.array(new_I0E), (N,1))






### Functions to make the fit
#### von misses
def von_misses(x,mu,k):
    return (exp( k * cos(x-mu))) / (2*pi*scipy.special.i0(k)) 


#### bi von misses
def bi_von_misses(x,mu1,k1,mu2,k2):
    return von_misses(x,mu1,k1) + von_misses(x,mu2,k2)


#### gaussian
def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)




####



##model 

### he quitado toda la parte del fit
### segun si el parámtero save_RE es True or False se usará para correr en paralelo (False) o para ejemplos (True)


def model1(totalTime, targ_onset, dist_onset, presentation_period, angle_separation, order_2,  
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4,
    GEE=0.016, GEI=0.015, GIE=0.012 , GII=0.007, sigE=0.06, sigI=0.04,
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=20,
    N=512, plot_connectivity=True, plot_dyniamic=True, plot_heatmap=True, plot_fit=True, save_RE=True):
    ##
    #Temporal and spatial settings
    st_sim =time.time()
    dt=2;
    nsteps=int(floor(totalTime/dt));
    origin = pi;
    targ_offset = targ_onset + presentation_period;
    targon = floor(targ_onset/dt);
    targoff = floor(targ_offset/dt);
    diston = floor(dist_onset/dt);    
    dist_offset = dist_onset  + presentation_period;
    distoff = floor(dist_offset/dt);
    stim_sep =  angle_separation*pi/360
    angle_target=180+angle_separation/2
    angle_distractor=180-angle_separation/2
    # Connectivitiess
    v_E=np.zeros((N));
    v_I=np.zeros((N));
    WE=np.zeros((N,N));
    WI=np.zeros((N,N));
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)];
    for i in range(0, N):
        v_E_new=[e**(kappa_E*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_E)) for f in range(0, len(theta))]    
        v_I_new=[e**(kappa_I*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_I)) + k_inhib for f in range(0, len(theta))] 
        #    
        vE_NEW=np.roll(v_E_new,i)
        vI_NEW=np.roll(v_I_new,i) 
        # 
        WE[:,i]=vE_NEW
        WI[:,i]=vI_NEW
    
    #   
    ###### Stimuli
    target=np.zeros((N))
    distractor=np.zeros((N))
    for i in range(0, N):
        target[i]=e**(kappa_stim*cos(theta[i] + origin - stim_sep))  / (2*pi*scipy.special.i0(kappa_stim)) ## target at (origin + sep)
        distractor[i]=e**(kappa_stim*cos(theta[i] + origin + stim_sep)) / (2*pi*scipy.special.i0(kappa_stim)) ## distractor at (origin -sep)
    
    
    #
    noise_stim =  0 #np.random.normal(0, 0.01, N)
    target = target+ noise_stim
    target=reshape(target, (N,1))
    distractor = distractor + noise_stim
    distractor=reshape(distractor, (N,1)) 
    # Model
    mf=1
    rE=np.zeros((N,1));
    rI=np.zeros((N,1)); 
    u = np.ones((N,1))*U
    x = np.ones((N,1))
    RE=np.zeros((N,nsteps));
    RI=np.zeros((N,nsteps));
    p_u=np.ones((N,nsteps));
    p_x=np.ones((N,nsteps));
    #
    ## Different quadrant_selectivity options gaussian
    I0E_open =  1.2 
    I0E_close= 0.5 
    quadrant_selectivity_close = model_I0E_constant(I0E_close)
    quadrant_selectivity_open =  model_I0E_flat( np.degrees(origin + stim_sep))*(I0E_open-I0E_close) + I0E_close
    quadrant_selectivity_standard = model_I0E_constant(I0E_open)
    ##
    ## state before the simulation strats (depending on the cue)
    quadrant_selectivity = quadrant_selectivity_standard
    ##
    if order_2 == True: 
        quadrant_selectivity = quadrant_selectivity_close  
    else:
        quadrant_selectivity= quadrant_selectivity_standard  
    ##
    ### currents duing the simulation
    f = lambda x : x*x*(x>0)*(x<1) + reshape(array([cmath.sqrt(4*x[i]-3) for i in range(0, len(x))]).real, (N,1)) * (x>=1)
    for i in range(0, nsteps):
        noiseE = sigE*random.randn(N,1);
        noiseI = sigI*random.randn(N,1);
        #differential equations for current
        IE= GEE*dot(WE, (rE/mf*u*x)) - GIE*dot(WI,rI/mf) + quadrant_selectivity;
        II= GEI*dot(WE,rE/mf) +  (I0I-GII*mean(rI/mf))*ones((N,1));
        #
        ## presentation stims (target and distractor)
        if i>targon and i<targoff:
            IE=IE+target;
            II=II+target;
        #
        if i>diston and i<distoff: 
            IE=IE+distractor;
            II=II+distractor;
        #
        ## state depending on the time. Changes one you detect the stimulus!
        if i< targon:
            if order_2==True: 
                quadrant_selectivity = quadrant_selectivity_close #always closed for order 2 until you detect the stimui
            elif order_2==False:
                quadrant_selectivity = quadrant_selectivity_standard # all opened
        else:
            quadrant_selectivity = quadrant_selectivity_open ## opened just around, the rest is closed
        #####################################################
        #rates of exit and inhib   
        rE = rE*mf + (f(IE)*mf - rE*mf + noiseE)*dt/tauE;
        rI = rI*mf + (f(II)*mf - rI*mf + noiseI)*dt/tauI;
        ### formulas for synaptic plasticity: paper mongillo 2008
        u = u + ((U - u) / tauf + U*(1-u)*(rE/mf)/1000)*dt;
        x = x + ((1 - x)/taud - u*x*(rE/mf)/1000)*dt;
        rEr=np.reshape(rE, N)*10
        rIr=np.reshape(rI, N)
        ur=np.reshape(u, N)
        xr=np.reshape(x, N)
        #append
        RE[:,i] = rEr;
        RI[:,i] = rIr;
        p_u[:,i] = ur;
        p_x[:,i] = xr;   

        rE=rE/mf 
        rI=rI/mf 
    #
    #### Interference
    interference = Interference_effects( [decode_rE(target)], [decode_rE(rE)], [decode_rE(distractor)])[0]
    
    
    #### Plots 
    # Plot of the connectivity profile
    if plot_connectivity ==True:
        plt.figure()
        plt.plot(WE[250, :], label='E')
        plt.plot(WI[250, :], label = 'I')
        plt.ylim(0,6)
        plt.gca().spines['right'].set_visible(False) #no right axis
        plt.gca().spines['top'].set_visible(False) #no  top axis
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.title('Connectivity WE & WI')
        plt.show(block=False)
        plt.figure()
        plt.plot(WE[250, :] - WI[250, :] , label='E-I')
        plt.gca().spines['right'].set_visible(False) #no right axis
        plt.gca().spines['top'].set_visible(False) #no  top axis
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.title('Effective onnectivity')
        plt.show(block=False)
    
    #### plot dynamics
    p_targ = int((N * np.degrees(origin + stim_sep))/360)
    if plot_dyniamic==True:        
        fig = plt.figure()
        fig.tight_layout()
        fig.set_size_inches(13, 4)
        fig.add_subplot(131)
        p_targ = int((N * np.degrees(origin + stim_sep))/360)
        plt.title('Synaptic dynamics for target')
        plt.plot(p_u[p_targ, :], 'b', label='prob. release')
        plt.plot(p_x[p_targ, :], 'r', label='pool vesicles')
        plt.xlabel('time (ms)')
        plt.xticks(np.linspace(0, int(totalTime/dt), 7), np.array(np.linspace(0, int(totalTime/dt), 7)*2, dtype=int))        
        plt.gca().spines['right'].set_visible(False)  # aesthetics                                                                              # remove right spines
        plt.gca().spines['top'].set_visible(False)                                                                                  # remove top spines
        plt.gca().get_xaxis().tick_bottom()                                                                                         
        plt.gca().get_yaxis().tick_left()
        plt.gca().tick_params(direction='in') #direction
        plt.legend(frameon=False)
        fig.add_subplot(132)
        p_dist= int((N * np.degrees(origin - stim_sep))/360)
        plt.title('Synaptic dynamics for distractor')
        plt.plot(p_u[p_dist, :], 'b', label='prob.y release')
        plt.plot(p_x[p_dist, :], 'r', label='pool vesicles')
        plt.xlabel('time (ms)')
        plt.xticks(np.linspace(0, int(totalTime/dt), 7), np.array(np.linspace(0, int(totalTime/dt), 7)*2, dtype=int)) 
        plt.gca().spines['right'].set_visible(False)  # aesthetics                                                                              # remove right spines
        plt.gca().spines['top'].set_visible(False)                                                                                  # remove top spines
        plt.gca().get_xaxis().tick_bottom()                                                                                         
        plt.gca().get_yaxis().tick_left()
        plt.gca().tick_params(direction='in') #direction
        plt.legend(frameon=False)
        fig.add_subplot(133)
        plt.title('Rate dynamics')
        plt.plot(RE[p_targ, :], 'b', label='target')
        plt.plot(RE[p_dist, :], 'r', label='distractor')
        plt.xlabel('time (ms)')
        plt.xticks(np.linspace(0, int(totalTime/dt), 7), np.array(np.linspace(0, int(totalTime/dt), 7)*2, dtype=int)) 
        plt.ylabel('rate (Hz)')
        plt.gca().spines['right'].set_visible(False)  # aesthetics                                                                              # remove right spines
        plt.gca().spines['top'].set_visible(False)                                                                                  # remove top spines
        plt.gca().get_xaxis().tick_bottom()                                                                                         
        plt.gca().get_yaxis().tick_left()
        plt.gca().tick_params(direction='in') #direction
        plt.legend(frameon=False)
        plt.show(block=False)
    #
    #### plot heatmap
    if plot_heatmap==True:
        p_dist= int((N * np.degrees(origin - stim_sep))/360)
        plt.figure(figsize=(9,6))
        sns.heatmap(RE, cmap='viridis')
        plt.title('BUMP activity')
        plt.ylabel('Angle')
        plt.xlabel('time')
        plt.plot([targon, nsteps], [p_targ, p_targ], '--b',) ## flipped, so it is p_target 
        plt.plot([diston, nsteps], [p_dist, p_dist], '--r',) ## flipped, so it is p_target 
        plt.yticks([])
        plt.xticks([])
        plt.yticks([N/8, 3*N/8, 5*N/8, 7*N/8 ] ,['45','135','225', '315'])
        plt.plot([targ_onset/2, targ_onset/2,], [0+20, N-20], 'k-', label='onset')
        plt.plot([targ_offset/2, targ_offset/2,], [0+20, N-20], 'k--', label='offset')
        plt.plot([dist_onset/2, dist_onset/2,], [0+20, N-20], 'k-')
        plt.plot([dist_offset/2, dist_offset/2,], [0+20, N-20], 'k--')
        plt.legend(frameon=False)
        plt.show(block=False)
    
    ## print time consumed in each simulation
    end_sim =time.time()
    total_time= end_sim - st_sim 
    total_time = round(total_time, 1)
    #print('Simulation time: ' + str(total_time) + 's')
    ##

#     ### Fit
#     y=np.reshape(rE, (N)) 
#     X=np.reshape(np.linspace(-pi, pi, N), N)
#     df_n_p=pd.DataFrame()
#     df_n_p['rE'] = rE.reshape(512)
#     peaks_list=[]
#     for n_w_s in range(1, 100):
#         r = df_n_p['rE'].rolling(window=n_w_s).mean()
#         number_of_bumps = len(scipy.signal.find_peaks(r, 2)[0]) 
#         peaks_list.append(number_of_bumps)
#     #
#     if number_of_bumps == 0:
#         if peaks_list==[0 for i in range(len(peaks_list))]:
#             number_of_bumps = 0
#         else:
#             peaks_list[:] = (value for value in peaks_list if value != 0)
#             number_of_bumps=most_frequent(peaks_list)
#     #
#     number_of_bumps=most_frequent(peaks_list)
#     ### Fit
#     if number_of_bumps ==2:
#         target_pos_pi_pi = decode_rE(target) * 2*pi / 360 -pi
#         distractor_pos_pi_pi = decode_rE(distractor) * 2*pi / 360 -pi
#         param, covs = curve_fit(bi_von_misses, X, y, p0=[target_pos_pi_pi + pi, -36,  distractor_pos_pi_pi - pi, -36]) #p0=[separation, 75, -separation, 75]
#         ans = (exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1])) + (exp( param[3] * cos(X-param[2]))) / (2*pi*scipy.special.i0(param[3])) 
#         estimated_angles=[]
#         for p in [param[0], param[2]]:
#             estim_ang = np.degrees(p)  
#             if estim_ang<0: 
#                 estim_ang = 360+estim_ang
#             estimated_angles.append(estim_ang)
        
        
#         ###decoded angles
#         decoded_target = closest(estimated_angles, decode_rE(target))
#         decoded_distractor = closest(estimated_angles, decode_rE(distractor))
#         ###bias
#         bias_target = Interference_effects(  [decode_rE(target)],  [decoded_target],  [decode_rE(distractor)])[0]
#         bias_dist = Interference_effects(  [decode_rE(distractor)],  [decoded_distractor],  [decode_rE(target)])[0]
#         final_bias = [bias_target, bias_dist]
#         skip_r_sq=False
#         success=True

#     #
#     elif number_of_bumps ==1:
#         target_pos_pi_pi = decode_rE(target) * 2*pi / 360 -pi
#         distractor_pos_pi_pi = decode_rE(distractor) * 2*pi / 360 -pi
#         ###param, covs = curve_fit(bi_von_misses, X, y, p0=[target_pos_pi_pi + pi, -36,  distractor_pos_pi_pi - pi, -36]) #p0=[separation, 75, -separation, 75]
#         ###ans = (exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1])) + (exp( param[3] * cos(X-param[2]))) / (2*pi*scipy.special.i0(param[3])) 
#         param, covs = curve_fit(gauss, X, y) #p0=[separation, 75, -separation, 75]
#         ans = param[2]*exp(-(X-param[0])**2/2/param[1]**2)      #(exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1]))  
#         estimated_angles=[]
#         for p in [param[0]]: #, param[2]]:
#             estim_ang = np.degrees(p) + 180
#             if estim_ang<0:
#                 estim_ang = 360+estim_ang
#             estimated_angles.append(estim_ang)
        
#         decoded_target = closest(estimated_angles, decode_rE(target))
#         bias_target = Interference_effects(  [decode_rE(target)],  [decoded_target],  [decode_rE(distractor)])[0]
#         bias_dist = 0
#         skip_r_sq=False
#         success=True
#     #
#     else:
#         print('Error simultaion')
#         bias_target =999
#         bias_dist =999
#         final_bias=[999, 999]
#         plot_fit=False
#         skip_r_sq=True
#         r_squared=0
#         success=False ## to eliminate wrong simulations easily at the end
#     #error_fit (r_squared)
#     if skip_r_sq==False:
#         residuals = y - ans
#         ss_res = np.sum(residuals**2)
#         ss_tot = np.sum((y-numpy.mean(y))**2)
#         r_squared = 1 - (ss_res / ss_tot)
#     #Plot Fit
#     if plot_fit==True:
#         plt.figure()
#         plt.plot(X, y, 'o', color ='red', label ="data") 
#         plt.plot(X, ans, '--', color ='blue', label ="fit") 
#         plt.legend() 
#         plt.show(block=False) 
    
    
    
    ### Output
    ###return bias_target, bias_dist, number_of_bumps, angle_separation, RE #rE[p_targ][0], I0E
    if save_RE==True:
        return interference, RE
    else:
        return interference
    





 ###############



 ##plotear el hetamap (se necesiat el RE)

def simulation_heatmap(RE, time_simulation, order2, angle_separation, target_onset, distractor_onset, pres_period):
    pal_cyan = sns.color_palette("RdBu_r", n_colors=200)[40:] #RdBu_r
    
    dims=np.shape(RE)
    dimN = dims[0]
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(RE, cmap=pal_cyan, vmin=0, vmax=45,  cbar=True, 
                cbar_kws={"shrink": .82, 'ticks' : [0, 15, 30, 45], 'label': 'rate (Hz)'})
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.figure.axes[-1].tick_params(labelsize=20)
    plt.gca().set_ylabel('')
    plt.gca().set_xlabel('')
    plt.gca().set_title('')
    p_stim = (180+angle_separation/2) * (dims[0]/360)
    p_dist = (180-angle_separation/2) * (dims[0]/360) 

    stimon = target_onset/2
    stimoff = (target_onset + pres_period) / 2
    diston = distractor_onset/2
    distoff = (distractor_onset + pres_period) / 2


    plt.gca().plot([stimon, stimon+400], [p_stim, p_stim], ls='--', color ='blue', linewidth=1) 
    plt.gca().plot([diston, diston+400], [p_dist, p_dist], ls='--', color ='red', linewidth=1) 

    plt.gca().set_xticks([])
    plt.gca().set_xticklabels([])
    #
    plt.gca().set_yticks([0, int(dimN/4), int(dimN/2),  int(3*dimN/4), int(dimN) ])
    plt.gca().set_yticklabels(['0','','180', '', '360'], fontsize=20)

    plt.gca().set_xlabel('', fontsize=20);
    plt.gca().set_ylabel('neuron preferred ($^\circ$)', fontsize=20);
    plt.gca().set_ylim(dimN+60, -45)

    ###

    ##line stims  
    if order2==False:
        s1on=stimon
        s1off=stimoff
        s2on=diston
        s2off=distoff
    else:
        s1on=diston
        s1off=distoff
        s2on=stimon
        s2off=stimoff

    plt.plot([0, s1on], [-15, -15], 'k-', linewidth=2)
    plt.plot([s1on, s1on], [-15, -40], 'k-', linewidth=2)
    plt.plot([s1on, s1off], [-40, -40], 'k-', linewidth=2)
    plt.plot([s1off, s1off], [-15, -40], 'k-', linewidth=2)
    plt.plot([s1off, s2on], [-15, -15], 'k-', linewidth=2)
    plt.plot([s2on, s2on], [-15, -40], 'k-', linewidth=2)
    plt.plot([s2on, s2off], [-40, -40], 'k-', linewidth=2)
    plt.plot([s2off, s2off], [-15, -40], 'k-', linewidth=2)
    plt.plot([s2off, dims[1]], [-15, -15], 'k-', linewidth=2)

    #time
    x1sec = 1000 * dims[1] / time_simulation
    plt.plot([dims[1]-x1sec, dims[1]], [dimN+30, dimN+30], 'k-', linewidth=2)
    plt.text(dims[1]-300, 600, '1s', fontsize=20);
    plt.show()

