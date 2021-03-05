# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""


from linares_plot import *
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
import scipy.signal


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



############################################################################################################




def model_radial_linear2(totalTime, targ_onset, presentation_period, position, 
                         N=512, rint = 1, rext = 6,
                         tauE=9, tauI=4,  
                         I0E=0.1, I0I=0.5, 
                         GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
                         NsigE=0.8, NsigI=1.7, 
                         plot_connectivity=False, plot_rate=False, save_RE=False):
    #
    st_sim =time.time()
    dt=2
    nsteps=int(floor(totalTime/dt));
    rE=zeros((N,1));
    rI=zeros((N,1));
    #Connectivities
    v_E=zeros((N));
    v_I=zeros((N));
    WE=zeros((N,N));
    WI=zeros((N,N));
    ###    
    ###     
    radii = np.linspace(rint, rext, N) ##all radii (linear)
    #### sigmasE (taking the kappaE in radius 1 in the angular model)
    ser1 = 1/300
    ke1=1.  
    ke2=1.7     ##above 1 to be supralinear (inconsistency: the change in kappaE is lesss stepper then in kappaI)         
    ke3=0.05             
    SE = ke1*ser1*radii**ke2 + ke3 ##all sigmasE (supralinear increase)
    #### sigmasI (taking the kappaI in radius 1 in the angular model) (alth¡hough I had to add the constant to correct)
    sir1 = 1/30 
    ki1=0.5
    ki2= 1.6  ##above 1 to be supralinear
    ki3=0.2
    SI = ki1*sir1*radii**ki2 + ki3 ##all sigmasI (supralinear increase)
    ###
    ### connectivity profile for E and I in the radial dimension
    for j in range(0, N): #para cada distancia hay una sigma diferente
        for i in range(0, N): #dentro de cada distancia se calcula la conectividad con el resto teniendo en cuenta una j concreta (una sigma especifica)
            v_E[i]= 1/(sqrt(2*pi)*SE[j])*e**(-((radii[i]- radii[j])**2)/(2*(SE[j]**2)))
            v_I[i]= 1/(sqrt(2*pi)*SI[j])*e**(-((radii[i]- radii[j])**2)/(2*(SI[j]**2)))
        ####
        WE[:,j]=v_E;
        WI[:,j]=v_I;
    ###
    # Plot of the connectivity profile
    if plot_connectivity ==True:
        plt.figure()
        p_cols=['#98c1d9', '#ee6c4d' ]
        for con_w in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450]:
            plt.plot(WE[con_w, :], p_cols[0])
            plt.plot(WI[con_w, :], p_cols[1])
        plt.xlabel('eccentricity (cm)')
        plt.ylabel('pdf')
        plt.gca().spines['right'].set_visible(False)  # aesthetics                                                                              # remove right spines
        plt.gca().spines['top'].set_visible(False)                                                                                  # remove top spines
        plt.gca().get_xaxis().tick_bottom()                                                                                         
        plt.gca().get_yaxis().tick_left()
        plt.gca().tick_params(direction='in') #direction
        plt.plot(WE[475, :], p_cols[0], label='WE')
        plt.plot(WI[475, :], p_cols[1], label='WI')
        plt.ylim(0,8)
        plt.yticks([0,4,8])
        plt.xlim(-10, 520)
        plt.xticks([0, int(512/2), 512], [str(rint), '', str(rext)])
        l = plt.legend(loc=1, frameon=False, prop={'size': 20})
        for i_h, h_idx in enumerate(['WE', 'WI']):
            l.get_texts()[i_h].set_text(h_idx)
            l.legendHandles[i_h].set_visible(False);
            l.get_texts()[i_h].set_color(p_cols[i_h]);
        #
        plt.show(block=False)
    ##
    ###
    # Stimulus profile
    stimulus = zeros((N));
    for j in range(0, N): #para cada distancia hay una sigma diferente
        for i in range(0, N): #dentro de cada distancia se calcula la conectividad con el resto teniendo en cuenta una j concreta (una sigma especifica)
            stimulus[i]= 1/(sqrt(2*pi)*SE[j])*e**(-((position- radii[i])**2)/(2*(SE[j]**2))) 
    
    stimulus=reshape(stimulus, (N,1))
    ###
    ###
    stimon = floor(targ_onset/dt);
    stimoff = floor(targ_onset/dt) + floor(presentation_period/dt) ;
    #Simulation
    #generation of the noise and the connectivity between inhib and exit
    RE=zeros((N,nsteps));
    RI=zeros((N,nsteps));
    f = lambda x : x*x*(x>0)*(x<1) + reshape(array([cmath.sqrt(4*x[i]-3) for i in range(0, len(x))]).real, (N,1)) * (x>=1)
    ### diferential equations
    for i in range(0, nsteps):
        noiseE = NsigE*random.randn(N,1);
        noiseI = NsigI*random.randn(N,1);
        #differential equations for connectivity
        IE= GEE*dot(WE,rE) - GIE*dot(WI,rI) + I0E*ones((N,1));
        II= GEI*dot(WE,rE) +  (I0I-GII*mean(rI))*ones((N,1));
        #
        if i>stimon and i<stimoff:
            IE=IE+stimulus;
            II=II+stimulus;
        #
        #rates of exit and inhib
        rE = rE + (f(IE) - rE + noiseE)*dt/tauE;
        rI = rI + (f(II) - rI + noiseI)*dt/tauI;
        rEr=reshape(rE, N)
        rIr=reshape(rI, N)
        #drawnow
        RE[:,i] = rEr;
        RI[:,i] = rIr;
    #
    #
    if plot_rate==True:
        #### plot dynamics
        fig = plt.figure()
        plt.title('')
        plt.plot(rE)
        plt.xlabel('neuron')
        plt.ylabel('rate (Hz)')
        plt.show(block=False)

    ###
    ###
    final_readout = decode_rE(rE)
    final_radius = final_readout * (float((rext-rint)) / 360) + rint ##conversion a espacio (rint, rext)
    error =  position - final_radius
    error = round(error, 3)
    ### if error>0 means attraction to fixation
    ### if error<0 means repulsion to fixation
    if save_RE==True:
        return error, position, totalTime, RE
    else:
        return error, position, totalTime
    #
##



###############################################
###############################################
############################################### plot the heatmap nice
###############################################
###############################################


def simulation_heatmap_rad(RE, time_simulation, position, target_onset, pres_period, rext=6, rint=1):
    pal_cyan = sns.color_palette("RdBu_r", n_colors=200)[40:] #RdBu_r
    #
    dims=np.shape(RE)
    dimN = dims[0]
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(RE, cmap=pal_cyan, vmin=0, vmax=8,  cbar=True, 
                cbar_kws={"shrink": .82, 'ticks' : [0,2,4,6,8], 'label': 'rate (Hz)'})
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.figure.axes[-1].tick_params(labelsize=20)
    plt.gca().set_ylabel('')
    plt.gca().set_xlabel('')
    plt.gca().set_title('')
    p_stim = (position-rint) * dims[0]/(rext-rint)
    #
    stimon = target_onset/2
    stimoff = (target_onset + pres_period) / 2
    #
    plt.gca().plot([stimon, stimon+400], [p_stim, p_stim], ls='--', color ='blue', linewidth=1) 
    #
    plt.gca().set_xticks([])
    plt.gca().set_xticklabels([])
    #
    plt.gca().set_yticks([0, int(dimN/4), int(dimN/2),  int(3*dimN/4), int(dimN) ])
    plt.gca().set_yticklabels([str(rint),'',str((rext+rint)/2), '', str(rext)], fontsize=20)
    #
    plt.gca().set_xlabel('', fontsize=20);
    plt.gca().set_ylabel('neuron preferred (cm)', fontsize=20);
    plt.gca().set_ylim(dimN+60, -45)
    ###
    ##line stims 
    s1on=stimon
    s1off=stimoff
    plt.plot([0, s1on], [-15, -15], 'k-', linewidth=2)
    plt.plot([s1on, s1on], [-15, -40], 'k-', linewidth=2)
    plt.plot([s1on, s1off], [-40, -40], 'k-', linewidth=2)
    plt.plot([s1off, s1off], [-15, -40], 'k-', linewidth=2)
    plt.plot([s1off, dims[1]], [-15, -15], 'k-', linewidth=2)
    #
    #time
    x1sec = 1000 * dims[1] / time_simulation
    plt.plot([dims[1]-x1sec, dims[1]], [dimN+30, dimN+30], 'k-', linewidth=2)
    plt.text(dims[1]-300, 600, '1s', fontsize=20);
    plt.show()