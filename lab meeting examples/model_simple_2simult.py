

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.signal
from scipy.optimize import curve_fit 



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

# model(totalTime=2000, targ_onset=100,  presentation_period=100, separation=2) 

def model_2s(totalTime, targ_onset, presentation_period, positions, positions2, tauE=9, tauI=4,  n_stims=2, I0E=0.1, I0I=0.5, GEE=0.022, GEI=0.019, 
 GIE=0.01 , GII=0.1, sigE=0.5, sigI=1.6, kappa_E=100, kappa_I=1.75, kappa_stim=100, N=512, plot_connectivity=False, plot_rate=False, 
 plot_hm=True , plot_fit=True, save_RE=False):
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
    p1 = np.radians(positions)
    p2= np.radians(positions2)
    ### p1 goes from 0 -360 and you convert it to 0-2pi radians (0 is fixation and 2pi is limit)    
    ###
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
    ###
    kappas_e_range= np.linspace(100, 300, N) ##100-300 ok
    kappas_e_range = np.flip(kappas_e_range)
    ###
    ###
    for i in range(0, N):
        v_E_new=[e**(kappa_E*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_E)) for f in range(0, len(theta))]    
        v_I_new=[e**(kappa_I*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_I)) for f in range(0, len(theta))]
        ###    
        vE_NEW=roll(v_E_new,i)
        vI_NEW=roll(v_I_new,i) #to roll
        ###    
        WE[:,i]=vE_NEW
        WI[:,i]=vI_NEW
    #
    # Plot of the connectivity profile
    if plot_connectivity ==True:
        plt.figure()
        p_cols=['#98c1d9', '#ee6c4d' ]
        for con_w in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450]:
            plt.plot(WE[con_w, :], p_cols[0])
            plt.plot(WI[con_w, :], p_cols[1])
        plt.xlabel('eccentricity ($^\circ$)')
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
        plt.xticks([0, int(512/2), 512], ['0', '180', '360'])
        l = plt.legend(loc=1, frameon=False, prop={'size': 20})
        for i_h, h_idx in enumerate(['WE', 'WI']):
            l.get_texts()[i_h].set_text(h_idx)
            l.legendHandles[i_h].set_visible(False);
            l.get_texts()[i_h].set_color(p_cols[i_h]);
        #
        plt.show(block=False)
    ##
    # Stims
    stimulus1=zeros((N));
    stimulus2=zeros((N));
    for i in range(0, N):
        stimulus1[i]=e**(kappa_stim*cos(theta[i] - p1)) / (2*pi*scipy.special.i0(kappa_stim))
        stimulus2[i]=e**(kappa_stim*cos(theta[i] - p2)) / (2*pi*scipy.special.i0(kappa_stim))
    stimulus=np.array(stimulus1) + np.array(stimulus2)
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
        noiseE = sigE*random.randn(N,1);
        noiseI = sigI*random.randn(N,1);
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
    if plot_hm==True:
        #### plot heatmap
        RE_sorted=RE
        plt.figure(figsize=(9,6))
        sns.heatmap(RE_sorted, cmap='viridis')
        plt.title('BUMP activity')
        plt.ylabel('eccentricity')
        plt.xlabel('time')
        #plt.plot([stimon, nsteps], [p_targ2, p_targ2], '--b',) ## flipped, so it is p_target 
        #plt.plot([stimon, nsteps], [p_targ1, p_targ1], '--r',) ## flipped, so it is p_target 
        plt.yticks([])
        plt.xticks([])
        plt.yticks([N/8, 3*N/8, 5*N/8, 7*N/8 ] ,['45','135','225', '315'])
        plt.plot([stimon, stimon,], [0+20, N-20], 'k-', label='onset')
        plt.plot([stimoff, stimoff,], [0+20, N-20], 'k--', label='offset')
        plt.plot([stimon, stimon,], [0+20, N-20], 'k-')
        plt.plot([stimoff, stimoff,], [0+20, N-20], 'k--')
        plt.legend()
        plt.show(block=False)
    ###
    ###
    final_readout = decode_rE(rE)
    error =  np.degrees(p1)  - final_readout 
    ### if error>0 means attraction to fixation
    ### if error<0 means repulsion to fixation
    if save_RE==True:
        return error, positions, RE
    else:
        return  error, positions, I0E, max(rE)
    #


### plotear el hetamap (se necesiat el RE)

def simulation_heatmap_R(RE, time_simulation, position, target_onset, pres_period):
    pal_cyan = sns.color_palette("cividis", n_colors=200)[40:] #RdBu_r
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
    p_stim = position * (dims[0]/360)
    #
    stimon = target_onset/2
    stimoff = (target_onset + pres_period) / 2
    #
    #plt.gca().plot([stimon, stimon+400], [p_stim, p_stim], ls='--', color ='blue', linewidth=1) 
    #
    plt.gca().set_xticks([])
    plt.gca().set_xticklabels([])
    #
    plt.gca().set_yticks([0, int(dimN/4), int(dimN/2),  int(3*dimN/4), int(dimN) ])
    plt.gca().set_yticklabels(['0','','180', '', '360'], fontsize=20)
    #
    plt.gca().set_xlabel('', fontsize=20);
    plt.gca().set_ylabel('neuron ($^\circ$)', fontsize=20);
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


####
## Example
# bias, total_sep, GEE, rE, error, success, number_of_bumps = model(totalTime=2000, targ_onset=100,  presentation_period=350, separation=5, tauE=9, tauI=4,  n_stims=2, I0E=0.1, I0I=0.5, GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.5, sigI=1.6, kappa_E=200, kappa_I=20, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=True) 
# print(bias, number_of_bumps, total_sep)


## 2 bumps radial
# from joblib import Parallel, delayed
# import multiprocessing

# numcores = multiprocessing.cpu_count() - 1
# distances_test = [5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24]
# kappa_e_test = [100, 200] #[0.024, 0.025]
# rep_dist = 50
# n_kappas= len(kappa_e_test)
# n_sepa = len(distances_test)

# separations= distances_test * rep_dist * n_kappas

# kappas_e=[]
# for k in kappa_e_test:
#     kappas_e = kappas_e + [k]*n_sepa*rep_dist

# results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=sep, tauE=9, tauI=4,  n_stims=2, I0E=0.1, I0I=0.5,
#  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.8, sigI=1.6, kappa_E=kappas, kappa_I=20, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for sep, kappas in zip(separations, kappas_e)) 

# biases = [results[i][0] for i in range(len(results))]
# separationts = [results[i][1] for i in range(len(results))]   
# kappas = [results[i][2] for i in range(len(results))]                                                             
# succs = [results[i][5] for i in range(len(results))]   

# df=pd.DataFrame({'bias':biases, 'separation':separationts, 'kappas_E':kappas, 'success':succs })
# df.to_excel('simulations_2bumps.xlsx')



### 1 bump 3 ridus
# from joblib import Parallel, delayed
# import multiprocessing

# numcores = multiprocessing.cpu_count() - 1

# kappa_e_test = [100, 150, 200] #[0.024, 0.025]
# rep_dist = 200
# kappas_e=kappa_e_test*rep_dist

# results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5,
#     GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.8, sigI=1.6, kappa_E=kappas, kappa_I=20, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False ,
#      plot_fit=False)  for  kappas in  kappas_e)

# biases = [results[i][0] for i in range(len(results))]
# kappas = [results[i][2] for i in range(len(results))]                                                             
# succs = [results[i][5] for i in range(len(results))]   

# df=pd.DataFrame({'bias':biases, 'kappas_E':kappas, 'success':succs })
# df.to_excel('single_item_drift_eccentricity.xlsx')

# df = df.loc[df['success']==True] 
# plt.figure(figsize=(8,6))
# linares_plot( x="kappas_E", y="bias", order=[100, 150, 200],  pallete='viridis', alpha=1, point_size=3, df=df) 
# plt.title('Drift with eccentricity separation', fontsize=15) #condition title
# plt.gca().spines['right'].set_visible(False) #no right axis
# plt.gca().spines['top'].set_visible(False) #no  top axis
# plt.gca().get_xaxis().tick_bottom()
# plt.gca().get_yaxis().tick_left()
# #plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
# plt.show(block=False)




    # ## print time consumed
    # end_sim =time.time()
    # total_time= end_sim - st_sim 
    # total_time = round(total_time, 1)
    # print('Simulation time: ' + str(total_time) + 's')

    # ###### Fit
    # def von_misses(x,mu,k):
    #     return (exp( k * cos(x-mu))) / (2*pi*scipy.special.i0(k)) 

    # def bi_von_misses(x,mu1,k1,mu2,k2):
    #     return von_misses(x,mu1,k1) + von_misses(x,mu2,k2)

    # ##
    # y=np.reshape(rE, (N)) 
    # X=np.reshape(np.linspace(-pi, pi, N), N)

    # ### Fit
    # df_n_p=pd.DataFrame()
    # df_n_p['rE'] = rE.reshape(512)
    # r = df_n_p['rE'].rolling(window=20).mean()
    # number_of_bumps = len(scipy.signal.find_peaks(r, 2)[0]) 

    # if number_of_bumps ==2:
    #     param, covs = curve_fit(bi_von_misses, X, y, p0=[p1, 75, -p1, 75])
    #     ans = (exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1])) + (exp( param[3] * cos(X-param[2]))) / (2*pi*scipy.special.i0(param[3])) 
    #     estimated_angle_1=np.degrees(param[0]+pi)  
    #     estimated_angle_2=np.degrees(param[2]+pi)  
    #     estimated_angles = [estimated_angle_1, estimated_angle_2]
    #     estimated_angles.sort()
    #     bias_b1 = estimated_angles[0] -  np.degrees(p1) ### change the error stuff
    #     bias_b2 =  np.degrees(p2) - estimated_angles[1]
    #     final_bias = [bias_b1, bias_b2]
    #     skip_r_sq=False
    #     success=True

    # elif number_of_bumps ==1:
    #     param, covs = curve_fit(von_misses, X, y)
    #     ans = (exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1])) 
    #     estimated_angle=np.degrees(param[0]+pi)  
    #     bias_b1 = estimated_angle - np.degrees(p1)
    #     bias_b2 = np.degrees(p2) - estimated_angle  ## bias (positive means attraction)
    #     final_bias = [bias_b1, bias_b2]  
    #     skip_r_sq=False
    #     success=True

    # else:
    #     print('Error simultaion')
    #     final_bias=[999, 999]
    #     plot_fit=False
    #     skip_r_sq=True
    #     r_squared=0
    #     success=False ## to eliminate wrong simulations easily at the end

    # #error_fit (r_squared)
    # if skip_r_sq==False:
    #     residuals = y - ans
    #     ss_res = np.sum(residuals**2)
    #     ss_tot = np.sum((y-numpy.mean(y))**2)
    #     r_squared = 1 - (ss_res / ss_tot)

    # #plot fit
    # if plot_fit==True:
    #     plt.figure()
    #     plt.plot(X, y, 'o', color ='red', label ="data") 
    #     plt.plot(X, ans, '--', color ='blue', label ="fit") 
    #     plt.legend() 
    #     plt.show(block=False) 


    # ### Output
    # total_sep=np.degrees(2*p1)
    # final_bias = np.mean(final_bias)
    # #print(total_sep)