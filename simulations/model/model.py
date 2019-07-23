

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


def decode_rE(rE, a_ini=0, a_fin=360, N=512):
    #Population vector for a given rE
    # return ( angle in radians, absolut angle in radians, abs angle in degrees )
    N=len(rE)
    Angles = np.linspace(a_ini, a_fin, N) 
    angles=np.radians(Angles)
    rE = np.reshape(rE, (1,N))
    R = numpy.sum(np.dot(rE,exp(1j*angles)))/numpy.sum(rE)
    
    angle_decoded = np.degrees(np.angle(R))
    if angle_decoded<0:
        angle_decoded = 360+angle_decoded
    
    return angle_decoded
    #Mat.append(  [angle(R), abs(angle(R)) , degrees(abs(angle(R)))]  )
    #return round( np.degrees(abs(np.angle(R))), 2)


def circ_dist(a1,a2):
    ## Returns the minimal distance in angles between to angles 
    op1=abs(a2-a1)
    angs=[a1,a2]
    op2=min(angs)+(360-max(angs))
    options=[op1,op2]
    return min(options)




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


def model(totalTime, targ_onset, presentation_period, separation, tauE=9, tauI=4,  n_stims=2, I0E=0.1, I0I=0.5, GEE=0.022, GEI=0.019, 
 GIE=0.01 , GII=0.1, sigE=1.5, sigI=1.6, kappa_E=100, kappa_I=1.75, kappa_stim=100, N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=True):
    #
    st_sim =time.time()
    dt=2
    nsteps=int(floor(totalTime/dt));
    origin = pi
    rE=zeros((N,1));
    rI=zeros((N,1));
    #Connectivities
    v_E=zeros((N));
    v_I=zeros((N));
    WE=zeros((N,N));
    WI=zeros((N,N));
    separation= pi/separation
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
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
        plt.plot(WE[250, :])
        plt.plot(WI[250, :])
        plt.show(block=False)
    ##
    # Stims
    if n_stims==2:
        stimulus1=zeros((N))
        stimulus2=zeros((N))
        for i in range(0, N):
            stimulus1[i]=2*e**(kappa_stim*cos(theta[i] + origin - separation)) / (2*pi*scipy.special.i0(kappa_stim))
            stimulus2[i]=2*e**(kappa_stim*cos(theta[i] + origin + separation)) / (2*pi*scipy.special.i0(kappa_stim))
        stimulus= (stimulus1 + stimulus2);
        stimulus=reshape(stimulus, (N,1))
    elif n_stims==1:
        stimulus2=zeros((N));
        for i in range(0, N):
            stimulus2[i]=2*e**(kappa_stim*cos(theta[i] + origin)) / (2*pi*scipy.special.i0(kappa_stim))
        stimulus=stimulus2
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
    ## metrics
    if n_stims==2:
        interference = Interference_effects( [decode_rE(stimulus1)], [decode_rE(rE)], [decode_rE(stimulus2)])[0]

    p_targ1 = int((N * np.degrees(origin + separation))/360)
    p_targ2 = int((N * np.degrees(origin - separation))/360)
    #
    if plot_rate==True:
        #### plot dynamics
        fig = plt.figure()
        plt.title('Rate dynamics')
        plt.plot(RE[p_targ1, :], 'b', label='target1')
        plt.plot(RE[p_targ2, :], 'r', label='target2')
        plt.xlabel('time (ms)')
        plt.ylabel('rate (Hz)')
        plt.legend()
        plt.show(block=False)
    if plot_hm==True:
        #### plot heatmap
        RE_sorted=flipud(RE)
        plt.figure(figsize=(9,6))
        sns.heatmap(RE_sorted, cmap='viridis')
        plt.title('BUMP activity')
        plt.ylabel('Angle')
        plt.xlabel('time')
        plt.plot([stimon, nsteps], [p_targ2, p_targ2], '--b',) ## flipped, so it is p_target 
        plt.plot([stimon, nsteps], [p_targ1, p_targ1], '--r',) ## flipped, so it is p_target 
        plt.yticks([])
        plt.xticks([])
        plt.yticks([N/8, 3*N/8, 5*N/8, 7*N/8 ] ,['45','135','225', '315'])
        plt.plot([stimon, stimon,], [0+20, N-20], 'k-', label='onset')
        plt.plot([stimoff, stimoff,], [0+20, N-20], 'k--', label='offset')
        plt.plot([stimon, stimon,], [0+20, N-20], 'k-')
        plt.plot([stimoff, stimoff,], [0+20, N-20], 'k--')
        plt.legend()
        plt.show(block=False)
    
    
    ## print time consumed
    end_sim =time.time()
    total_time= end_sim - st_sim 
    total_time = round(total_time, 1)
    print('Simulation time: ' + str(total_time) + 's')

    ###### Final bias
    y=np.reshape(rE, (N)) 
    X=np.reshape(np.arange(0, N), (N,1))
    # Visualizing the Polymonial Regression results
    ### Fit
    poly_reg = PolynomialFeatures(degree=6) ## 6 is the optimal for both
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    #score = pol_reg.score(X_poly, y) 
    if plot_fit==True:
        viz_polymonial(X, y, poly_reg, pol_reg)

    #peaks bump
    line_pred = pol_reg.predict(poly_reg.fit_transform(X)) 
    peaks = scipy.signal.find_peaks(line_pred, height=2)[0]

    #return final_bias
    if n_stims ==2:
        if len(peaks)==2:
            pb1, pb2 = peaks
            bias_b1 = np.degrees(theta[pb1]) - np.degrees(pi - separation) 
            bias_b2 =  np.degrees(pi + separation) - np.degrees(theta[pb2])
            final_bias = [bias_b1, bias_b2]
        elif len(peaks)==1: 
            pb = peaks[0]
            bias_b1 = np.degrees(theta[pb]) - np.degrees( pi - separation)
            bias_b2 = np.degrees(pi + separation) - np.degrees(theta[pb])  ## bias (positive means attraction)
            final_bias = [bias_b1, bias_b2]   

    ### Output
    total_sep=np.degrees(2*separation)
    return(bias_b1, bias_b2, total_sep, GEE) #bias_b1, bias_b2)


###
####

## Example
#bias, sep = model(totalTime=2000, targ_onset=100,  presentation_period=100, separation=16) 
#print(bias, sep)

from joblib import Parallel, delayed
import multiprocessing

numcores = multiprocessing.cpu_count() - 3
separations=list(np.linspace(2.1,15,25)) * 4
gees=[0.021, 0.023]

results_gee1 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=100, plot_hm=False , plot_fit=False, separation=seps, GEE=gees[0])  for seps in separations) 
results_gee2 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=100, plot_hm=False , plot_fit=False, separation=seps, GEE=gees[1])  for seps in separations) 

df_1 = pd.DataFrame(results_gee1)
df_1.columns=['b1', 'b2', 'distance', 'GEE']
df_2 = pd.DataFrame(results_gee2)
df_2.columns=['b1', 'b2', 'distance', 'GEE']
df = pd.concat([df_1, dfs])

sns.lineplot( x="distance", y="b1", hue='GEE',  ci=95 , data=df) 
plt.plot([0, max(df['distance'])], [0,0], 'k--') 
plt.title('Bias with distance') #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.show(block=False)


##fit
###### Final bias
y=np.reshape(df['b1'].values, (len(df))) 
X=np.reshape(df['distance'].values, (len(df),1))
# Visualizing the Polymonial Regression results
### Fit
poly_reg = PolynomialFeatures(degree=4) ## 6 is the optimal for both
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)
viz_polymonial(X, y, poly_reg, pol_reg)



# score=[]
# min_ = 1
# max_ = 10
# for deg_fir in range(min_,max_):    
#     poly_reg = PolynomialFeatures(degree=deg_fir)
#     X_poly = poly_reg.fit_transform(X)
#     pol_reg = LinearRegression()
#     pol_reg.fit(X_poly, y)
#     score.append( pol_reg.score(X_poly, y) )
#     viz_polymonial(X, y, poly_reg, pol_reg)




# plt.figure()
# plt.plot(np.arange(min_,max_), score)
# plt.show(block=False)


