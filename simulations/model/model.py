

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
    peaks = scipy.signal.find_peaks(line_pred)[0]

    angles_final = [theta[peaks[x]] for x in range(len(peaks))]
    #return angles_final

    if n_stims ==2:
        if len(peaks)==2:
            pb1, pb2 = scipy.signal.find_peaks(line_pred)[0]
            theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
            ang_pb1=theta[pb1]
            bias_b1 = ang_pb1 - (pi-pi/separation) ## bias (positive means attraction)
            ang_pb2=theta[pb2]
            bias_b2 = (pi+pi/separation) - ang_pb2 ## bias (positive means attraction)
            angles_final = [bias_b1, bias_b2]
        elif len(peaks)==1:   
            pb = scipy.signal.find_peaks(line_pred)[0]
            theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
            ang_pb1=theta[pb]
            bias_b1 = ang_pb1 - (pi-pi/separation) ## bias (positive means attraction)
            ang_pb2=theta[pb]
            bias_b2 = (pi+pi/separation) - ang_pb2 ## bias (positive means attraction)
            angles_final = [bias_b1, bias_b2]    
    
    # theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
    # ang_pb1=theta[pb1]
    # bias_b1 = ang_pb1 - (pi-pi/separation) ## bias (positive means attraction)
    
    # ang_pb2=theta[pb2]
    # bias_b2 = (pi+pi/separation) - ang_pb2 ## bias (positive means attraction)ç

    ### Output
    return(rE, peaks) #bias_b1, bias_b2)


###




####

rE, angles = model(totalTime=2000, targ_onset=100,  presentation_period=100, separation=0.1) 
angles





# N=512
# separation=2
# y=np.reshape(rate, (N)) 
# X=np.reshape(np.arange(0, N), (N,1))



# # Visualizing the Polymonial Regression results
# def viz_polymonial():
#     plt.figure()
#     plt.scatter(X, y, color='red')
#     plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
#     plt.title('Fit Bump')
#     plt.xlabel('Neuron')
#     plt.ylabel('rate')
#     plt.show(block=False)
#     return



# ### Fit
# poly_reg = PolynomialFeatures(degree=6)
# X_poly = poly_reg.fit_transform(X)
# pol_reg = LinearRegression()
# pol_reg.fit(X_poly, y)
# #score = pol_reg.score(X_poly, y) 
# viz_polymonial()


# line_pred = pol_reg.predict(poly_reg.fit_transform(X)) 
# pb1, pb2 = scipy.signal.find_peaks(line_pred)[0]


# theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 
# ang_pb1=theta[pb1]
# bias b1 = ang_pb1 - (pi-pi/separation) ## bias (positive means attraction)

# ang_pb2=theta[pb2]
# bias_b2 = (pi+pi/separation) - ang_pb2 ## bias (positive means attraction)







#         ### Plot of activity
# #         # %matplotlib inline
# #         # %config InlineBackend.figure_format = 'png'

# #         plt.ion()

# #         #RE_sorted=flipud(RE)
# #         RE_sorted=RE
# #         rE = RE[:,-1]
# #         M=(rE>2)*1

# #         plt.figure(figsize=(9,6))
# #         sns.heatmap(RE_sorted, cmap='viridis', vmin=-2.5, vmax=10)
# #         plt.title('BUMP activity')
# #         plt.ylabel('Angle (deg)')
# #         plt.xlabel('time')
# #         plt.yticks([])
# #         plt.xticks([])

# #         plt.yticks([N/8, 3*N/8, 5*N/8, 7*N/8 ] ,['45','135','225', '315'])

# #         plt.plot([stim_onset/2, stim_onset/2,], [0+20, N-20], 'k-', label='onset')
# #         plt.plot([stim_offset/2, stim_offset/2,], [0+20, N-20], 'k--', label='offset')


# #         # Neu = spread_start(rE)[2]
# #         # spr = spread_start(rE)[1]

# #         # plt.plot([0, 1000], [ st1_i  ,st1_i ], 'r--', label='st1')
# #         # plt.plot([0, 1000], [ st2_i , st2_i ], 'b--', label='st2')

# #         # plt.plot([0, 1000], [ Neu[1]  , Neu[1] ], 'b--', label='offset')
# #         # plt.plot([0, 1000], [ Neu[1] + spr[1]  , Neu[1] + spr[1] ], 'b--', label='offset')


# #         # plt.plot([0, 1000], [ 129  , 129], 'g--', label='bump')
# #         # plt.plot([0, 1000], [ 391.5  , 391.5], 'g--', label='bump')


# #         plt.legend()
# #         plt.show(block=False)

# #         mx_fr= round(max(rE),2)
# #         #Firing_rates.append(mx_fr)

# #         #plot Firing rate RE
# #         plt.figure()
# #         h = sns.tsplot(rE, color='r')
# #         plt.title('Firing rate', weight='demibold')
# #         plt.gca().spines['right'].set_visible(False)
# #         plt.gca().spines['top'].set_visible(False)
# #         plt.gca().get_xaxis().tick_bottom()
# #         plt.gca().get_yaxis().tick_left()
# #         plt.ylabel('rE')
# #         plt.xticks([N/8, 3*N/8, 5*N/8, 7*N/8 ] ,['45','135','225', '315'])
# #         plt.xlabel('angle')
# #         plt.show(block=False)

# #         print( 'Max firing rate = ' + str(round(max(rE),2)))


# #         # BIAS due to noise

# #         if n_stims==1:
# #             Decoded_ang = round(decode(RE)[-1][2],2)

# #             #Set the stimulus center
# #             Decoded_stim = round(decode(stimulus)[-1][2],2)

# #             # BIAS
# #             bias = abs(Decoded_ang - Decoded_stim)

# #             print(bias)

# #         #Biases.append(bias)
#         # BUMP movement
#         if n_stims==2: 
#         #     CENTERS = []

#         #     for r in range(0, len(RE)):
#         #         width, width_p, i_pos, ang_pos, i_cent, ang_cent = spread_start(RE[:,r])
#         #         CENTERS.append(ang_cent)


#         #     df_c = pd.DataFrame(CENTERS)
#         #     centers_ap = [df_c.iloc[:,i].mean() for i in range(0, shape(df_c)[1])]
#         #     zeros_m = zeros((len(CENTERS), len(centers_ap)))

#         #     for i in range(0, len(CENTERS)):
#         #         values = CENTERS[i]
#         #         if len(values) == 0:
#         #             zeros_m[i,:] = ['NaN' for n in range(0, len(centers_ap))]
#         #         elif len(values) == len(centers_ap):
#         #             zeros_m[i,:] = values
#         #         elif len(values) == len(centers_ap) -1:
#         #             default = ['NaN' for n in range(0, len(centers_ap))]
#         #             a = min(centers_ap, key=lambda x:abs(x-values[0]))
#         #             pos_bump = where(array(centers_ap)==a)[0][0]
#         #             default[pos_bump] = values[0]
#         #             zeros_m[i,:] = default


#         #     zeros_m
#         #     df = pd.DataFrame(zeros_m)
#         #     #print(mean(df.iloc[:, 0] - st1)) #pos means attraction
#         #     #print(mean(st2 - df.iloc[:, 1])) #pos means atraction

#             # BIAS in the las rE
#             width, width_p, i_pos, ang_pos, i_cent, ang_cent = spread_start(RE[:,-1])

#             if len(width) ==0:
#                 Interference = 'NaN'
#             elif len(width) ==1:
#                 dec_b1 = round(decode(RE)[-1][2],2)
#                 bias_b1 = dec_b1 - degrees(pi - stim_sep) #pos att 
#                 Interference = bias_b1
#             elif len(width) ==2:
#                 m_i = int(mean(i_cent))
#                 rE = RE[:,-1]
#                 b1 = rE[0:m_i]
#                 b2 = rE[m_i::]
#                 angle_mid = neur2angle(m_i)
#                 dec_b1 = decode_rE_lim(b1, a_ini=0, a_fin=angle_mid)
#                 dec_b2 = angle_mid + decode_rE_lim(b2, a_ini=0, a_fin=360-angle_mid)
#                 bias_b1 = dec_b1 - degrees(pi - stim_sep) #pos att 
#                 bias_b2 = degrees(pi + stim_sep) - dec_b2 #pos att
#                 Interference =  bias_b1

#             elif len(width)>2:
#                 Interference = 'NaN'
            
            
            
            
#             if Interference != 'NaN':
#                 Interference=round(Interference, 2)
            
#             #print(Interference)

#             Matrix.append([Interference, GEE, VALUE])
#             print(Interference, GEE, VALUE)