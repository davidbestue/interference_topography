# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

from model_radial_linear import *

numcores = multiprocessing.cpu_count() - 3


############################################# delay 2 seconds ##############################################################################################################


paths_save_= '/home/david/Desktop/IDIBAPS/Simulations_radial/results_simul_radial_linear.xlsx'

frames=[]

for idx, TIMES in enumerate(list(np.arange(0,4000, 1000) + 450 ) ):
	Positions = list(np.arange(1.5,5.25,0.25))*500  
	print(TIMES)
	outputs= Parallel(n_jobs = numcores)(delayed(model_radial_linear)(totalTime=TIMES, 
	           targ_onset=100,  
	           presentation_period=350,
	           position=posx, 
	           tauE=9, tauI=4,  
	           I0E=0.1, I0I=0.5,
	           GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
	           NsigE=0.8, NsigI=1.7, 
	           N=512, rint = 1, rext = 6,
	           plot_connectivity=False, 
	           plot_rate=False, save_RE=False) for posx in Positions) 
	#
	df = pd.DataFrame(outputs)
	df.columns=['interference', 'position']
	df['delay_time']=TIMES-450
	frames.append(df)
	############

##
df_tot = pd.concat(frames)
df_tot.to_excel(paths_save_)





# ############################################# delay 2 seconds ##############################################################################################################

# path_save_d= '/home/david/Desktop/IDIBAPS/Simulations_radial/results_simul_d3_1000.xlsx'

# Positions = list(np.arange(60,310,10))*1000  
# print('delay')

# outputs= Parallel(n_jobs = numcores)(delayed(model)(totalTime=3000, 
#            targ_onset=100,  
#            presentation_period=350,
#            positions=posx, 
#            tauE=9, tauI=4,  n_stims=1, 
#            I0E=0.1, I0I=0.5,
#            GEE=0.025, GEI=0.02, GIE=0.01 , GII=0.1, 
#            sigE=0.8, sigI=1.6, 
#            #sigE=0., sigI=0., 
#            kappa_E=100, kappa_I=20, 
#            kappa_stim=100, N=512,
#            plot_connectivity=False, 
#            plot_rate=False, plot_hm=False, 
#            plot_fit=False, save_RE=False) for posx in Positions) 


# dfd = pd.DataFrame(outputs)
# dfd.columns=['interference', 'position']

# #############

# dfd.to_excel(path_save_d)



# ############################################# delay 2 seconds ##############################################################################################################

# path_save_p= '/home/david/Desktop/IDIBAPS/Simulations_radial/results_simul_d0_1000.xlsx'

# Positions = list(np.arange(60,310,10))*1000 #00  

# print('no-delay')

# outputsp= Parallel(n_jobs = numcores)(delayed(model)(totalTime=600, 
#            targ_onset=100,  
#            presentation_period=350,
#            positions=posx, 
#            tauE=9, tauI=4,  n_stims=1, 
#            I0E=0.1, I0I=0.5,
#            GEE=0.025, GEI=0.02, GIE=0.01 , GII=0.1, 
#            sigE=0.8, sigI=1.6, 
#            #sigE=0., sigI=0., 
#            kappa_E=100, kappa_I=20, 
#            kappa_stim=100, N=512,
#            plot_connectivity=False, 
#            plot_rate=False, plot_hm=False, 
#            plot_fit=False, save_RE=False) for posx in Positions) 


# dfp = pd.DataFrame(outputsp)
# dfp.columns=['interference', 'position']

# #############

# dfp.to_excel(path_save_p)
