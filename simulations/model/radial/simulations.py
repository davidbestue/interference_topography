# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

from model_radial2 import *

numcores = multiprocessing.cpu_count() - 4


############################################# delay 2 seconds ##############################################################################################################

# path_save= '/home/david/Desktop/IDIBAPS/Simulations_radial/results_simul_.xlsx'

# Positions = list(np.arange(60,310,10))*1000  

# outputs= Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, 
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
#            plot_rate=False, plot_hm=False, plot_fit=False) for posx in Positions) 


# df = pd.DataFrame(outputs)
# df.columns=['interference', 'position']

# #############

# df.to_excel(path_save)



############################################# delay 2 seconds ##############################################################################################################

path_save= '/home/david/Desktop/IDIBAPS/Simulations_radial/results_simul_d0.xlsx'

Positions = list(np.arange(60,310,10))*1000  

outputs= Parallel(n_jobs = numcores)(delayed(model)(totalTime=600, 
           targ_onset=100,  
           presentation_period=350,
           positions=posx, 
           tauE=9, tauI=4,  n_stims=1, 
           I0E=0.1, I0I=0.5,
           GEE=0.025, GEI=0.02, GIE=0.01 , GII=0.1, 
           sigE=0.8, sigI=1.6, 
           #sigE=0., sigI=0., 
           kappa_E=100, kappa_I=20, 
           kappa_stim=100, N=512,
           plot_connectivity=False, 
           plot_rate=False, plot_hm=False, 
           plot_fit=False, save_RE=False) for posx in Positions) 


df = pd.DataFrame(outputs)
df.columns=['interference', 'position']

#############

df.to_excel(path_save)
