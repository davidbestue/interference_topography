# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

from model_radial_linear import *

numcores = multiprocessing.cpu_count() - 3


############################################# delay 2 seconds ##############################################################################################################

###paths_save_= '/home/david/Desktop/IDIBAPS/Simulations_radial/results_simul_radial_linear_all.xlsx'

frames=[]

for idx, TIMES in enumerate([3450]): # enumerate(list(np.arange(0,4000, 1000) + 450 ) ): ##4000
	print(TIMES)
	Positions = [5]*8 ##list(np.arange(1.5,5.25,0.25))*500 ##0.25
	Times=[TIMES for i in range(len(Positions))]
	outputs= Parallel(n_jobs = numcores)(delayed(model_radial_linear)(totalTime=tim, 
	           targ_onset=100,  
	           presentation_period=350,
	           position=posx, 
	           tauE=9, tauI=4,  
	           I0E=0.1, I0I=0.5,
	           GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
	           NsigE=0.8, NsigI=1.7, 
	           N=512, rint = 1, rext = 6,
	           plot_connectivity=False, 
	           plot_rate=False, save_RE=False) for posx, tim in zip(Positions, Times)) 
	#
	df = pd.DataFrame(outputs)
	df.columns=['interference', 'position', 'simul_time']
	df['delay_time']=TIMES-450
	frames.append(df)
	############

##
df_tot2 = pd.concat(frames)
###df_tot.to_excel(paths_save_)
