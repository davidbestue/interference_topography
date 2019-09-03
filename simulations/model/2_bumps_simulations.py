
#from model import *

# from joblib import Parallel, delayed
# import multiprocessing

# numcores = multiprocessing.cpu_count() - 1

# distances_test = [5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24]
# kappa_e_test = [100, 200] 
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