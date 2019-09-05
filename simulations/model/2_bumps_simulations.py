
from model import *
from joblib import Parallel, delayed
import multiprocessing


# ##### 2 bumps
# numcores = multiprocessing.cpu_count() -2

# distances_test = [5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24]
# kappa_e_test = [100, 200] 
# kappa_i_test = [7, 20] 
# rep_dist = 10
# n_kappas= len(kappa_e_test)
# n_sepa = len(distances_test)

# separations= distances_test * rep_dist * n_kappas

# kappas_e=[]
# kappas_i=[]

# for idx, k in enumerate(kappa_e_test):
#     kappas_e = kappas_e + [k]*n_sepa*rep_dist
#     kappas_i = kappas_i + [kappa_i_test[idx]]*n_sepa*rep_dist


# results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=sep, tauE=9, tauI=4,  n_stims=2, I0E=0.1, I0I=0.5,
#  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.9, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for sep, kape, kapi in zip(separations, kappas_e, kappas_i)) 

# biases = [results[i][0] for i in range(len(results))]
# separationts = [results[i][1] for i in range(len(results))]   
# kappas__e = [results[i][2] for i in range(len(results))]      
# kappas__i = [results[i][3] for i in range(len(results))]                                                         
# succs = [results[i][6] for i in range(len(results))]   

# df=pd.DataFrame({'bias':biases, 'separation':separationts, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs })
# #df.to_excel('simulations_2bumps_ke_ki2.xlsx')

# df = df.loc[df['success']==True] 

# plt.figure(figsize=(8,6))
# g = sns.lineplot( x="separation", y="bias", hue='kappas_E', ci=95 , hue_order=[100,200], palette='viridis', 
#                  data=df, legend=False) 
# plt.plot([0, max(df['separation'])], [0,0], 'k--') 
# plt.title('Bias with separation', fontsize=15) #condition title
# plt.gca().spines['right'].set_visible(False) #no right axis
# plt.gca().spines['top'].set_visible(False) #no  top axis
# plt.gca().get_xaxis().tick_bottom()
# plt.gca().get_yaxis().tick_left()
# plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
# plt.show(block=False)



###################### 1 bump

# numcores = multiprocessing.cpu_count() - 1

# kappa_e_test = [100, 200] 
# kappa_i_test = [7,  20] 
# rep_dist = 200
# n_kappas= len(kappa_e_test)

# kappas_e=[]
# kappas_i=[]

# for idx, k in enumerate(kappa_e_test):
#     kappas_e = kappas_e + [k]*rep_dist
#     kappas_i = kappas_i + [kappa_i_test[idx]]*rep_dist


# results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5,
#  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.9, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for kape, kapi in zip( kappas_e, kappas_i)) 

# biases = [results[i][0] for i in range(len(results))]
# separationts = [results[i][1] for i in range(len(results))]   
# kappas__e = [results[i][2] for i in range(len(results))]      
# kappas__i = [results[i][3] for i in range(len(results))]                                                         
# succs = [results[i][6] for i in range(len(results))]   


# df=pd.DataFrame({'bias':biases, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs })
# df.to_excel('single_item_drift_eccentricity_ke_ki.xlsx')


# df = df.loc[df['success']==True] 
# plt.figure(figsize=(8,6))
# linares_plot( x="kappas_E", y="bias", order=[100, 200],  palette='viridis', alpha=0.4, point_size=5, df=df) 
# plt.title('Drift with eccentricity separation', fontsize=15) #condition title
# plt.gca().spines['right'].set_visible(False) #no right axis
# plt.gca().spines['top'].set_visible(False) #no  top axis
# plt.gca().get_xaxis().tick_bottom()
# plt.gca().get_yaxis().tick_left()
# plt.ylim(0, 20)
# #plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
# plt.show(block=False)


# #### MODEL
# import statsmodels.formula.api as smf

# res_m = smf.ols(formula='bias ~ kappas_E', data=df).fit()
# print(res_m.summary())


r = model(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1,
      I0I=0.5,  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.9, sigI=1.6, kappa_E=200, kappa_I=20, kappa_stim=75,
      N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=False)


r = model(totalTime=2000, targ_onset=100,  presentation_period=350, separation=7, tauE=9, tauI=4,  n_stims=2, I0E=0.1,
      I0I=0.5,  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.9, sigI=1.6, kappa_E=150, kappa_I=6, kappa_stim=75,
      N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=False)




df_n_p=pd.DataFrame()
df_n_p['rE'] = rE.reshape(512)
peaks_list=[]
for n_w_s in range(1, 20):
    r = df_n_p['rE'].rolling(window=n_w_s).mean()
    number_of_bumps = len(scipy.signal.find_peaks(r, 2)[0]) 
    peaks_list.append(number_of_bumps)

number_of_bumps=min(peaks_list)
number_of_bumps

# ###### pruebas


##### 2 bumps
numcores = multiprocessing.cpu_count() -2

distances_test =  [2,3,4,5, 7, 9, 11, 13, 15, 19, 25]    #[5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24]
kappa_e_test = [200, 150] 
kappa_i_test = [20, 25] 
rep_dist = 20
n_kappas= len(kappa_e_test)
n_sepa = len(distances_test)

separations= distances_test * rep_dist * n_kappas

kappas_e=[]
kappas_i=[]

for idx, k in enumerate(kappa_e_test):
    kappas_e = kappas_e + [k]*n_sepa*rep_dist
    kappas_i = kappas_i + [kappa_i_test[idx]]*n_sepa*rep_dist


results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=sep, tauE=9, tauI=4,  n_stims=2, I0E=0.1, I0I=0.5,
 GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.9, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for sep, kape, kapi in zip(separations, kappas_e, kappas_i)) 

biases = [results[i][0] for i in range(len(results))]
separationts = [results[i][1] for i in range(len(results))]   
kappas__e = [results[i][2] for i in range(len(results))]      
kappas__i = [results[i][3] for i in range(len(results))]                                                         
succs = [results[i][6] for i in range(len(results))]   

df=pd.DataFrame({'bias':biases, 'separation':separationts, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs })
#df.to_excel('simulations_2bumps_ke_ki2.xlsx')

df = df.loc[df['success']==True] 

plt.figure(figsize=(8,6))
g = sns.lineplot( x="separation", y="bias", hue='kappas_E', ci=95 , hue_order=kappa_e_test, palette='viridis', 
                 data=df, legend=False) 
plt.plot([0, max(df['separation'])], [0,0], 'k--') 
plt.title('Bias with separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.legend(title='kappaE', loc='upper right', labels=[str(i) for i in kappa_e_test] )
plt.xlim(0,70)
plt.show(block=False)


rep_dist = 50
n_kappas= len(kappa_e_test)

kappas_e=[]
kappas_i=[]

for idx, k in enumerate(kappa_e_test):
    kappas_e = kappas_e + [k]*rep_dist
    kappas_i = kappas_i + [kappa_i_test[idx]]*rep_dist


results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5,
 GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.9, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for kape, kapi in zip( kappas_e, kappas_i)) 

biases = [results[i][0] for i in range(len(results))]
separationts = [results[i][1] for i in range(len(results))]   
kappas__e = [results[i][2] for i in range(len(results))]      
kappas__i = [results[i][3] for i in range(len(results))]                                                         
succs = [results[i][6] for i in range(len(results))]   


df=pd.DataFrame({'bias':biases, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs })

df = df.loc[df['success']==True] 
plt.figure(figsize=(8,6))
linares_plot( x="kappas_E", y="bias", order=kappa_e_test,  palette='viridis', alpha=0.4, point_size=5, df=df) 
plt.title('Drift with eccentricity separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.ylim(0, 20)
#plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
plt.show(block=False)


import statsmodels.formula.api as smf

res_m = smf.ols(formula='bias ~ kappas_E', data=df).fit()
print(res_m.summary())

