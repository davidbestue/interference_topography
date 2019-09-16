
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


from model import *
from joblib import Parallel, delayed
import multiprocessing



r1 = model(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1,
      I0I=0.5,  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=1, sigI=1.6, kappa_E=225, kappa_I=15, kappa_stim=75,
      N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=False)



for n in range(0,10):
    r2 = model(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1,
          I0I=0.5,  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=1, sigI=1.6, kappa_E=300, kappa_I=30, kappa_stim=75,
          N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=False)




rE = r[-4]
### Fit
df_n_p=pd.DataFrame()
df_n_p['rE'] = rE.reshape(512)
peaks_list=[]
for n_w_s in range(1, 100):
    r = df_n_p['rE'].rolling(window=n_w_s).mean()
    number_of_bumps = len(scipy.signal.find_peaks(r, 2)[0]) 
    peaks_list.append(number_of_bumps)

number_of_bumps=most_frequent(peaks_list)
if number_of_bumps == 0:
    if peaks_list==[0 for i in range(len(peaks_list))]:
        number_of_bumps = 0
    else:
        peaks_list[:] = (value for value in peaks_list if value != 0)
        number_of_bumps=most_frequent(peaks_list)



peaks_list[:] = (value for value in peaks_list if value != 0)


###### pruebas

##### 2 bumps
numcores = multiprocessing.cpu_count() 

distances_test =  [2,3,4,5, 7, 9, 11, 13, 15, 19, 25, 30, 35]    #[5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24]
# kappa_e_test = [200, 150, 100, 300, 250] 
# kappa_i_test = [20, 25, 10, 30, 15] 

# kappa_e_test = [ 100, 300, 250, 200] 
# kappa_i_test = [ 10, 30, 15, 20] 

kappa_e_test = [ 300, 225] #[300, 300, 300, 250, 250, 250, 200, 200, 200, 150, 150, 150]
kappa_i_test = [ 30, 15]       #[30, 20, 10, 30, 20, 10, 30, 20, 10, 30, 20, 10]
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
 GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=1.0, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for sep, kape, kapi in zip(separations, kappas_e, kappas_i)) 

biases = [results[i][0] for i in range(len(results))]
separationts = [results[i][1] for i in range(len(results))]   
kappas__e = [results[i][2] for i in range(len(results))]      
kappas__i = [results[i][3] for i in range(len(results))]                                                         
succs = [results[i][6] for i in range(len(results))]   

df=pd.DataFrame({'bias':biases, 'separation':separationts, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs })
#df.to_excel('simulations_2bumps_ke_ki2.xlsx')

df = df.loc[df['success']==True] 
#df_x = df.loc[df['kappas_E']!=250] 

plt.figure(figsize=(8,6))
g = sns.lineplot( x="separation", y="bias", hue='kappas_E', ci=95 , hue_order=kappa_e_test, palette='tab10', 
                 data=df, legend=False) 
plt.plot([0, max(df['separation'])], [0,0], 'k--') 
plt.title('Bias with separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
#plt.legend(title='kappaE', loc='upper right', labels=[str(i) for i in kappa_e_test] )
#plt.xlim(0,70)
plt.show(block=False)


# kappa_e_test = [ 200, 201, 300] 
# kappa_i_test = [ 8, 30, 30] 


rep_dist = 300
n_kappas= len(kappa_e_test)

kappas_e=[]
kappas_i=[]

for idx, k in enumerate(kappa_e_test):
    kappas_e = kappas_e + [k]*rep_dist
    kappas_i = kappas_i + [kappa_i_test[idx]]*rep_dist


results = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5,
 GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=1, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for kape, kapi in zip( kappas_e, kappas_i)) 

biases = [results[i][0] for i in range(len(results))]
separationts = [results[i][1] for i in range(len(results))]   
kappas__e = [results[i][2] for i in range(len(results))]      
kappas__i = [results[i][3] for i in range(len(results))]                                                         
succs = [results[i][6] for i in range(len(results))]   
num_bumps = [results[i][-1] for i in range(len(results))]  


df1=pd.DataFrame({'bias':biases, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs, 'n_bumps':num_bumps })
df1_corr = df1.loc[df1['success']==True] 
df1_corr = df1_corr.loc[df1_corr['n_bumps']==1] 


#df1 = df1.loc[(df1['kappas_E']==200) | (df1['kappas_E']==300) ] 
plt.figure(figsize=(8,6))
linares_plot( x="kappas_E", y="bias", order=kappa_e_test,  palette='viridis', alpha=0.4, point_size=5, df=df1_corr) 
plt.title('Drift with eccentricity separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.ylim(0, 20)
#plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
plt.show(block=False)


import statsmodels.formula.api as smf

res_m = smf.ols(formula='bias ~ kappas_I', data=df1_corr).fit()
print(res_m.summary())


# rE=r[-4]
# df_n_p=pd.DataFrame()
# df_n_p['rE'] = rE.reshape(512)
# peaks_list=[]
# for n_w_s in range(1, 100):
#     r = df_n_p['rE'].rolling(window=n_w_s).mean()
#     number_of_bumps = len(scipy.signal.find_peaks(r, 2)[0]) 
#     peaks_list.append(number_of_bumps)

# number_of_bumps=most_common(peaks_list)

# number_of_bumps