
from model import *
from joblib import Parallel, delayed
import multiprocessing


##### 2 bumps
numcores = multiprocessing.cpu_count() 

distances_test =  range(2,35)   

# kappa_e_test = [ 200, 250, 201, 300, 100] 
# kappa_i_test = [ 9, 30, 30, 30, 10] 


kappa_e_test = [ 300, 225] #[300, 300, 300, 250, 250, 250, 200, 200, 200, 150, 150, 150]
kappa_i_test = [ 30, 15]       #[30, 20, 10, 30, 20, 10, 30, 20, 10, 30, 20, 10]

rep_dist = 25

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

df_300 = df.loc[df['kappas_E']==300]
df_201 = df.loc[df['kappas_E']==201]
df_250 = df.loc[df['kappas_E']==250]
df_test = df_250 # pd.concat([df_300, df_201, df_250])

plt.figure(figsize=(8,6))
g = sns.lineplot( x="separation", y="bias", hue='kappas_E', ci=95 , palette='tab10', data=df_test) 
plt.plot([0, max(df['separation'])], [0,0], 'k--') 
plt.title('Bias with separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
g.legend()
#lt.legend(title='kappaE', loc='upper right', labels=[str(i) for i in [201, 300]] )
#plt.xlim(0,70)
plt.show(block=False)


###################### 1 bump


rep_dist = 250
n_kappas= len(kappa_e_test)

kappas_e=[]
kappas_i=[]

for idx, k in enumerate(kappa_e_test):
    kappas_e = kappas_e + [k]*rep_dist
    kappas_i = kappas_i + [kappa_i_test[idx]]*rep_dist


results2 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5,
 GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=1, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for kape, kapi in zip( kappas_e, kappas_i)) 

biases = [results2[i][0] for i in range(len(results2))]
separationts = [results2[i][1] for i in range(len(results2))]   
kappas__e = [results2[i][2] for i in range(len(results2))]      
kappas__i = [results2[i][3] for i in range(len(results2))]                                                         
succs = [results2[i][6] for i in range(len(results2))]   
num_bumps = [results2[i][-1] for i in range(len(results2))]  


df1=pd.DataFrame({'bias':biases, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs, 'n_bumps':num_bumps })

df1_corr = df1.loc[df1['success']==True] 
df1_corr = df1_corr.loc[df1_corr['n_bumps']==1] 
df3 = df1_corr.reset_index()

#df1 = df1.loc[(df1['kappas_E']==200) | (df1['kappas_E']==300) ] 
plt.figure(figsize=(8,6))
linares_plot( x="kappas_E", y="bias", order=kappa_e_test,  palette='tab10', alpha=0.4, point_size=5, df=df3) 
sns.boxplot(x='kappas_E', y="bias",  df=df3)
plt.title('Drift with eccentricity separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
#plt.ylim(0, 20)
#plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
plt.show(block=False)


import statsmodels.formula.api as smf

res_m = smf.ols(formula='bias ~ kappas_I', data=df1_corr).fit()
print(res_m.summary())


