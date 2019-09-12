
from model import *
from joblib import Parallel, delayed
import multiprocessing


##### 2 bumps
numcores = multiprocessing.cpu_count() 

distances_test =  [2,3,4,5, 7, 9, 11, 13, 15, 19, 25, 30, 35]    

kappa_e_test = [ 200, 250, 201, 300, 100] 
kappa_i_test = [ 9, 30, 30, 30, 10] 
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
plt.legend(title='kappaE', loc='upper right', labels=[str(i) for i in kappa_e_test] )
#plt.xlim(0,70)
plt.show(block=False)

