
from model import *
from joblib import Parallel, delayed
import multiprocessing

##### 2 bumps simultaneous
numcores = multiprocessing.cpu_count() 

distances_test =  list(np.linspace(1.5, 35, 150))  #range(2,35)   

kappa_e_test = [ 300, 225] 
kappa_i_test = [ 30, 15]      

rep_dist = 100

n_kappas= len(kappa_e_test) # len of kappas 
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
###df.to_excel('/home/david/Desktop/nice_all.xlsx')
#df.to_excel('simulations_2bumps_ke_ki2.xlsx')

df_x = df.loc[df['success']==True] 

plt.figure(figsize=(8,6))
g = sns.lineplot( x="separation", y="bias", hue='kappas_E', ci=95 , palette='tab10', data=df_x, legend=False) 
plt.plot([0, max(df_x['separation'])], [0,0], 'k--') ## plot the 0 line (separate attraction from repulsion)
plt.title('Bias with separation', fontsize=15) #condition title
plt.gca().spines['right'].set_visible(False) #no right axis
plt.gca().spines['top'].set_visible(False) #no  top axis
plt.gca().get_xaxis().tick_bottom() # remove the bottom ticks
plt.gca().get_yaxis().tick_left() # remove the left ticks
plt.ylabel('Attraction bias (deg)')
plt.legend(title='kappaE', loc='upper right', labels=[str(i) for i in [225, 300]] ) # add the legend
plt.show(block=False)


###################### 1 bump
# rep_dist = 250
# n_kappas= len(kappa_e_test)

# kappas_e=[]
# kappas_i=[]

# for idx, k in enumerate(kappa_e_test):
#     kappas_e = kappas_e + [k]*rep_dist
#     kappas_i = kappas_i + [kappa_i_test[idx]]*rep_dist


# results2 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=2000, targ_onset=100,  presentation_period=350, separation=0, tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5,
#  GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=1, sigI=1.6, kappa_E=kape, kappa_I=kapi, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=False , plot_fit=False)  for kape, kapi in zip( kappas_e, kappas_i)) 

# biases = [results2[i][0] for i in range(len(results2))]
# separationts = [results2[i][1] for i in range(len(results2))]   
# kappas__e = [results2[i][2] for i in range(len(results2))]      
# kappas__i = [results2[i][3] for i in range(len(results2))]                                                         
# succs = [results2[i][6] for i in range(len(results2))]   
# num_bumps = [results2[i][-1] for i in range(len(results2))]  


# df1=pd.DataFrame({'bias':biases, 'kappas_E':kappas__e, 'kappas_I':kappas__i, 'success':succs, 'n_bumps':num_bumps })


# ####df1.to_excel('/home/david/Desktop/nice_1.xlsx')
# df1_corr = df1.loc[df1['success']==True] 
# df1_corr = df1_corr.loc[df1_corr['n_bumps']==1] 
# df3 = df1_corr.reset_index()

# #df1 = df1.loc[(df1['kappas_E']==200) | (df1['kappas_E']==300) ] 
# plt.figure(figsize=(8,6))
# linares_plot( x="kappas_E", y="bias", order=kappa_e_test,  palette='viridis', alpha=0.4, point_size=5, df=df3) 
# #sns.boxplot(x='kappas_E', y="bias",  df=df3)
# plt.title('Drift with eccentricity separation', fontsize=15) #condition title
# plt.gca().spines['right'].set_visible(False) #no right axis
# plt.gca().spines['top'].set_visible(False) #no  top axis
# plt.gca().get_xaxis().tick_bottom()
# plt.gca().get_yaxis().tick_left()
# #plt.ylim(0, 20)
# #plt.legend(title='kappaE', loc='upper right', labels=['100', '200'])
# plt.show(block=False)


# import statsmodels.formula.api as smf

# res_m = smf.ols(formula='bias ~ kappas_I', data=df3).fit()
# print(res_m.summary())

###
