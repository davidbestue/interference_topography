

from scipy.optimize import curve_fit 



#### Gaussian

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)


N=512
starting_point_std = 15
starting_point_A = 5

y=np.reshape(rE, (N)) 
X=np.reshape(np.linspace(1, N, N), N)

param, covs = curve_fit(gauss, X, y, p0 = np.array([N/2, starting_point_std, starting_point_A] ))


ans = param[2]*exp(-(X-param[0])**2/2/param[1]**2)
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
plt.figure()
plt.plot(X, y, 'o', color ='red', label ="data") 
plt.plot(X, ans, '--', color ='blue', label ="fit") 
plt.legend() 
plt.show(block=False) 


theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 

estimated_angle=np.degrees(theta[int(param[0])])
print(estimated_angle)


################ Bimodal

N=512
sep=5
y=np.reshape(rE, (N)) 
X=np.reshape(np.linspace(1, N, N), N)

theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)] 


abs_du = np.abs(np.array(theta)- (pi - pi/sep))
orig_1 = np.where( abs_du == abs_du.min())[0][0]

abs_du = np.abs(np.array(theta)- (pi + pi/sep))
orig_2 = np.where( abs_du == abs_du.min())[0][0]


starting_point_std = 15
starting_point_A = 5

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)



def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

param, covs = curve_fit(bimodal, X, y, p0 = np.array([orig_1, starting_point_std, starting_point_A, orig_2, starting_point_std, starting_point_A] ))

ans = param[2]*exp(-(X-param[0])**2/2/param[1]**2) +   param[5]*exp(-(X-param[3])**2/2/param[4]**2) 
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
plt.figure()
plt.plot(X, y, 'o', color ='red', label ="data") 
plt.plot(X, ans, '--', color ='blue', label ="fit") 
plt.legend() 
plt.show(block=False) 


estimated_angle_1 = np.degrees(theta[int(param[0])] )
estimated_angle_2 = np.degrees(theta[int(param[3])] )
print(estimated_angle_1, estimated_angle_2)




########################### von misses

def von_misses(x,mu,k):
    return (exp( k * cos(x-mu))) / (2*pi*scipy.special.i0(k)) 

N=512
y=np.reshape(rE, (N)) 
X=np.reshape(np.linspace(-pi, pi, N), N)


param, covs = curve_fit(von_misses, X, y)

ans = (exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1])) 

  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
plt.figure()
plt.plot(X, y, 'o', color ='red', label ="data") 
plt.plot(X, ans, '--', color ='blue', label ="fit") 
plt.legend() 
plt.show(block=False) 

estimated_angle=np.degrees(param[0]+pi)  

print(estimated_angle)





########################### von misses double


def von_misses(x,mu,k):
    return (exp( k * cos(x-mu))) / (2*pi*scipy.special.i0(k)) 


def bi_von_misses(x,mu1,k1,mu2,k2):
    return von_misses(x,mu1,k1) + von_misses(x,mu2,k2)


N=512
y=np.reshape(rE, (N)) 
X=np.reshape(np.linspace(-pi, pi, N), N)

param, covs = curve_fit(bi_von_misses, X, y)

ans = (exp( param[1] * cos(X-param[0]))) / (2*pi*scipy.special.i0(param[1])) + (exp( param[3] * cos(X-param[2]))) / (2*pi*scipy.special.i0(param[3])) 
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
plt.figure()
plt.plot(X, y, 'o', color ='red', label ="data") 
plt.plot(X, ans, '--', color ='blue', label ="fit") 
plt.legend() 
plt.show(block=False) 


estimated_angle_1=np.degrees(param[0]+pi)  
estimated_angle_2=np.degrees(param[2]+pi)  

print(estimated_angle_1, estimated_angle_2 )





