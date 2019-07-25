

import numpy as np 
  
# curve-fit() function imported from scipy 
from scipy.optimize import curve_fit 
  
from matplotlib import pyplot as plt 
  
# numpy.linspace with the given arguments 
# produce an array of 40 numbers between 0 
# and 10, both inclusive 
x = np.linspace(0, 10, num = 40) 
  
  
# y is another array which stores 3.45 times 
# the sine of (values in x) * 1.334.  
# The random.normal() draws random sample  
# from normal (Gaussian) distribution to make 
# them scatter across the base line 
y = 3.45 * np.sin(1.334 * x) + np.random.normal(size = 40) 
  
# Test function with coefficients as parameters 
def test(x, a, b): 
     return a * np.sin(b * x) 



def test_n(x, mu, sigma, k):     
     return( k *  1/(sigma * np.sqrt(2 * np.pi)) *  np.exp( - (x - mu)**2 / (2 * sigma**2) ))

 
#
# curve_fit() function takes the test-function 
# x-data and y-data as argument and returns  
# the coefficients a and b in param and 
# the estimated covariance of param in param_cov 
x= np.arange(0,512, 1)[:250]
y = rate.reshape(1, 512)[0] [:250]

param, param_cov = curve_fit(test_n, x, y)
  
  
print("Sine funcion coefficients:") 
print(param) 
print("Covariance of coefficients:") 
print(param_cov) 
  
# ans stores the new y-data according to  
# the coefficients given by curve-fit() function 
ans = (param[0]*(np.sin(param[1]*x))) 
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
  
plt.plot(x, y, 'o', color ='red', label ="data") 
plt.plot(x, ans, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.show(block=False) 



y=np.reshape(rE, (512)) 
X=np.reshape(np.arange(0, 512), (512,1))




# Fitting Polynomial Regression to the dataset
def viz_polymonial():
    plt.figure()
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Fit Bump')
    plt.xlabel('Neuron')
    plt.ylabel('rate')
    plt.show(block=False)
    return


score=[]
for deg_fir in range(5,15):    
    poly_reg = PolynomialFeatures(degree=deg_fir)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    score.append( pol_reg.score(X_poly, y) )
    viz_polymonial()



plt.figure()
plt.plot(np.arange(5,15), score)
plt.show(block=False)




from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.signal


y=np.reshape(rE, (N)) 
X=np.reshape(np.arange(0, N), (N,1))



# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.figure()
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Fit Bump')
    plt.xlabel('Neuron')
    plt.ylabel('rate')
    plt.show(block=False)
    return



### Fit
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)
#score = pol_reg.score(X_poly, y) 
viz_polymonial()


line_pred = pol_reg.predict(poly_reg.fit_transform(X)) 


pb1, pb2 = scipy.signal.find_peaks(line_pred)[0]




from pylab import *
from scipy.optimize import curve_fit

# y=np.reshape(rE, (512)) 
# X=np.reshape(np.arange(0, 512), (512,1))


data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))
data=np.reshape(rE, (512)) 
y,x,_=hist(data,100,alpha=.3,label='data')


y=np.reshape(rE, (512)) 
y=scipy.stats.zscore(y)
x=np.reshape(np.arange(0, 512), (512))

#y=np.reshape(rE, (512)) 
X=np.reshape(np.arange(0, 512), (512,1))

#x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


params =curve_fit(bimodal,x,y)
sigma=sqrt(diag(cov))
plot(x,bimodal(x,*params),color='red',lw=3,label='model')
legend()
print(params,'\n',sigma)  


sklearn.preprocessing.normalize(y)


bias, total_sep, GEE, rE = model(totalTime=2000, targ_onset=100,  presentation_period=350, separation=5,tauE=9, tauI=4,  n_stims=1, I0E=0.1, I0I=0.5, GEE=0.025, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.5, sigI=1.6, kappa_E=200, kappa_I=20, kappa_stim=75, N=512, plot_connectivity=False, plot_rate=False, plot_hm=True , plot_fit=True) 




from pylab import *
from scipy.optimize import curve_fit

data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))
y,x,_=hist(data,100,alpha=.3,label='data')

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected=(1,.2,250,2,.2,125)
params,cov=curve_fit(bimodal,x,y,expected)
plt.figure()
sigma=sqrt(diag(cov))
plot(x,bimodal(x,*params),color='red',lw=3,label='model')
legend()
print(params,'\n',sigma)  
plt.show()




def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)



x= np.arange(0,512, 1)
y = rE.reshape(1, 512)[0]

param, covs = curve_fit(gauss, x, y)
#sigma=sqrt(diag(covs))
#plot(x,gauss(x,*param),color='red',lw=3,label='model')
#legend()
#print(param,'\n',sigma)  

  
  
print("Sine funcion coefficients:") 
print(param) 
print("Covariance of coefficients:") 
print(param_cov) 
  
# ans stores the new y-data according to  
# the coefficients given by curve-fit() function 
ans = param[2]*exp(-(x-param[0])**2/2/param[1]**2)
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
  
plt.plot(x, y, 'o', color ='red', label ="data") 
plt.plot(x, ans, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.show(block=False) 






def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)




score=[]
min_ = 1
max_ = 20

y=np.reshape(rE, (512)) 
X=np.reshape(np.arange(0, 512), (512,1))

#y=np.reshape(rE[int(512/4) : int(512*3/4)]  , (int(512/2)))  
#X=np.reshape(np.arange(0, int(512/2) ), (int(512/2) ,1))

for deg_fir in range(min_,max_):    
    poly_reg = PolynomialFeatures(degree=deg_fir)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    score.append( pol_reg.score(X_poly, y) )
    viz_polymonial(X, y, poly_reg, pol_reg)




plt.figure()
plt.plot(np.arange(min_,max_), score)
plt.show(block=False)









from pylab import *
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





