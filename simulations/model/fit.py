

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



y=np.reshape(rate, (512)) 
X=np.reshape(np.arange(0, 512), (512,1))

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


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

score=[]
for deg_fir in range(1,10):    
    poly_reg = PolynomialFeatures(degree=deg_fir)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    score.append( pol_reg.score(X_poly, y) )
    viz_polymonial()



plt.figure()
plt.plot(np.arange(1,10), score)
plt.show(block=False)



