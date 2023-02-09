# Locally Weighted Regression
### DATA 441 Project #1 - Sam Joyner

### Linear Regression

Linear regression is a technique used to predict an output value given at least one input. Mathematically, this can be represented as:

<p align="center">
  <b>y</b> = <b>X&beta;</b>+<b>&epsilon;</b>
</p>

Where <b>y</b> is the vector of outputs, <b>x</b> is the vector of inputs, <b>&beta;</b> is the vector of coefficients, and <b>&epsilon;</b> is the random error for each prediction. An import assumption is that the error term is normally distributed with a mean of zero and standard deviation of one. With this assumption, we can then solve for <b>&beta;</b>: 

<p align="center">
  <b>X<sup>T</sup>y</b> = <b>X<sup>T</sup>X&beta;</b>+<b>X<sup>T</sup>&epsilon;</b> = <b>X<sup>T</sup>X&beta;</b>
  <br>
  <b>(X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y</b> = <b>&beta;</b>
</p>

Using this we can also plug <b>&beta;</b> back in to get a explicit formula for <b>y</b>, our predictions. As the name suggests, this technique works great for linear datasets. 

The following example shows the relationship between car weight and mileage, and the line in blue is a standard linear regression, done with the technqiue described above, that attempts to predict miles per gallon based off weight.

<p align="center">
<img src='WGT_MPG_Linear_Regression-2.png'>
</p>

We can see that linear regression does an decent job, but there is certainly room to improve this model as a straight line does not appear to capture the curvature in the data, particularly with lower car weights where the trend appears more quadratic than linear.

### Adding Weights to the Regression Equation

To fit a better model to this data, we can employ <b>Locally Weighted Linear Regression (LOWESS)</b>, which is an extension of linear regression but allows for curvature in the model by performing linear regression over very small portions of the data to collectively create a nonlinear model across the entire dataset. To do this, we use a matrix of weights (more on this later) that determines how important values will be to the given prediction. Nearby values will have a relatively large weight, while those further away will have little to no impact. To represent this mathematically, we begin with the same regression equation as before and then multiply by <b>W</b>, which is a matrix containing weights along the diagonal and zeros everywhere else: 

<p align="center">
  <b>Wy</b> = <b>WX&beta;</b>+<b>W&epsilon;</b>
  <br>
  <b>X<sup>T</sup>Wy</b> = <b>X<sup>T</sup>WX&beta;</b>+<b>X<sup>T</sup>&epsilon;</b> = <b>X<sup>T</sup>WX&beta;</b>
  <br>
  <b>(X<sup>T</sup>WX)<sup>-1</sup>X<sup>T</sup>Wy</b> = <b>&beta;</b>
</p>

The obvious question becomes how to determine the ideal weights, and this is done through kernels that determine the weights and a hyperparameter, tau, that specifies the width of the kernel. These kernels are based off of mathematical models, and specify how nearby values should be weighted for a prediction. Examples of kernels include the Gaussian, Epanechnikov, and Tricubic kernels, seen below.

<p>
  <div class='row'>
      <img src='Gaussian_Kernel.png' style='width: 33%' align='center'>
      <img src='Epanechnikov_Kernel.png' style='width: 33%' align='center'>
      <img src='Tricubic_Kernel.png' style='width: 33%' align='center'>
  </div>
</p>

Locally weighted linear regression iterates over the data and for every point it applies the weights determined by the kernel to nearby datapoints, creating neighborhoods for which a small linear regression model will be made, and these small linear models are combined to make a larger, nonlinear model.

### Applying Weighted Linear Regression

Having explained the math and concepts behind locally weighted linear regression, we can now begin to develop code to create models using this approach. The code below shows how to run these sets of small linear regressions to make the overall model using the locally weighted linear regression approach.

```Python
def locally_weighted_linear_regression(x, y, kern='Gaussian', tau=0.05):
    # Get input length and initialize output
    n = len(x)
    y_out = np.zeros(n)

    # Call the given kernel, return if not one of these three options
    # w is the weights matrix based on each points distance from
    #   the current point x[i]
    if kern == 'Gaussian':
      w = np.array([Gaussian((x-x[i])/(2*tau)) for i in range(n)])   
    elif kern == 'Epanechnikov':
      w = np.array([Epanechnikov((x-x[i])/(2*tau)) for i in range(n)])  
    elif kern == 'Tricubic':
      w = np.array([Tricubic((x-x[i])/(2*tau)) for i in range(n)]) 
    else:
      print('Invalid Kernel.') 
      return

    # Iterate over every values and fit a linear regression model based
    #   on the weighted data values and update the output
    for i in range(n):
        weights = w[:, i]
        lm = LinearRegression()
        lm.fit(np.diag(w[:,i]).dot(x.reshape(-1,1)),np.diag(w[:,i]).dot(y.reshape(-1,1)))
        y_out[i] = lm.predict(x[i].reshape(-1,1)) 

    return y_out

# Gaussian Kernel
def Gaussian(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*d**2))

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)
```

The code below shows the use of the previous function and generation of the plot below, which shows the locally weighted linear regression for the car weight and mileage data from before. The gray dots represent the data, and each line represents a model using the specified kernel.

```Python
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the data in and identify x and y
cars = pd.read_csv('/content/cars.csv')
x = np.array(cars['WGT'])
y = np.array(cars['MPG'])

# Run each model with the specified kernel
gaussian = locally_weighted_linear_regression(x,y,'Gaussian',tau=1)
epanenchnikov = locally_weighted_linear_regression(x,y,'Epanechnikov',tau=1)
tricubic = locally_weighted_linear_regression(x,y,'Tricubic',tau=1)

# Plot the results with matplotlib
plt.figure(figsize=[14,8])
plt.scatter(cars['WGT'],cars['MPG'], color = 'gray', alpha=.5)
plt.plot(x[np.argsort(x)][::-1],tricubic[np.argsort(tricubic)], linewidth = 3, alpha=.75,color='red', label = 'Tricubic Kernel, tau=1')
plt.plot(x[np.argsort(x)][::-1],epanenchnikov[np.argsort(epanenchnikov)], linewidth = 3, alpha=.75,color='green',label='Epanenchnikov Kernel, tau=1')
plt.plot(x[np.argsort(x)][::-1],gaussian[np.argsort(gaussian)], linewidth = 3, alpha=.75, label = 'Gaussian Kernel, tau=1')
plt.ylabel('Miles Per Gallon (mi)')
plt.xlabel('Weight (lbs)')
plt.title('Locally Weighted Linear Regression')
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('WGT_MPG_LOWESS.png', bbox_inches='tight')
plt.show()
```

<p align = 'center'>
  <img src='WGT_MPG_LOWESS.png'>
</p>

Visually, these models fit the data better than the normal linear regression model we began with. By adjusting our hyperparameters we can further adjust the fit of the model. Lets continue with the Gaussian kernel to highlight the importance of tuning the tau parameter. The following code and plot demonstrate how changing tau can impact the quality and fit of the model by adjusting the width of the neighborhoods.

```Python
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the data in and identify x and y
cars = pd.read_csv('/content/cars.csv')
x = np.array(cars['WGT'])
y = np.array(cars['MPG'])

# Run the model with the Gaussian kernel for each tau
gaussian1 = locally_weighted_linear_regression(x,y,'Gaussian',tau=1)
gaussian10 = locally_weighted_linear_regression(x,y,'Gaussian',tau=10)
gaussian100 = locally_weighted_linear_regression(x,y,'Gaussian',tau=100)
gaussian500 = locally_weighted_linear_regression(x,y,'Gaussian',tau=500)

# Plot the results with matplotlib
plt.figure(figsize=[14,8])
plt.scatter(cars['WGT'],cars['MPG'], color = 'gray', alpha=.5)
plt.plot(x[np.argsort(x)][::-1],gaussian500[np.argsort(gaussian500)], linewidth = 3, alpha=.75, color = 'purple', label = 'Gaussian Kernel, tau=500')
plt.plot(x[np.argsort(x)][::-1],gaussian100[np.argsort(gaussian100)], linewidth = 3, alpha=.75, color = 'black', label = 'Gaussian Kernel, tau=100')
plt.plot(x[np.argsort(x)][::-1],gaussian10[np.argsort(gaussian10)], linewidth = 3, alpha=.75, color = 'orange', label = 'Gaussian Kernel, tau=10')
plt.plot(x[np.argsort(x)][::-1],gaussian1[np.argsort(gaussian1)], linewidth = 3, alpha=.75, label = 'Gaussian Kernel, tau=1')
plt.ylabel('Miles Per Gallon (mi)')
plt.xlabel('Weight (lbs)')
plt.title('Locally Weighted Linear Regression')
plt.legend()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('WGT_MPG_LOWESS.png', bbox_inches='tight')
plt.show()
```

<p align = 'center'>
  <img src='WGT_MPG_LOWESS-2.png'>
</p>

Furthermore, we can also use SciKitLearn's mean squared error function to numerically compare the models.

```Python
from sklearn.metrics import mean_squared_error as mse

print(mse(y,gaussian1))
print(mse(y,gaussian10))
print(mse(y,gaussian100))
print(mse(y,gaussian500))
```

This code yields the following outputs: 4.0140561326113255, 13.714786141738632, 17.399647692514126, 57.37351262683244. These results confirm what was clear visually: increasing tau leads to a higher MSE for this dataset. This makes sense, as when tau increases the linear regressions get larger and eventually when tau becomes big enough the model will be the same as the standard linear regression model which had a poor fit for this data.

It is important to note, however, that higher MSE is not necessarily a bad thing. As with any data oriented project your model depends heavily on the context and the data you are using. For example, even though the model with a tau of one had the lowest MSE, it may be overfit to the data and may perform worse than the models with tau equal to 10, 100, or even 500. It is always import to avoid over or underfitting a predictive model, and to protect against this it is important to cross validate your model using a technique like k-fold cross validation.
