# Locally Weighted Regression
### DATA 441 Project #1 - Sam Joyner

### Linear Regression

Linear regression is a technique used to predict an output value given at least one input. Mathematically, this can be represented as:

<p align="center">
  <b>y</b> = <b>X&beta;</b>+<b>&epsilon;</b>
</p>

Where <b>y</b> is the vector of outputs, with <b>x</b> as the vector of inputs, <b>&beta;</b> as the weights vector, and <b>&epsilon;</b> as the random error for each prediction. An import assumption is that the error term is normally distributed with a mean of zero and standard deviation of one. With this assumption in hand, we can solve for <b>&beta;</b>: 

<p align="center">
  <b>X<sup>T</sup>y</b> = <b>X<sup>T</sup>X&beta;</b>+<b>X<sup>T</sup>&epsilon;</b> = <b>X<sup>T</sup>X&beta;</b>
  <br>
  <b>(X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y</b> = <b>&beta;</b>
</p>

Using this we can also plug <b>&beta;</b> back in to get a explicit formula for <b>y</b>, our predictions. As the name suggests, this technique works great for linear datasets. The following example shows the relationship between car weight and mileage, and the line in blue is a standard linear regression, done with the technqiue described above, that attempts to predict miles per gallon based off weight.

<p align="center">
<img src='WGT_MPG_Linear_Regression-2.png'>
</p>

Linear regression does an decent job, but there is certainly room to improve this model, particularly with lower weights where the trend appears more quadratic rather than linear.

### Adding Weights to the Regression Equation

To fit a better model to this data, we can employ <b>Locally Weighted Linear Regression (LOWESS)</b>, which is similar to the technique above, but allows for curvature by performing linear regression over very small portions of the data that collectively create a curved, better fit model. To do this, we now use a matrix of weights (more on this later) that determines how important values will be to the given prediction. To represent this mathematically, we begin with the same regression equation as before and then multiply by <b>W</b>, which is a matrix containing weights along the diagonal and zeros everywhere else: 

<p align="center">
  <b>Wy</b> = <b>WX&beta;</b>+<b>W&epsilon;</b>
  <br>
  <b>X<sup>T</sup>Wy</b> = <b>X<sup>T</sup>WX&beta;</b>+<b>X<sup>T</sup>&epsilon;</b> = <b>X<sup>T</sup>WX&beta;</b>
  <br>
  <b>(X<sup>T</sup>WX)<sup>-1</sup>X<sup>T</sup>Wy</b> = <b>&beta;</b>
</p>

The obvious question becomes how to determine the proper weights, and this is done through kernels that determine the weights and a hyperparameter that specifies the width of the kernel. All of the points that have a nonzero weight make up a neighborhood, and from each neighborhood a linear regression is made, and all of these collectively make up the nonlinear model. Examples of kernels include the Gaussian, Epanechnikov, and Tricubic kernels, seen below.

<p>
  <div class='row'>
      <img src='Gaussian_Kernel.png' style='width: 33%' align='center'>
      <img src='Epanechnikov_Kernel.png' style='width: 33%' align='center'>
      <img src='Tricubic_Kernel.png' style='width: 33%' align='center'>
  </div>
</p>

What locally weighted linear regression does is iterate over the data and for every point it applies the weights to nearby kernels, according to the choosen kernel and width, and creates the neighborhood that creates a small linear regression that combines with the others to create a nonlinear model.

### Applying Weighted Linear Regression

Having explained the math and concepts behind locally weighted linear regression, we can now begin to develop code to create models using this approach. The code below shows how to run these sets of small linear regressions to make the overall model using the weighted approach.

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
    #   on the weights and update the output
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

We can use this method to employee these three kernels with a specified value of tau. The three plots below show a locally weighted linear regression model for the car weight and mileage data from above for three different kernels. All three have a tau value of 1.



Clearly, this model fits the data better than the standard linear regression model, and by adjusting our hyperparameters we can further test with and adapt the fit of the model to avoid over or underfitting. The ______ kernel appears to work best, so we can begin further testing and validation to tune our model.
