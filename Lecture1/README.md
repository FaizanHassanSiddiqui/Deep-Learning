# Simple Linear Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

This lecture is on simple linear regression (one explanatory variable and one response variable). 

Other types of regression are:
* multiple linear regression
* multivariate linear regression

**Note:** It is misleading to use the term multi-variate, when you simply mean multiple linear regression.

***

## Mathematical Knowledge Required

Mathematical knowledge required to understand linear regression includes very basic knowedge about coordinate geometry. You should know how to form equation of a straight line. Below is the general standard form of equation of a straight line:

**y = mx + c**

where, **'m'** is the **gradient** and **'c'** is the **y-intercept**.

In the notebook, we use this equation as **y = wx + b** to emphasize the context; **'w'** for **weight** and **'b'** for **bias** (b is beccause dependent variable is not fully explained by the chosen independent variable). Weights and Bias are collectively referred to as **parameters**.  These are learnt by the model during training.

Also, you should know a litte linear algebra (atleast matrix algebra), some statistics, and basic optimization theory and all that it takes to learn these subjects if you want to learn how artificial neural networks work in more depth. This is not covered here.  


## To understand code:

### Prerequisites

* You should know basic python, including a little knowledge of numpy and pandas. For data visualization, some knowledge of matplotlib is also expected, although not necessary to understand the overall lecture.

* Tensorflow 2.0 is a must. 

### Code Description

If you want to know linear regression using machine learning, skip to part 2. Though, part 1 helps to see what is going on in a linear regression problem. 

#### part 1
First we read the data from a .csv file. The data has sample of height and weight of ten-thousand people. The relationship between height and weight is quite obvious. They are directly proportional to each other in general; meaning if a person is taller, he is likely to have more weight. We should expect to get a line with positive cortrelation. For our example, we treat height as the independent variable (explanatory variable) and weight as the independent variable (response variable). Note, there is a subtle difference between independent variable and expalanatory variable. But in our case, we can use the terms interchangeably. 

Next, the heights (input) are separated from weights (output) and are converted to arrays. There are thousand data points in total as mentioned before. After we have data, we define two functions; one for calculating the value of weight for a given height and the other for calculating the mean squared error between the actual output (weight given in data) and calculated output (weight calculated using our function). The function is linear, given by the expression **wx + b**. The value of **w** and **b** are chosen randomly to try to luckily get a pair that makes a line that best fits our orginal data. This is not machine learning because we are choosing the parameters (**w** and **b**) on our own. Once we get a suitable pair of 'w' and 'b', we have a trained model. We also take the help of the loss function (MSE) to calculate loss at differnt values of w and b. This loss curve is plotted to better see how loss varied as we tweaked the values of w and b. 

**Note**: No machine learning has happened so far. 

#### part 2
In this part, we do machine learning (ML). In order to do this, we take the help of Deep Learing framework called tensorflow. We could have written our model using only python, but we use this framework to make things easier. All of the heavy lifting is done by tensorflow. All we have to do is connect things. We take layers according to our need, connect them, and we have our network. Then we feed the data to the network and the network tries to learn the pattern. By learning I mean it tries to find the best possible values of **'w'** and **'b'**, so that our data is best modeled. In simple words, linear regression is to get the line of best fit. Previously, we were tweaking the values ourselves and tried to get the best pair. Now the machine (our network) is doing the job for us. Again, once we have the best pair (not necessarily the best as it depends on the hyperparameters of our choosing; infact we will almost never be able to choose the perfect pair), we say that we have trained the network, and then we call it our model. The machine has learnt! Now we can save our model and start inferring. 

So to go again, first we create our network, in this case we are choosing only a single hidden layer with a single hidden unit (because our prolem is linear). That's the architecture of our network. The learning process can be broken down into two broad steps, the **forward pass** (forward-propagation) and the **backward pass** (back-propagation). When we feed the data to the network it first moves forward through the network. It makes a prediction. We compare that prediction to the actual value and calculate the loss (here we are using MSE loss, which is typical for regression problems). Once we have the loss, we try to minimize our error (loss). That's now simply an optimization problem. The error is propagated back through the network to change the values of parameters (weight and bias) in such a way that the parameters that contributed more to the error are changed more and those that contributed less are changed less. Consequently, next time the network makes a better prediction. How to know how much to update the weights? We take the gradient of the loss function with respect to weights. The gradient points in the direction of steepest increase in the loss function. So we make the movement in the direction opposite to the gradient. Since, we do not want to miss the minima, we make sure we only make a small step in the right direction **(gradient descent)** and so we multiply the gradient by a constant called learning rate (usually denoted by alpha or eta). That's how we update the weight and bias. In each iteration (epoch), we make a forward pass and a backward pass. And the network learns in each iteration and finally converges to a solution which is realtively close to the ideal point.    

If there was no tensorflow (or other DL frameworks), we'd have to write all of these steps explicitly. But fortunately, we HAVE tensorflow and all has been abstracted away. That's what tensorflow does for us!



**ps.: if you need any more help to understand better, feel free to contact.**
