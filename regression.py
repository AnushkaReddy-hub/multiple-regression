"""
In this assignment we create a Python module
to perform some basic data science tasks. While the
instructions contain some mathematics, the main focus is on 
implementing the corresponding algorithms and finding 
a good decomposition into subproblems and functions 
that solve these subproblems. 

To help you to visually check and understand your
implementation, a module for plotting data and linear
prediction functions is provided.

The main idea of linear regression is to use data to
infer a prediction function that 'explains' a target variable 
of interest through linear effects of one 
or more explanatory variables. 

Part I - Univariate Regression

Task A: Optimal Slope

-> example: price of an apartment

Let's start out simple by writing a function that finds
an "optimal" slope (a) of a linear prediction function 
y = ax, i.e., a line through the origin. A central concept
to solve this problem is the residual vector defined as

(y[1]-a*x[1], ..., y[1]-a*x[1]),

i.e., the m-component vector that contains for each data point
the difference of the target variable and the corresponding
predicted value.

With some math (that is outside the scope of this unit) we can show
that for the slope that minimises the sum of squared the residual

x[1]*(y[1]-a*x[1]) + ... + x[m]*(y[m]-a*x[m]) = 0

Equivalently, this means that

a = (x[1]*y[1]+ ... + x[m]*y[m])/(x[1]*y[1]+ ... + x[m]*y[m])

Write a function slope(x, y) that, given as input
two lists of numbers (x and y) of equal length, computes
as output the lest squares slope (a).

Task B: Optimal Slope and Intercept

To get a better fit, we have to consider the intercept b as well, 
i.e., consider the model f(x) = ax +b. 
To find the slope of that new linear model, we centre the explanatory variable
by subtracting the mean from each data point. 
The correct slope of the linear regression f(x)=ax + b is the same 
slope as the linear model without intercept, f(x)=ax, calculated on the 
centred explanatory variables instead of the original ones. 
If we have calculated the correct slope a, we can calculate the intercept as
b = mean(y) - a*mean(x).

Write a function line(x,y) that, given as input
two lists of numbers (x and y) of equal length, computes
as output the lest squares slope a and intercept b and
returns them as a tuple a,b.


Task C: Choosing the Best Single Predictor

We are now able to determine a regression model that represents 
the linear relationship between a target variable and a single explanatory variable.
However, in usual settings like the one given in the introduction, 
we observe not one but many explanatory variables (e.g., in the example `GDP', `Schooling', etc.). 
As an abstract description of such a setting we consider n variables 
such that for each j with 0 < j < n we have measured m observations 

$x[1][j], ... , x[m][j]$. 

These conceptually correspond to the columns of a given data table. 
The individual rows of our data table then become n-dimensional 
data points represented not a single number but a vector.

A general, i.e., multi-dimensional, linear predictor is then given by an n-dimensional 
weight vector a and an intercept b that together describe the target variable as

y = dot(a, x) + b 

i.e., we generalise y = ax + b by turning the slope a into an n-component linear weight vector
and replace simple multiplication by the dot product (the intercept b is still a single number).
Part 2 of the assignment will be about finding such general linear predictors. 
In this task, however, we will start out simply by finding the best univariate predictor 
and then represent it using a multivariate weight-vector $a$. %smooth out with the text that follows.

Thus, we need to answer two questions: (i) how do we find the best univariate predictor, 
and (ii) how to we represent it as a multivariate weight-vector. 

Let us start with finding the best univariate predictor. For that, we test all possible
predictors and use the one with the lowest sum of squared residuals.
Assume we have found the slope a^j and intercept b^j of the best univariate predictor---and assume it 
uses the explanatory variable x^j---then we want to represent this as a multivariate 
slope a and intercept b. That is, we need to find a multivariate slop a such that dot(a, x) + b 
is equivalent to a^jx^j + b^j. Hint: The intercept remains the same, i.e., $b = b^j$.

Task D: Regression Analysis

You have now developed the tools to carry out a regression analysis. 
In this task, you will perform a regression analysis on the life-expectancy 
dataset an excerpt of which was used as an example in the overview. 
The dataset provided in the file /data/life_expectancy.csv.


Part 2 - Multivariate Regression

In part 1 we have developed a method to find a univariate linear regression model 
(i.e., one that models the relationship between a single explanatory variable and the target variable), 
as well as a method that picks the best univariate regression model when multiple 
explanatory variables are available. In this part, we develop a multivariate regression method 
that models the joint linear relationship between all explanatory variables and the target variable. 


Task A: Greedy Residual Fitting

We start using a greedy approach to multivariate regression. Assume a dataset with m data points 
x[1], ... , x[m] 
where each data point x[i] has n explanatory variables x[i][1], ... , x[i][m], 
and corresponding target variables y[1], ... ,y[m]. The goal is to find the slopes for 
all explanatory variables that help predicting the target variable. The strategy we 
use greedily picks the best predictor and adds its slope to the list of used predictors. 
When all slopes are computed, it finds the best intercept. 
For that, recall that a greedy algorithm iteratively extends a partial solution by a 
small augmentation that optimises some selection criterion. In our setting, those augmentation 
options are the inclusion of a currently unused explanatory variable (i.e., one that currently 
still has a zero coefficient). As selection criterion, it makes sense to look at how much a 
previously unused explanatory variable can improve the data fit of the current predictor. 
For that, it should be useful to look at the current residual vector r,
because it specifies the part of the target variable that is still not well explained. 
Note that a the slope of a predictor that predicts this residual well is a good option for 
augmenting the current solution. Also, recall that an augmentation is used only if it 
improves the selection criterion. In this case, a reasonable selection criterion is 
again the sum of squared residuals.

What is left to do is compute the intercept for the multivariate predictor. 
This can be done  as


b = ((y[1]-dot(a, x[1])) + ... + (y[m]-dot(a, x[m]))) / m

The resulting multivariate predictor can then be written as 

y = dot(a,x) + b .



Task B: Optimal Least Squares Regression

Recall that the central idea for finding the slope of the optimal univariate regression line (with intercept) 
that the residual vector has to be orthogonal to the values of the centred explanatory variable. 
For multivariate regression we have many variables, and it is not surprising that for an optimal 
linear predictor dot(a, x) + b, it holds that the residual vector is orthogonal to each of the 
centred explanatory variables (otherwise we could change the predictor vector a bit to increase the fit). 
That is, instead of a single linear equation, we now end up with n equations, one for each data column.
For the weight vector a that satisfies these equations for all i=1, ... ,n, you can again simply find the 
matching intercept b as the mean residual when using just the weights a for fitting:

b = ((y[1] - dot(a, x[1])) + ... + (y[m] - dot(a, x[m])))/m .

In summary, we know that we can simply transform the problem of finding the least squares predictor to solving a system
of linear equation, which we can solve by Gaussian Elimination as covered in the lecture. An illustration of such a
least squares predictor is given in Figure~\ref{fig:ex3dPlotWithGreedyAndLSR}.
"""

from math import inf, sqrt
import numpy as np
import pandas as pd


def slope(x, y):
    """
    Computes the slope of the least squares regression line
    (without intercept) for explaining y through x.

    For example:
    >>> slope([0, 1, 2], [0, 2, 4])
    2.0
    >>> slope([0, 2, 4], [0, 1, 2])
    0.5
    >>> slope([0, 1, 2], [1, 1, 2])
    1.0
    >>> slope([0, 1, 2], [1, 1.2, 2])
    1.04
    """
    """

A function slope(x, y) that computes the optimal slope for the simple regression model
Input: Two lists of numbers (x and y) of equal length representing data of an explanatory and a target variable.
Output: The optimal least squares slope (a) with respect to the given data.
For example:
>>> slope([0, 1, 2], [0, 2, 4])
2.0
>>> slope([0, 2, 4], [0, 1, 2])
0.5
>>> slope([0, 1, 2], [1, 1, 2])
1.0
>>> slope([0, 1, 2], [1, 1.2, 2])
1.04
This is a list processing problem where the dot product has to be calculated for two lists of numbers (x and y) of equal
length representing data of an explanatory and a target variable. This is the formula to be used here (x · x) a = x · y,
which we need to reassemble for calculating the value of a, which is our optimal slope.

In my implementation, I want to be clear about what I am doing. So, I have made another function for calculating the dot
product and called it inside the main slope function. Also, I have done that so that I can call it in
my next tasks as well In my dot calculation function, I have tried to shorten my code by using the in operator 
membership function which means it is running for every element in (x,y) and for pairing it to the respective element in
the list, I have used the zip() function. For iteration I have used for loop inside the return statement. I have done 
such a arrangement to make the calculation in one go. I have stored the answer in 2 variables of (x · x) and x · y in c 
and b respective. As I reassembled to calculate a from the equation was a = (x · y)/(x · x).
"""
    b = dot(x, y)
    c = dot(x, x)
    a = b/c
    return a


def dot(m, n):
    return sum(m_i * n_i for m_i, n_i in zip(m, n))


def line(x, y):
    """
    Computes the least squares regression line (slope and intercept)
    for explaining y through x.

    For example:
    >>> a, b = line([0, 1, 2], [1, 1, 2])
    >>> round(a,1)
    0.5
    >>> round(b,2)
    0.83
    """
    """
The function is calculating the optimal least squares slope and the optimal intercept with respect to
the given data.

Input: Two lists of numbers (x and y) of equal length representing data of an explanatory and a target variable.
Output: A tuple (a,b) where a is the optimal least squares slope and b the optimal intercept with respect to the given
data.

For example
>>> line([0, 1, 2], [1, 1, 2])
(0.5, 0.8333333333333333)

This problem moves around two lists, using which we have to do some calculations to find the optimal least squares slope
and the optimal intercept with respect to the given data. First we need to find the centred data vector using this
formula on the first list provided by the user. x bar = (x1 − μ, x2 − μ ,. . .,xm − μ) where
μ = (x1 + x2 + · · · + xm)/m being the mean value of the explanatory data, which we also need to calculate. Then we need
to reformulate this expression ((x_ · x_) a = x_ · y) to find a. Also we need to find the optimal intercept (say b)
using this formula b=((y1 −ax1)+···+(ym −axm))/m. For calculating b, we need to first multiply all the elements of list
x,which is the first list provided by the user to the slope which we just calculated. Then we need to subtract this term
from the y list, which is the second list provided by the user and then we have to add all of those terms of the
resulting list and divide that number by the length of x and y.

In my implementation, I chose to make it look as simple and do it as direct as possible in order to be safe from huge
complexities. I have first calculated the average using the in-built sum function and len function to calculate the
length. I have divided it in 1 line to get the average. Then I have initialized a empty list, similar to x bar mentioned
in the task question. For storing the subtracted element in the list, I have used x_ to store by using append function
and running it for every element in the first list given by the user by using a for loop in the range of length of x.
Before it I have also, copied the whole string of x to copy_list_of_x as to append the element, I will have to reassign
the subtracted value, because of which my whole x is changing and I will not be able to use it for later calculations.
Then I have called the dot function from my task A to dot x_ and x_ and then to dot x_ and y.I have stored it in z and y
respectively so that I can divide both of them to calculate a, which is the slope.
For calculating b, I have made a new list called multiplied_list in which I am storing the optimal slope, a,multiplied
by every element in x in just one line that is using for loop for every element in x. Then for subtracting this
mulitiplied_list from y, I have zipped them together because I wanted to subtract the similar index of number to similar 
one and therefore created a zip instance for the storage of the list.
"""
    u = sum(x) / len(x)
    x_ = []
    copy_list_of_x = x.copy()
    difference = []
    for i in range(len(x)):
        x[i] = x[i] - u
        x_.append(x[i])
    z = dot(x_, x_)
    w = dot(x_, y)
    a = w/z
    multiplied_list = [element * a for element in copy_list_of_x]
    zip_object = zip(y, multiplied_list)
    for y, multiplied_list in zip_object:
        difference.append(y - multiplied_list)
    b = sum(difference)/len(x)
    for_returning = (a, b)
    return for_returning


def best_single_predictor(data, y):

    """
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, b = best_single_predictor(data, y)
    >>> weights[0]
    0.0
    >>> round(weights[1],2)
    -0.29
    >>> round(b,2)
    2.14
    """
    """
     A function best single predictor(data, y) that computes a general linear predictor (i.e., one that is defined for multi
     -dimensional data points) by picking the best uni variate prediction model from all available explanatory variables.

    Input: Table data with m > 0 rows and n > 0 columns representing data of n explanatory variables and a list y of length
    m containing the values of the target variable corresponding to the rows in the table.
    Output: A pair (a, b) of a list a of length n and a number b representing a linear predictor with weights a and
    intercept b with the following properties:
    a) There is only one non-zero element of a, i.e., there is one i in range(n) such that a[j]==0 for all indices j!=i.
    b) The predictor represented by (a, b) has the smallest possible squared error among all predictors that satisfy
    property a).

    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> weights, b = best_single_predictor(data, y)
    >>> weights, b
    ([0.0, -0.2857142857142858], 2.1428571428571432)

    The main challenge of this problem was to understand the math implied. As a coder, it is very important because no
    matter how complex the implementation is, if a coder doesn't understand the objectve or the motive, there is no 
    point of implementing it. I have tried my best to make this problem simple and break it into parts. We need to start
    by calculating a and b for all the rows. We had to calculate r1 using this following formula, r1 = y1 − ax1 - b,
    r2= y2 - a1x2-b1..so on till m (manipulation of formula in TASK A), calculating r2 similarly just with different a 
    and b on till rm. Then calculate ex1 = y1 − ax1 - b for every row and find the minimum out of all of the r, 
    calculad above, which will be used to select the best single predictor. Here y1 - ax1 is our actual b and the b in 
    the formula is our calculated Then applying the conditions given in TASK C for the desired output. 
    
    In my implementation, I chose to make it look as simple and do it as direct as possible in order to be safe from 
    huge complexities. To solve this problem, I separated the the lists which was given as combined, for further 
    calculations. I have saved the element 0 and element 1 of every row in separate lists. After that I have used my 
    defined function of line, written in TASK B to calculate a and b for every row. I've stored the result in a list. 
    To get the exact desired output. I have stored the dot product results in a list and the selected single predictor 
    is enclosed with list in a tuple. I have used a global symbol to store all information related to the global scope 
    of the program. I found the %s token easy to use as the it is replaced by whatever I pass after the % symbol. 
    It makes me able to insert and format in one statement. "The %s placeholder and the globals() function is inspired 
    by www.codeacademy.com".
    """
    list1 = []
    lst = []
    for element in range(len(data[0])):
        globals()['x_%s' % element] = [globals()['data_x%s' % element][element] for globals()['data_x%s' % element]
                                           in data]
        for i in range(len(data[0])):
            globals()['var_%s' % i] = line(globals()['x_%s' % i], y)
            lst.append(globals()['var_%s' % i])
    for j in range(len(data[0])):
        list1.append(globals()['var_%s' % j][0])
    min_ = list1.index(min(list1))
    for k in range(len(data[0])):
        if k != min_:
            list1[k] = 0.0
    return (list1, lst[min_][1])

"""
In regression analysis we had to calculate the life expectancy according to our model, for Rwanda? and life expectancy 
according to our model, for Liberia, if Liberia improved its “schooling” to the level of Austria, then what would 
happen? We had to clean up the whole file, convert it into a table format. 
In my implementation, I have appended all rows after converting it into a list to a table named multi-dimensional list.
I was able to write around this much code only, which you can find below. 
table = []
file = open('life_expectancy.csv','r')
for line in file:
    line=line.strip() #to remove the /n character
    line=line.split(',') #to make it a list
    table.append(line)
table = table[1:]
for row in table:
    
    

"""


def greedy_predictor(data, y, alpha=0.1, iterations=50):
    """
    This implements a greedy correlation pursuit algorithm.

    Input: A numpy data table data of explanatory variables with m rows and n columns and a list of corresponding target
    variables y.
    Output: A tuple (a,b) where a is the weight vector and b the intercept obtained by the greedy strategy.

    Since I have imported numpy for this question. I have changed the input from normal array to a
    numpy array.

    For example:
    >>> data=np.array([[1,0],[2,3],[4,2]])
    >>> y=[2,1,2]
    >>> theta=np.random.randn(data.shape[1])
    >>> weights, intercept = greedy_predictor(data,y)
    >>> weights, b
    [ 0.32210066 -0.50022212] 1.6580770413211332

    The main challenge of this problem was to understand the math implied. As a coder, it is very important because no
    matter how complex the implementation is, if a coder doesn't understand the objectve or the motive, there is no
    point of implementing it. I have tried my best to make this problem simple and break it into parts as 3 extra
    functions: normalise, computeCost and hypothesis. Also, I have changed the arguments passed,added some default
    arguments to the greedy_predictor function. Imported and used NumPy as there was scientific computing and multi-
    dimensional array involved in the question. To minimise the error, I have used standard deviation too.

    Starting by explaining the normalise function, here I have tried to compute the arithmetic mean along the column,
    stored it in mu variable and then compute the standard deviation along the column, stored in sigma variable.
    Standard deviation is a measure of the spread of a distribution, of the array elements. The mu and sigma variable is
    storing the mean and standard deviation of all the array elements respectively. Then data-mu is giving the error and
    is divided by standard deviation because I wanted to convert the answer into 0 and 1 range. The next function is
    hypothesis which is created to return the transposed matrix which is done using the .T of numpy library which is
    same as self.transpose(), except that self is returned if self.ndim < 2. The next function is computeCost which is
    created to return the computed mathematical equation . m variable stores the length of the array.  Basically
    in the next line the cost function maps values of one or more variables onto a real number intuitively representing 
    some “cost” associated with the event. Then in the greedy_predictor function, I have calculated the weight vector
    and b, the intercept using the greedy strategy.

    Since it's numpy in which almost all calculations are O(n). If it involves each element of an array, speed will
    depend on the size of the array. Array manipulations for reshaping is O(1) because they don't actually do anything
    with the data; they change properties like shape.
    """
    n = len(data)
    data = normalise(data)
    data = np.append(np.ones((len(data), 1)), values=data, axis=1)
    theta = np.zeros((1, len(data[0])))
    J_hist = []
    y = np.array(y).reshape(3, 1)
    for i in range(iterations):
        slope = hypothesis(data, theta) - y
        theta -= (alpha / n) * slope.T @ data
        cost = computeCost(data, y, theta)
        J_hist.append(cost)
    return theta[0][1:], theta[0][0]

def normalise(data):
    mu = np.mean(np.array(data), axis=0)
    sigma = np.std(data, axis=0)
    data = (data - mu) / sigma
    return data

def hypothesis(data, theta):
    return data @ theta.T

def computeCost(data, y, theta):
    m = len(data)
    J = (1 / (2 * m)) * np.sum((hypothesis(data, theta) - y) ** 2)
    return J

def equation(i, data, y):
    """
    Finds the row representation of the i-th least squares condition,
    i.e., the equation representing the orthogonality of
    the residual on data column i:

    (x_i)'(y-Xb) = 0
    (x_i)'Xb = (x_i)'y

    x_(1,i)*[x_11, x_12,..., x_1n] + ... + x_(m,i)*[x_m1, x_m2,..., x_mn] = <x_i , y>

    Input: Integer i with 0 <= i < n, data matrix data with m rows and n columns such that m > 0 and n > 0, and list of
    target values y of length n.
    Output: Pair (c, d) where c is a list of coefficients of length n and d is a float representing the coefficients and
    right-hand-side of Equation 8 for data column i.

    For example:
    >>> data = [[1, 0],
    ...         [2, 3],
    ...         [4, 2]]
    >>> y = [2, 1, 2]
    >>> coeffs, rhs = equation(0, data, y)
    >>> coeffs, rhs
    ([4.666666666666666, 2.3333333333333335], 0.3333333333333326)

    The main challenge of this program was again, to understand what's going on. Here, I have used lot of list
    comprehensions, almost in every line to keep the function short and simple.

    Firstly I have calculated the mean, stored it in the mean variable. Then, I have calculated the current column.
    Then I have generated a for loop for finding the coefficients, the result of which I have stored in coeff variable.
    Then, I have found the rhs of the equation.

    The complexity of the function equation is O(n^2) as there is the use of 2 nested for loops to the max. The other
    time complexity of every statement is mostly O(n) because of 1 for loop used.
    """
    mean = (sum([x[i] for x in data]) / len([x[i] for x in data]))
    x_i = [x[i] - mean for x in data]
    coeff = []
    for j in range(len(data[0])):
        mean = (sum([x[j] for x in data]) / len([x[j] for x in data]))
        x_j = (([x[j] - mean for x in data]))
        coeff.append(sum([x_i[k] * x_j[k] for k in range(len(y))]))
    rhs = sum([x_i[k] * y[k] for k in range(len(y))])
    return coeff, rhs

def least_squares_predictor(data, y):
    """
    This function least_squares_predictor(data, y) finds the optimal least squares predictor for the given data matrix
    and target vector.

    For example:
    >>> data = [[0, 0],
    ...         [1, 0],
    ...         [0, -1]]
    >>> y = [1, 0, 0]
    >>> weights, intercept = least_squares_predictor(data, y)
    ([-1.0, 1.0], 0.3333333333333333)

    Input: Data matrix data with m rows and n columns such that m > 0 and n > 0.
    Output: Optimal predictor (a, b) with weight vector a (len(a)==n) and intercept b such that a, b minimise the sum of
    squared residuals.

    The main challenge of this program was again, to understand what's going on. Here, I have used lot of list
    comprehensions, almost in every line to keep the function short and simple. Also, I have used back substitution to
    perform gaussian elemination.

    I am storing in n the no. of unknowns. a variable is defined for storing in the matrix. Inside of x, I've initiazed
    unknown weights as x. The first for loop is for building the matrix for gaussian elimination. Once thatt's done we
    perform gaussian elemination in the next for loop, which is nested.Then, I've written some lines using the Back
    Substitution technique. Followed by intialiszig a variable to calculate the Intercept. Then I have calculate the
    intercept. I haven't used any of my previous functions because it was different from the official answers. I have
    merger those functions in this functions.

    The time complexity of this algorithm is O(n^3) because I have used 3 nested loops to perform the gaussian
    elemination. There, the k-loop has O(j-i) complexity, the j-loop has O((n-i)*(n-i)) complexity and the i-loop has
    O(n*n*n)=O(n^3) complexity.
    """
    n = len(data[0])
    a = []
    x = [0 for x in range(n)]
    for i in range(len(data[0])):
        coeff, rhs = equation(i, data, y)
        coeff.append(rhs)
        a.append(coeff)
    for i in range(n):
        for j in range(i + 1, n):
            ratio = a[j][i] / a[i][i]
            for k in range(n + 1):
                a[j][k] = a[j][k] - ratio * a[i][k]
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = a[i][n]
        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]
        x[i] = x[i] / a[i][i]
    a_x = []
    for i in range(len(data)):
        sum = 0
        for j in range(len(data[0])):
            sum = sum + (x[j] * data[i][j])
            a_x.append(sum)
            b = 0
        for i in range(len(y)):
            b += y[i] - a_x[i]
            b = b / len(y)
            return x, b


def regression_analysis():        
    """
    The regression analysis can be performed in this function or in any other form you see
    fit. The results of the analysis can be provided in this documentation. If you choose
    to perform the analysis within this funciton, the function could be implemented 
    in the following way.
    
    The function reads a data provided in "life_expectancy.csv" and finds the 
    best single predictor on this dataset.
    It than computes the predicted life expectancy of Rwanda using this predictor, 
    and the life expectancy of Liberia, if Liberia would improve its schooling 
    to the level of Austria.
    The function returns these two predicted life expectancies.
    
    For example:
    >>> predRwanda, predLiberia = regression_analysis()
    >>> round(predRwanda)
    65
    >>> round(predLiberia)
    79
    """
    pass
    
if __name__=='__main__':
    import doctest
    doctest.testmod()
