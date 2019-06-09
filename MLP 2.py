
# coding: utf-8

# # Exercise Sheet 11 (Bonus)- MLP
# Yiping Deng, Shalom-David Anifowoshe

# In[36]:


# initialize numpy enviroment
get_ipython().magic(u'matplotlib notebook')
import numpy as np
from numpy import square
import matplotlib.pyplot as plt


# In[2]:


# Code for visualizing MLP structure

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


# ## Structure of the Multiple Layer Perceptron
# We use the classic structure of Multiple Layer Perceptron, with
# 
# 1. Input Layer: 2 input neurons and 1 bias unit
# 2. One Hidden Layer: 2 hidden neurons with 1 bias Unit
# 3. Output Layer: 1 output neurons
# 
# ### Bias unit
# Bias unit guarantees that it is a affine combination of previous layer inside of the neuron, increasing the model flexibility of each neurons.
# 
# 
# The following is the visualization of the MLP ( note that bias unit is emitted)

# In[3]:


mlp_fig = plt.figure(figsize=(6, 6))
draw_neural_net(mlp_fig.gca(), .2, .8, .1, .9, [2, 2, 1])


# ## Implementation of the MLP
# Firstly, let's define the necessary activation function $\sigma$ and the quadratic loss function

# In[4]:


def sigmoid(x):
    """Sigmoid Function
    Input: A vector in any dimension
    Output: Applying Sigmoid function elementwise
    """
    exp_part = np.exp(-x)
    return 1 / (1 + exp_part)

def sigmoid_grad(x):
    """Gradient of the sigmoid function
    Input: A vector in any dimension
    Output: Calculate Gradient of Sigmoid function elementwise
    """
    sig = sigmoid(x)
    return np.multiply(sig, 1 - sig)

def loss(y_pred, y):
    diff = y_pred - y
    np.sum(np.square(diff))
    

    
def mean_square_error(y_pred, y):
    m, n = y_pred.shape
    l = loss(y_pred, y)
    return l / m / n


# The Gradient of the Sigmoid function is calculated using the following formula:
# $$ \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) $$

# Given two weight matrix $W_1, W_2$, representing the weight from the input layer to the hidden layer and from the hiddden layer to the output layer, including the bias term, we can conclude that $W_1 \in \mathbb{R}^{2 \times 3}$ and $W_2 \in \mathbb{R}^{1 \times 3}$.
# 
# ## Forward Propagation
# Our Entire Neural Network can be represent as a single function:
# $$ N: \mathbb{R}^2 \to \mathbb{R} $$
# Such function will:
# 1. Assiciate the input $X \in \mathbb{R}^2$ with a bias term, forming $X_{biased}$
# 2. Calculate the offine combination
# $$ z_2 = W_1 \cdot X_{biased} $$, where $z_2 \in \mathbb{R}^2$
# 3. Apply the Sigmoid function, obtaining the value of the neurons
# $$ a_2 = \sigma(z_2) $$, where $a_2 \in \mathbb{R}^2$
# 4. Assiciate $a_2$ with a bias term
# 5. Calculate the affine combination and return
# $$ z_3 = W_2 \cdot a_2 $$, where $z_3 \in \mathbb{R}$
# 6. Calculate $a_3 = \sigma(z_3)$. Note that in the lecture note, we do not apply sigmoid function for the last layer.
# However, we applied sigmoid here because we want the result to be in the inteval $[0,1]$ instead of $\mathbb{R}$

# In[5]:


def neural_net(X, W1, W2):
    """Implementation of the MLP
    Input: takes input data X in 2 dimension, 2 weight matrix
    """
    # bias term
    X_b = np.ones((3,1))
    X_b[1:] = X
    a_1 = X_b
    
    # calculate z_2
    z_2 = np.matmul(W1, a_1)
    
    # calculate a_2
    a_2 = sigmoid(z_2)
    
    # associate with bias
    b_2 = np.ones((3, 1))
    b_2[1:] = a_2
    a_2 = b_2
    
    # calculate z_3
    z_3 = np.matmul(W2, a_2)
    
    # calculate a_3
    a_3 = sigmoid(z_3)
    
    return a_3[0][0]


# ## Calculate the gradient with respect to $W_1$ and $W_2$ using Backpropagation

# We are using backpropagation to train $W^{(1)}$ and $W^{(2)}$
# 
# First, perform a full forward propagation of the neural network to obtain
# $a_1, a_2, a_3, z_2, z_3$
# 
# Second, calculate
# $$ \delta^{(3)} = a_k^{(3)} - y_k $$
# $$ \delta^{(2)} = W_2' \delta^{(3)}.*\sigma'(z^{(2)}) $$
# 
# Finally, calculate the gradient at layer $l$ using
# \begin{equation}
# \frac{\partial}{\partial W_{ij}^{(l)}} = \delta^{(l + 1)} \cdot (a^{(l)})'
# \end{equation}

# In[6]:


def neural_net_train(X, W1, W2, Y):
    """Implementation of the MLP
    Input: takes input data X in 2 dimension, 2 weight matrix, y
    Output: 
    """
    # bias term
    X_b = np.ones((3,1))
    X_b[1:] = X
    a_1 = X_b
    
    # calculate z_2
    z_2 = np.matmul(W1, a_1)
    
    # calculate a_2
    a_2 = sigmoid(z_2)
    
    # associate with bias
    b_2 = np.ones((3, 1))
    b_2[1:] = a_2
    a_2 = b_2
    
    # calculate z_3
    z_3 = np.matmul(W2, a_2)
    
    a_3 = sigmoid(z_3)
    # forward propagation end
    # -----------------------
    
    # start of backpropagation
    
    # delta 3 in  1 x 1
    delta_3 = a_3 - Y 
    
    # delta 2 in 2 x 1
    delta_2 = np.multiply(np.matmul(W2.T, delta_3)[1:], sigmoid_grad(z_2))
    
    # we don't have delta 1
    grad_w_1 = np.matmul(delta_2, a_1.T)
    
    grad_w_2 = np.matmul(delta_3, a_2.T)
    # grad_w_2 = np.matmul(delta_3.T, a_2)
    
    return a_3, grad_w_1, grad_w_2# grad_w_2


# ## Dealing with Multiple training data at the same time
# The gradient formula is averaged. Namely
# $$\frac{\partial}{\partial W_{ij}^{(l)}} = \frac{1}{m} \sum_{i = 1}^{m} \delta_{(i)}^{(l + 1)} \cdot (a_{(i)}^{(l)})' $$

# In[7]:


# accumulate the gradient result
def batch_neural_net_train(X, W1, W2, Y):
    # X contains m training examples as row vectors
    # X is   m x 3
    m, _ = X.shape
    
    grad_w_1_acc = np.zeros((2,3))
    grad_w_2_acc = np.zeros((1,3))
    pred_all = np.zeros((m, 1))
    for i in range(m):
        pred, grad_w_1, grad_w_2 = neural_net_train(np.asmatrix(X[i]).T, W1, W2, np.asmatrix(Y[i]))
        grad_w_1_acc = grad_w_1_acc - grad_w_1
        grad_w_2_acc = grad_w_2_acc - grad_w_2
        pred_all[i] = pred[0][0]
    error = np.sum(np.square(pred_all - Y)) / m
    return pred_all, grad_w_1_acc / m, grad_w_2_acc / m, error


# ## Gradient Decent
# We optimize our weight using gradient decent.
# We update the weight using the following formula:
# $$ W_{ij}^{(l)} = W_{ij}^{(l)} - \alpha * \frac{\partial}{\partial W_{ij}^{(l)}} $$

# In[8]:


def train(X, Y, step = 2000, alpha = 4.0):
    # initiate w_1, w_2
    w_1 = np.random.randn(2,3)
    w_2 = np.random.randn(1, 3)
    
    # train
    for i in range(step):
        pred, grad_w_1, grad_w_2, err = batch_neural_net_train(X, w_1, w_2, Y)
        w_1_new = w_1 + alpha * grad_w_1
        w_2_new = w_2 + alpha * grad_w_2
        w_1 = w_1_new
        w_2 = w_2_new
        print('step: {} - error: {}'.format(i, err))
    return w_1, w_2


# We explicitly print the training error, and after the experiment, we find the optimal $\alpha = 4.0$ and $step = 2000$

# In[9]:


# build up a training data
X = np.asmatrix([[1,1], [1,0], [0,1], [0, 0]])
Y = np.asmatrix([[0], [1], [1], [0]])

# train the neural network
w_1, w_2 = train(X, Y)


# The prediction is just performing the forward propagation algorithms.

# In[10]:


def predict(X, w_1, w_2):
    m, _ = X.shape
    pred = np.zeros((m, 1))
    for i in range(m):
        if neural_net(np.asmatrix(X[i]).T, w_1, w_2) > 0.5:
            pred[i][0] = 1
        else:
            pred[i][0] = 0
    return pred     


# In[11]:


predict(X, w_1, w_2)


# # Bonus Question
# We use the same neural network architecture. However, we give a different training set for the neural network.
# 
# ## Understanding $Sign$ function
# \begin{equation}
# Sign(x) = 
# \begin{cases}
# -1 & x < 0 \\
# 0 & x = 0 \\
# 1 & x > 0
# \end{cases}
# \end{equation}
# 
# However, we can safely ignore the second case when $x = 0$, since the propability for such a case is $0$.
# 
# ### Mapping the data into inteval $[0, 1]$
# We simply define $S(Sign(x)) = \frac{Sign(x) + 1}{2}$. To recover, we have $S^{-1}(x) = 2x - 1$

# In[12]:


def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    return -1

def S(x):
    return (sign(x) + 1) / 2

def S_inv(x):
    return 2 * x - 1


# ## Initiate the data set
# We initiate the dataset using a standard normal detribution.
# Let's take $m = 4000$

# In[13]:


Xs = np.random.randn(4000, 2)
Ys = []
for data in Xs:
    Ys.append(S(data[0] * data[1]))
Ys = np.asmatrix(Ys).T


# ## Train the dataset
# We use the same function for the training.

# In[14]:


M_1, M_2 = train(Xs, Ys, 500, 2.0)


# In[15]:


Ys_pred = predict(Xs, M_1, M_2)


# In[18]:


# calculathe the error ratio
np.sum(np.square(Ys_pred - Ys)) / 4000


# ## Error Analysis
# $24.725\%$ of error is acceptable for such a simple neural network structure. Now we construct a function using the neural net and plot the function together with the original function

# In[41]:


def original_fct(x, y):
    return S(x * y)
def approximate_fct(x, y):
    if neural_net(np.asmatrix([x, y]).T, M_1, M_2) > 0.5:
        return 1
    return 0


# In[42]:


from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
fig = plt.figure()
ax = ax = Axes3D(fig)

x = y = np.arange(-1.0, 1.0, 0.05)
X, Y = np.meshgrid(x, y)
z1s = np.array([original_fct(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z1 = z1s.reshape(X.shape)
z2s = np.array([approximate_fct(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z2 = z2s.reshape(X.shape)

ax.plot_surface(X, Y, Z1)
ax.plot_surface(X, Y, Z2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Value')

plt.show()

