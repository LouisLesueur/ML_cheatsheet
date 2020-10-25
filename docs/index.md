---
title: Supervised learning - synthesis
---

## Data representation

+ pre treatment
+ missing data
+ defining a metric (cf LMNN DM)

## Discriminative problems

### Loss and risk functions

Goal of a machine learning algorthm: find function $f$ that can predict new values with the best accuracy.

+ Loss function $L(y,y_{pred})$: measure the error of the predictor.
+ Risk function $R(f) = \mathbb{E}_{XY}(L(Y,f(X)))$
+ Empirical risk function $R_{emp}(f) = \frac{1}{n} \sum_i L(y_i, f(x_i))$

The problem real solution is
$$
f^* = \arg \min_f R(f)
$$

To solve the problem, we chose a restricted set of possible function $\mathcal{F}$ (either parametric model, either not). We note:
$$
\tilde{f} = \arg \min_{f \in \mathcal{F}} R(f)
$$

As we only have a sample $\mathcal{D_n}$ of the distribution $p_{XY}$, which is unknown, in practise we compute:
$$
\hat{f} = \arg \min_{f \in \mathcal{F}} R_{emp}(f)
$$

Errors:

+ approximation: $f^*-\tilde{f}$
+ estimation: $\tilde{f}-\hat{f}$
+ total: $f^* - \hat{f}$


### Maximum Likelihood principle

In Machine Learning, most of the time we make assumption on data:

$$
y = f_\theta(x) + \epsilon
$$

with $\epsilon$ iid zero mean noise.

So : $y \sim F(f_\theta(x))$

Let $p_\theta$ be the estimated parametric distribution of $y|x$, then its log-likelihood is defined by:

$$
\mathcal{L}(f_\theta) = \sum_{i=0}^n \log{p_\theta(y|x)}
$$

The corresponding MLE estimator is: $\hat{f_\theta} = \arg \max_{\theta} \mathcal{L(f_\theta)}$

MLE can be seen as a particular case of risk minimization, with $L(f_\theta(x), y)=-\log{p_\theta(y|x)}$  (for a $L^2$ loss, with a linear model and a normal error, the Least-square model naturally appears). And we have:

$$
R_{emp}(f_\theta) = - \frac{1}{n} \sum_{i=0}^n \log{p_\theta(y_i|x)}
$$

$$
R(f_\theta) = \mathbb{E}(-\log{p_\theta(Y|x)})
$$

$$
R(f_\theta) - R(f^*) = \text{KL}(p_\theta, p^*)
$$


#### GLM (Generalized Linear Models)

General linear models are defined by the following assumptions on data:

+ Exponential family (cf table): $p_\theta(y|x, \theta) = b(y) \exp{\eta T(y) - a(\eta)}$
+ $f_\theta(x) = \mathbb{E}(y|x, \theta)$
+ $\eta = \theta^Tx$

|Distribution | $\eta$                      | $T(y)$ | $a(\eta)$                        | $b(y)$ |
|-------------|-----------------------------|--------|----------------------------------|--------|
|Bernoulli    | $\log{\frac{\phi}{1-\phi}}$ | y      | $\log(1+e^{\eta})$               | 1
|Gaussian     | $\mu$                       | y      | $\frac{\eta^2}{2}$               | $\frac{1}{\sqrt{2\pi}} \exp(\frac{-y^2}{2})$
|Poisson      | $\log{\lambda}$             | y      | $\exp{\eta}$                     | $\frac{1}{y!}$
|Geometric    | $\log{1-\phi}$              | y      | $\log \frac{e^\eta}{1-e^\eta}$   | 1

#### Loss minimization

So, minimizing the risk (or maximizing the likelihood) often leads to a problem of the form:

$$
\min_\theta \frac{1}{n} \sum_{i=1}^{n} L( f_\theta(x_i), y_i ) + \lambda C(\theta)
$$

where:

+   $L$ is a loss function
+   $f_\theta$ is the prediction function ($\theta$ is the parameter to optimize)
+   $C$ is a regularization function and $\lambda$ a regularization factor

##### Linear case

Here the model are linear, $f_\theta(x) = \theta^Tx$. For this kind of mdel, Fenchel duality can be used to simply express the dual problem, which can be exploited for the kernel trick.

|Primal                                                   | Dual                                                                                            |
|---------------------------------------------------------|-------------------------------------------------------------------------------------------------|
|$\min_\theta \frac{1}{n} \sum_{i=1}^{n} L(y_i, \theta^Tx_i) + C(\theta)$|$\max_\alpha \frac{1}{n} \sum_{i=1}^{n} -L^*(y_i,-\alpha_i ) - C^*(\sum_{i=1}^{n} \alpha_i x_i)$|

Where $L^*$ and $C^*$ are the convex conjugate of $L$ (for the second variable) and $C$ (defined by $f^*(a) = \max_z (za - f(z))$).
And $x_i^*(\cdot)$ is the adjoint operator of $(\cdot^Tx_i)$ (${\theta^T}^* = \overline{\theta}$)

It can be shown that if $\hat{\theta}$ and $\hat{\alpha}$ are the optimal solutions of those problems, then we have:
$$
P(\hat{\theta}) = P(\theta(\hat{\alpha})) = D(\hat{\alpha})
$$

#### The kernel trick for non-linear models

kernels

### Loss functions and corresponding problems:

#### Binary classification

|Name  | $L(y,x)$        |$L^*(y,-a)$ |  properties | comments |
|------|-----------------|------------|-------------|----------|
|

#### Multilabel classification 

#### Regression

|Name  | $L(y,x)$        |$L^*(y,-a)$ |  properties | comments |
|------|-----------------|------------|-------------|----------|

```{.matplotlib preamble=plot.py}
x = np.linspace(-3,5,100)
fig = plt.figure(figsize=(15,3))

ax = fig.add_subplot(1,3,1)
ax.set_title("L2 Loss, y=1")
ax.plot(x, (x-1)**2)

ax = fig.add_subplot(1,3,2)
ax.set_title("Logistic Loss, y=1")
ax.plot(x, np.log(1+np.exp(-x)))

ax = fig.add_subplot(1,3,3)
ax.set_title("Hinge Loss, y=1")
ax.plot(x, np.maximum(np.zeros_like(x),1-x))

```


Regularizations:

+   $L^2$: $C(w) = ||w||^2_2$
+   $L^1$: $C(w) = ||w||_1$



#### Algorithms to solve the minimization problem

Descent methods

|Name            | type      | update rule    |
|----------------|-----------|----------------|
|Gradient Descent|primal     ||
|SGD             |primal     ||
|SDCA            |primal-dual||
|SAG             |primal     ||
|SMO             |dual       ||
|Newton          |primal     ||
|BFGS            |primal     ||

### Non parametric models

kNN
decision trees
random forests
Maximum-entropy Markov models



# Bibliography

+ Rosasco, Lorenzo & De Vito, Ernesto & Caponnetto, Andrea & Piana, Michele & Verri, Alessandro. (2004). Are Loss Functions All the Same?. Neural computation. 16. 1063-76. 10.1162/089976604773135104.
