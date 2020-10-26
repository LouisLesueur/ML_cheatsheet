---
title: Supervised learning - synthesis
---

## Data representation

### Definitions

+ $X$: real random vector taking values $x \in \mathcal{X}$
+ $Y$: real random vector taking values $y \in \mathcal{Y}$
  + For a regression problem: $\mathcal{Y}=\mathbb{R}$
  + For a classification problem: $\mathcal{Y} = \{C_k \}_{k<K}$ ($K$ classes)
+ $p_{XY}$ joint distribution of $(X,Y)$, unknown

### Pre-treatments

TO-DO

### Handling missing data

TO-DO


## Discriminative problems

In discriminative problems, we have a labelized sample $\mathcal{D}_n = \{(x_i, y_i) \in \mathcal{X}\times\mathcal{Y}\}$, from $p_{XY}$, and we want to determine $p(y | x)$

### Loss and risk functions

Goal of a machine learning algorthm: find function $f$ that can predict new values with the best accuracy.

+ Loss function $L(y_{pred},y)$: measure the error of the predictor.
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
\mathcal{L}(f_\theta) = \sum_{i=0}^n \ln{p_\theta(y_i|x_i)}
$$

The corresponding MLE estimator is: $\hat{f_\theta} = \arg \max_{\theta} \mathcal{L(f_\theta)}$

MLE can be seen as a particular case of risk minimization, with $L(f_\theta(x), y)=-\ln{p_\theta(y|x)}$  (for a $L^2$ loss, with a linear model and a normal error, the Least-square model naturally appears). And we have:

$$
R_{emp}(f_\theta) = - \frac{1}{n} \sum_{i=0}^n \ln{p_\theta(y_i|x_i)}
$$

$$
R(f_\theta) = \mathbb{E}(-\ln{p_\theta(Y|X)})
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
|-------------|-----------------|-------------------|---------------------------|--------|
|Bernoulli    | $\ln{\frac{\phi}{1-\phi}}$ | y      | $\ln(1+e^{\eta})$               | 1
|Gaussian     | $\mu$                       | y      | $\frac{\eta^2}{2}$               | $\frac{1}{\sqrt{2\pi}} \exp(\frac{-y^2}{2})$
|Poisson      | $\ln{\lambda}$             | y      | $\exp{\eta}$                     | $\frac{1}{y!}$
|Geometric    | $\ln{(1-\phi)}$              | y      | $\ln \frac{e^\eta}{1-e^\eta}$   | 1
| Multinomial ($k$ classes) | $[\ln\frac{\phi_i}{\phi_k}]_{i<k}$ | $T(i) = (0...1...0)$ (defined on integers between 0 and k-1, 1 in the i-th position) | $-\ln(\phi_k)$ | 1

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
|$\min_\theta \frac{1}{n} \sum_{i=1}^{n} L(y_i, \theta^Tx_i) + C(\theta)$|$\max_\alpha \frac{1}{n} \sum_{i=1}^{n} -L^{\#}(y_i,-\alpha_i ) - C^{\#}(\sum_{i=1}^{n} \alpha_i x_i)$|

Where $L^\#$ and $C^\#$ are the convex conjugate of $L$ (for the second variable) and $C$ (defined by $f^\#(a) = \max_z (za - f(z))$).
And $x_i^\#(\cdot)$ is the adjoint operator of $(\cdot^Tx_i)$ (${\theta^T}^\# = \overline{\theta}$)

It can be shown that if $\hat{\theta}$ and $\hat{\alpha}$ are the optimal solutions of those problems, then we have:
$$
P(\hat{\theta}) = P(\theta(\hat{\alpha})) = D(\hat{\alpha})
$$

#### The kernel trick for non-linear models

TO-DO


#### Algorithms to solve the minimization problem

We note:
+ $J_i(\theta) = L( f_\theta(x_i), y_i ) + n\lambda C(\theta)$
+ $J(\theta) = \frac{1}{n} \sum_i J_i(\theta)$

Descent methods ($i_k$ is a random index):

|Name            | type      | update rule    |
|----------------|-----------|----------------|
|FG|primal     | $\theta \leftarrow  \theta - \gamma \nabla_\theta J(\theta)$|
|SGD             |primal     |$\theta \leftarrow  \theta - \gamma \nabla_\theta J_{i_k}(\theta)$|
|Newton          |primal     |$\theta \leftarrow  \theta - Hess^{-1}_\theta(J)\nabla_\theta J(\theta)$|
|SAG             |primal     |$\theta \leftarrow \theta - \frac{\gamma}{n} \sum_i(\nabla_\theta J_{i_k}(\theta) \mathbb{1}_{i=i_k}+y_i\mathbb{1}_{i\neq i_k})$|
|SMO             |dual       ||
|BFGS            |primal     ||
|SDCA            |primal-dual||


### Loss functions and corresponding problems:

A loss function is defined by:

$$
L: \mathcal{Y} \times \mathcal{Y} \rightarrow [0, +\infty )
$$

Detailed loss function theory in _How to compare different loss functions and their risks._

#### Binary classification

ATTENTION: here $f$ is not exactely the predictor !

Binary classifier: $g: \mathcal{X} \rightarrow \{-1,1\}$

In general, $g$ is decomposed as: $g(x) = sgn(h(x))$, with:

+ $h(x) = f(\eta(x))$, the predictor
  + $f: [0,1] \rightarrow \mathbb{R}$ a link function
  + $\eta(x) = \mathbb{P}_{Y | X}(1 | x)$, the learned probability

The optimal link function $f^*$ can generaly be obtained analiticaly, so classification problems are reduced to find an estimation $\hat{\eta}$ of $\eta$.

**To do so, one can set: $h(x) = h_\theta(x)$ ($=\theta^Tx$ most of the time), minimize the empirical risk (using the corresponding loss), and deduce an optimal predictor $\hat{h}$, and $\hat{\eta}(x) = f^{*-1}(\hat{h}(x))$**

For classification problem, loss can be written: $L(y,x) = \phi(yf(x))$ (margin-based loss functions)

The corresponding conditionnal risk: $C_\phi(\eta, f) = \eta \phi(f) + (1-\eta) \phi(-f)$


|Name         |$\phi(v)$        |  $f^*_\phi(\eta)$                      | $f^{*-1}_\phi(v)$ | $C^*_\phi(\eta)$                             | $L^{\#}(y,-a)$ | properties
|----------------|----------------------------|----------------------|----------------------|------------------------------------------|-----------------------|-----------------------------------|
| $0/1$       | $sgn(v)$        | $sgn(2\eta -1)$              |               |                                              |                | Not used in practise because $NP$ hard
| square      | $(1-v)^2$       | $2\eta -1$         |          $\frac{1}{2}(v+1)$          | $4\eta(1-\eta)$                              |                |
| modified LS | $\max(1-v,0)^2$ | $2\eta -1$    | NA                         | $4\eta(1-\eta)$                              |                |
| SVM         | $\max(1-v,0)$   | $sgn(2\eta-1)$     | NA                    | $1-|2\eta -1 |$                              | -ay            |
| Boosting    | $e^{-v}$        | $\frac{1}{2} \ln \frac{\eta}{1-\eta}$ | $\frac{e^{2v}}{1+e^{2v}}$ | $2 \sqrt{\eta(1-\eta)}$                      |                | The loss of a mis-prediction increases exponentially with the value of $-v$
| Logistic    | $\ln(1+e^{-v})$ | $\ln \frac{\eta}{1-\eta}$  | $\frac{e^v}{1+e^v}$           | $-\eta \ln{\eta} - (1-\eta) \ln{(1-\eta)}$ | $ay \ln{ay} + (1-ay) \ln(1-ay)$| It is a GLM (for Bernoulli distribution), equivalent to cross-entropy loss
| Savage      | $\frac{1}{(1+e^v)^2}$ | $\ln \frac{\eta}{1-\eta}$ | $\frac{e^v}{1+e^v}$  | $\eta(1-\eta)$ | | non-convex, better for outliers
| Tangent      | $(2 \arctan(v)-1)^2$ | $\tan(\eta - \frac{1}{2})$ | $\arctan(v) + \frac{1}{2}$ | $4\eta(1-\eta)$ | | non-convex, better for outliers


```{.matplotlib preamble=plot.py}
x = np.linspace(-6,2,100)
plt.figure(figsize=(15,5))
plt.plot(x, np.sign(x), label="0/1")
plt.plot(x, (1-x)**2, label="square")
plt.plot(x, np.maximum((1-x)**2,0), label="modified LS")
plt.plot(x, np.maximum((1-x),0), label="SVM")
plt.plot(x, np.exp(-x), label="Boost")
plt.plot(x, np.log(1+np.exp(-x)), label="Logistic")
plt.plot(x, 1/((1+np.exp(x))**2), label="Savage")
plt.plot(x, (2*np.arctan(x)-1)**2, label="Tangent")
plt.xlim(-1,2)
plt.ylim(-2, 4)
plt.title("Classification loss functions (y=1)")

plt.legend()

```

#### Multilabel classification

##### Softmax

It is built from multinomial GLM. Inversing its link function gives:

$$
\phi_i = \frac{e^{\eta_i}}{\sum_{j=1}^k e^{\eta_i}}
$$

The corresponding predictor:

$$
h_\theta(x) = (\frac{e^{\theta_i^T x}}{\sum_{j=1}^k e^{\theta_j^T x}})
$$

And the log Likelihood to maximize:

$$
l(\theta) = \sum_{i=0}^n \ln( \prod_{l=1}^k \frac{e^{\theta_l^T x_i}}{\sum_{j=1}^k e^{\theta_j^T x_i}})^{\mathbb{1}_{y_i = l}}
$$

The minimization problem is therefor:

$$
\min_\theta - \sum_{i=0}^n y_i \ln(\frac{e^{\theta_l^T x_i}}{\sum_{j=1}^k e^{\theta_j^T x_i}})
$$

#### Regression

|Name  | $L(y,x)$        |$L^(y,-a)$ |  properties  |
|------|-----------------|------------|-------------|
|square| $(y-x)^2$       | $-ay + \frac{a^2}{4}$ |estimates mean label, sensitive to outliers, differentiable everywhere
|absolute | $|y-x|$ | -ay | estimates median label, less sensitive to noise, not differentiable in 0
|Huber | $\frac{1}{2} (y-x)^2$ if $|x-y| < \delta$, $\delta(|y-x| - \frac{\delta}{2})$ otherwise | | "Best of Both Worlds" of Squared and Absolute Loss , Takes on behavior of Squared-Loss when loss is small, and Absolute Loss when loss is large.  Once differentiable
|log-ch | $\ln(\cosh(y-x))$ | | Similar to Huber Loss, but twice differentiable everywhere

```{.matplotlib preamble=plot.py}
x = np.linspace(-2,3,100)

out = (np.abs(1-x)-0.5)
out[np.where(np.abs(1-x)<1)] = 0.5*(1-x[np.where(np.abs(1-x)<1)])**2

plt.figure(figsize=(5,5))
plt.plot(x, (1-x)**2, label="square")
plt.plot(x, np.abs(1-x), label="absolute")
plt.plot(x, out, label="Huber, delta = 1")
plt.plot(x, np.log(np.cosh(1-x)), label="log-ch")

plt.xlim(-1,3)
plt.ylim(-0.5, 4.5)
plt.title("Classification loss functions (y=1)")

plt.legend()

```


#### Regularizations

|Name | $C(w)$ | properties |
|----------|--------|-------------------|
|$L^2$| $||w||^2_2$ | strictly convex (1 solution), differentiable, but relies on all features (dense solution)
|$L^1$| $||w||_1$ | convex, not differentiable, but performs feature selection (sparse solution)
|Elastic net| $\alpha ||w||_1 + (1-\alpha)||w||^2_2$ | strictly convex, not differentialbe
|$L^p$ (often $0<p<1$)| $||w||_p$ |non convex, very sparse solutions, initialization dependant, not differentiable


#### Focus on linear regression

|Name | loss | regularizer | Solution |
|-----|-----|-----|-----|
|OLS (ordinary least square) | square | NA | $\theta = (xx^T)^{-1}xy^T$ |
|Ridge regression | square | $L^2$  | $\theta = (xx^T + \lambda I)^{-1}xy^T$ |
|Lasso regression | square | $L^1$  | no analytical, sub gradient descent


### Non parametric models

#### Finding a metric

kNN
decision trees
random forests
Maximum-entropy Markov models


## Neural networks

TO-DO

## Generative Problems

TO-DO

## Model validation

TO-DO


## Bibliography

+ Rosasco, Lorenzo & De Vito, Ernesto & Caponnetto, Andrea & Piana, Michele & Verri, Alessandro. (2004). Are Loss Functions All the Same?. Neural computation. 16. 1063-76. 10.1162/089976604773135104.
