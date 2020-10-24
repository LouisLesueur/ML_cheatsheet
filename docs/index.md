# Supervised Learning cheat sheet

## Data representation

+ pre treatment
+ missing data
+ defining a metric (cf LMNN DM)

## Discriminative problems

### Parametrics Models (GLM, SVM...)

#### Loss and risk functions

$$
\min_w \frac{1}{n} \sum_{i=1}^{n} l ( f_w(x_i), y_i ) + \lambda C(w)
$$

where:

+   $l$ is a loss function
+   $f_w$ is the prediction function ($w$ is the parameter to optimize)
+   $C$ is a regularization function and $\lambda$ a regularization factor


Here the model are linear, $f_w(x) = w^Tx$, and for a given dataset in $\mathcal{X}\times\mathcal{Y}$, $\phi_i(\cdot) = l ( \cdot, y_i )$:
We can apply Fenchel duality and we have:

|Primal                                                   | Dual                                                                                            |
|---------------------------------------------------------|-------------------------------------------------------------------------------------------------|
|$\min_w \frac{1}{n} \sum_{i=1}^{n} \phi_i(w^Tx_i) + C(w)$|$\max_\alpha \frac{1}{n} \sum_{i=1}^{n} -\phi_i^*(-\alpha_i ) - C^*(\sum_{i=1}^{n} \alpha_i x_i)$|

Where $\phi_i^*$ and $C^*$ are the convex conjugate of $\phi_i$ and $C$ (defined by $f^*(a) = \max_z (za - f(z))$).
And $x_i^*(\cdot)$ is the adjoint operator of $(\cdot^Tx_i)$ (${w^T}^* = \overline{w}$)

It can be shown that if $\hat{w}$ and $\hat{\alpha}$ are the optimal solutions of those problems, then we have:
$$
P(\hat{w}) = P(w(\hat{\alpha})) = D(\hat{\alpha})
$$

#### Loss functions and corresponding problems:

|Name      |$l(x,y)$        |$l^*(-a,y)$                                    | problem
|----------|----------------|-----------------------------------------------|----------------------------------------
|$L^2$     |$(x-y)^2$       |$-ay + \frac{a^2}{4}$                          | linear regression (GLM normal, SVM regressor)
|Logistic  |$\ln(1+e^{-xy})$|$ay_i\ln(ay) + (1-ay)\ln(1-ay)$, $ay \in [0,1]$| logistic regression (GLM bernoulli)
|Hinge loss|$\max(0,1-xy)$  |$-ay$, $ay \in [0,1]$                          | classification (SVM)

Regularizations:

+   $L^2$: $C(w) = ||w||^2_2$
+   $L^1$: $C(w) = ||w||_1$

#### The kernel trick for non-linear models

kernels

#### Solvers

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

## Problems

test

### Regression

test

### Classification

test
