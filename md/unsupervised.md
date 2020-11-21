---
title: Unsupervised learning - synthesis
author: Louis Lesueur
---

## Definitions

unlike supervised learning, in unsupervised learning the input data are not labelized. The goal is to fond patterns in them. In other words, the dataset $\mathcal{D}$ is a radom sample $\{X_1, \dots, X_n \}$ from an unknown random variable $X$, and we want to find patterns in them.

## Dimension reduction

### Principal component analysis

#### Some reminders

+ Correlation matrix of the dataset: $K_n = \frac{1}{n} \sum_i (x_i- \bar{x_n})^T (x_i-\bar{x_n}) = \frac{1}{n} \sum_i x_i^Tx_i - \bar{x_n}^T \bar{x_n}$
+ Empirical variance: $\sigma_n^2 = \frac{1}{n} \sum_i ||x_i - \bar{x_n}||^2 = tr(K_n)$
+ Empirical correlation between features: $Corr(x^j, x^{j'}) = \frac{K_n^{jj'}}{\sqrt{K_n^{jj}} \sqrt{K_n^{j'j'}}} = \frac{K_n^{jj'}}{\sigma_n^j \sigma_n^{j'}}$

#### PCA principle

To apply PCA, we suppose that data is centered (ie $\bar{x_n}=0$). If the features are too different, the dataset could benefit from standardization.

Suppose that they are $p$ features (ie a data vector is in $\mathbb{R^p}$), the goal of PCA is to find an orthogonal projection on a subspace with dimension $k<p$ that best preserve the original shape of the set, that 'lose less information possible'.

As the data are centered, and using Pythagoras's theorem, orthogonal projection on $H$ gives:
$$
\sigma_n^2 = (\sigma_n^H)^2 + \frac{1}{n} \sum_i ||x_i - x_i^H||^2
$$

We want to maximize $(\sigma_n^H)^2$, to preserve the maximum of information. But, for $H_k = Span(e_1, \dot, e_k)$ we have:
$$
(\sigma_n^{H_k})^2 = \sum_{j=1}^k \lambda_j
$$
where the $\lambda_j$ are the eigenvalues of $K_n$.

And, for all linear spaces $H$ with dimension $k$, we have: $(\sigma_n^H)^2 \leq (\sigma_n^{H_k})^2$. Hence, **the computation of optimal subspaces is reduced to the diagonalisation of the empirical covariance matrix**.

#### Interpretation of PCA

The $l$-th principal component is the column vector $c^l = \sum_{j=1}^p e_l^j x^j$

The principal components are uncorrelated: $Corr(c^l, c^{l'}) = 0$ if $l \neq l'$, furthermore: $\forall l \in \{1, \dots, p \}, Corr(c^l, x^j) = \frac{\sqrt{\lambda_l}e_l^j}{\sigma_n^j}$ and:
$$
\sum_{l=1}^p Corr(c^l, x^j)^2 = 1
$$

The last equation implies that the point of $\mathbb{R}^p$ with coordinates $(Corr(c^1,x^j), \dots, Corr(c^p,x^j))$ are on the unit sphere of $\mathbb{R}^p$. It is called the correlation sphere. The plane spanned by $(c^1, c^2)$ is called the first factorial plane. When a feature is closed to the center of a factorial plan, it means that the selected principal component aren't enough to explain its variance in the dataset.

#### Proportion of variance explained by PCA:

The propotion of variance explained by PCA is the ratio:

$$
\frac{(\sigma_n^{H_k})^2}{\sigma_n^2} = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^p \lambda_j}
$$

It is usefull to control how much information is lost by performing dimension reduction with PCA.

### Independent component analysis

## Clustering

### $k$-means

In $k$-means, we want to partition the data in $k$ subsets $S = \{S_1,\dots,S_k\}$ so as to minimize the within-cluster sum of squares (which is linked to variance). Formally, we want to find:

$$
\arg \min_S \sum_{i=1}^k \sum_{x \in S_i} ||x- \mu_i ||^2 = \arg \min_S \sum_{i=1}^k |S_i| \text{Var} S_i = \arg \min_S \sum_{i=1}^k \frac{1}{2|S_i|} \sum_{x,y \in S_i} ||x - y ||^2
$$

To solve the problem, the idea is to chose $k$ initial points for cluster centroids, calculate the Voronoi regions associated to them and update them by the formed cluster means.

### hierarchical clustering

In hierarchical clstering, for a given metric and a linkage function (distance between clusters). The clustering consists in building a dendogram following the steps (bottom-up approach is presented here, but one can also imagine top-down methods):

1. Put each data in a unique cluster
2. Compute the pairwise linkage between each cluster
3. Group the two clusters with smallest linkage
4. Repeat steps two and three until there is on cluster

Here are some common linkage functions:

| Name | Formula ($A$ and $B$ are two sets of observation)
|-|---
|Maximum | $d(A,B) = \max(d(a,b) | a \in A, b \in B)$
|Minimum | $d(A,B) = \min(d(a,b) | a \in A, b \in B)$
|UPGMA   | $d(A,B) = \frac{1}{|A| |B|} \sum_{a \in A} \sum_{b \in B} d(a,b)$
|WPGMA   | $d(A \cup B, C) = \frac{d(A,C) + d(B,C)}{2}$
|UPGMC   | $d(1,B) = ||\mu_A - \mu_B||$ where $\mu_A$ and $\mu_B$ are the centroids of the clusters
|Energy distance | $d(A,B) = \frac{2}{nm} \sum_{i,j} ||a_i - b_j||_2 - \frac{1}{n^2} \sum_{i,j} ||a_i - a_j||_2 - \frac{1}{m^2} \sum_{i,j} ||b_i - a_j||_2$
|Ward distance | $d(A,B) = \frac{nm}{n+m}d(\mu_A, \mu_B)$

### Density based methods: DBSCAN and OPTICS

DBSCAN use the notion of neighbourhood to perform clustering. There are two parameters:

+ $\epsilon$, the radius of the neighborhood
+ $m$ the minimum number of points in an $\epsilon$-neighborhood

For $p \in \mathcal{D}$, let consider $N_\epsilon(p) = \{ q \in \mathcal{D} | d(p,q) < \epsilon \}$. We say that:

+ $q$ is directly density-reachable from $p$ if $q \in N_\epsilon(p)$ and $|N_\epsilon(p)| \geq m$.
+ $q$ is density-reachable from $p$ if there exist a sequence of two by two directly density-reachable points between them.
+ $q$ is density-connected to $p$ if there exist $o \in \mathcal{D}$ such as both $p$ and $q$ are density-reachable from $o$

Then, clusters are build by grouping points according to density-connectivity. And for a given cluster, we can distinguish core points (with dense neighborhood) from border points (which are in a cluster but with not dense neighborhood), and from noise (which are neither core neither border).

OPTICS algorithm is another density-based clustering methods, using the same idea.


### Expectation–maximization (EM)

## Density estimation

See [here](https://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf) for a detailed theory

### Kernel density estimation

In kernel density estimation, we want to find the density function of the data-distribution.

Naturally, the distribution $F$ can be estimated by the EDF: $\hat{F}(x) = n^{-1}\sum_i \mathbb{1}_{\{X_i \leq x}$.

To have an estimation of $f$ which is not a set of mass points, one can consider a discrete derivative of the form: $\hat{f}(x) = \frac{\hat{F}(x+h)-\hat{F}(x-h)}{2h} = \frac{1}{nh} \sum_i k(\frac{X_i-x}{h})$ with $k(u) = \frac{1}{2} \mathbb{1}_{|u| \leq 1}$. It is a special case of kernel estimator, which have the general form:

$$
\hat{f}(x) =\frac{1}{nh} \sum_i k(\frac{X_i-x}{h})
$$

where $k$ is a kernel function (ie $\int_\mathbb{R}k = 1$)

#### Some kernel functions and properties

+ A non-negative kernel is such as: $k \geq 0$ on $\mathbb{R}$ (in this case $k$ is a probability density)
+ The moments of a kernel are: $\kappa_j(k) = \int_\mathbb{R} u^jk(u)du$
+ A symmetric kernel satisfies $k(u)=k(-u)$ for all $u$. In this case, all odd moments are zero.
+ The order $\nu$ of a kernel is defined as the order of its first non-zero moment.

Here are some second order kernels:

| Kernel | Equation |
|-|-|
Uniform | $\frac{1}{2} \mathbb{1}_{\{|u| \leq 1 \}}$
Epanechnikov | $\frac{3}{4}(1-u^2) \mathbb{1}_{\{|u| \leq 1 \}}$
Gaussian | $\frac{1}{\sqrt{2 \pi}} \exp{(-\frac{u^2}{2})}$

## Anomaly detection
+ Local Outlier Factor
+ Isolation Forest

## Neural Networks

### Autoencoders

An autoencoder is a NN that is tarined to attempts to copy its input to its output. It is composed of two parts:

+ The encoding function: $f: \mathcal{X} \rightarrow \mathcal{F}$
+ The decoding function: $g: \mathcal{F} \rightarrow \mathcal{X}$

where $\mathcal{F}$ is the code space.

In fact, an autoencoder is learning the conditionnal distribution $p_{AE}(h|x)$ where $h \in \mathcal{F}$. And we have: $p_{encoder}(h|x) = p_{AE}(h|x)$ and $p_{decoder}(x|h) = p_{AE}(x|h)$. And the loss can be seen as a maximum-likelihood maximization, just like in supervized methods.

#### Vanilla autoencoders

+ Undercomplete autoencoder: code space dimension less than the input space.
+ Overcomplete autoencoder : code space dimension greater than the input space.

The learning consists in minimizing a loss function:
$$
L(x,g(f(x))) = ||x-g(f(x))||^2
$$

If $g$ and $f$ are linear, the undercomplete autoencoder is simply learning PCA subspaces ! So non-linear autoencoder can be seen as a non linear generalization of PCA.

#### Regularized autoencoders

If the encoder and the decoder have too much capacity, it is possible that they don't learn anything on the data distribution, but only specific things on the dataset (it is a kind of overfitting). For example, one can imagine an autoencoder which maps $x_i$ to $i$ and $i$ to $x_i$. The learned subset of indexes tells nothing about the data distribution. That's where regularization join the game.

##### Sparse Autoencoders (SAE)

A sparse autoencoder involves a saprsity penalty $\Omega(h)$ on the code layer $f(x)=h$:
$$
L(x,g(f(x))) + \Omega(h)
$$

They are typically used to learn features for other tasks such as classification.

|Regularization therm | $\Omega(h)$ | Remarks
|--|---|---|
| KL                  | $\sum_j KL(\rho || \frac{1}{n} \sum_i(h_j(x_i)))$ | $\rho$ is the sparcity parameter, close to zero. This regularization penalizes average activation of the neurones from $h$ on the dataset for deviating from $\rho$, and so force them to be inactive most of the time.
| $L^1$ and $L^2$ | $\lambda ||h||$ | It is known that these regularizations acheive sparcity.

##### Contractive Autoencoders (CAE)

Another regularization strategy consists in penalizing the gradient:
$$
L(x,g(f(x))) + \lambda \sum_i || \nabla_x h_i ||^2
$$

This forces the model to learn a function that does not changes much when $x$ changes slightly.

#### Denoising Autoencoders (DAE)

A denoising autoencoder minimizes:
$$
L(x,g(f(\tilde{x})))
$$

where $\tilde{x}$ is a copy of $x$ that has been corrupted by some noise.

#### Variational Autoencoders (VAE)


### Generative adversarial networks

### Self-organizing map

### Deep Belief Nets

### Hebbian Learning
