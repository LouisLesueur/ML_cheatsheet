---
title: ML woth graphical models - synthesis
author: Louis Lesueur
---

# THIS IS A WORK IN PROGRESS


## Definitions

Graphical models are usefull when the output set is structured (ex: connected body parts on a picture, grammatical functions in a sentence...).

There are several types of graphical models to do that:

+ Directed models : Bayesian networks (BN)
+ Undirected models: Markov Random Fields (MRF)
+ Other : chain graphs, influence diagrams...

The graph encodes conditional independance assumptions:

$$
p(y_i | y_{V \ {i}}) = p(y_i | y_{N(i)})
$$


## Factor graphs

Both directed and undirected models can be represented by factor graphs with:

+ variable nodes $V$ (represented by circles)
+ factor nodes $\mathcal{F}$ (represented by boxes)
+ edges $\mathcal{E}$ between factors and variables
+ a factor $F \in \mathcal{F}$ connects a subset of nodes, noted $y_F$

a factor graph represents a factorization of the type:

$$
p(y) = \frac{1}{Z} \prod_{F \in \mathcal{F}} \psi_F(y_F)
$$

where $\psi$ are "potentials" defined on factors, and $Z$ is a normalization constant, called the "partition function":

$$
Z = \sum_{y \in \mathcal{Y}} \prod_{F \in \mathcal{F}} \psi_F(y_F)
$$

### Conditional distributions

To add the fact that outputs in $\mathcal{Y}$ are conditioned by inputs in $\mathcal{X}$, potentials becom also function of $x$:

$$
p(y | x) = \frac{1}{Z(x)} \prod_{F \in \mathcal{F}} \psi_F(y_F, x_F)
$$

and:

$$
Z(x) = \sum_{y \in \mathcal{Y}} \prod_{F \in \mathcal{F}} \psi_F(y_F, x_F)
$$

### Energy

When the potentials are positive, reasoning in term of "energy" is more conveniant for minimization (see next):

$$
E_F(y_F, x_F) = - \log (\psi_F(x_F, y_F))
$$

and:

$$
E(x,y) = \sum_{F \in \mathcal{F}} E_F(y_F, x_F)
$$

## Inferences on factor graphs

When $p(y|x)$ is known (ie, when an expression of the energy is known), predict $f: \mathcal{X} \rightarrow \mathcal{Y}$ can be done by MAP:

$$
\hat{y} = \argmax_{y \in \mathcal{Y}} p(y | x) =  \argmin_{y \in \mathcal{Y}} E(y,x)
$$

### Belief propagation

To solve the minimization problem, one need to compute $p(y|x)$ for any $y \in \mathcal{Y}$, and so $Z(x)$ which is very expensive. When graphs are small it can be done by hand (variable elimination), else Belief propagation is the most common method.
