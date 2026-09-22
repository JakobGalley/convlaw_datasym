# Conservation Laws from Data Symmetry in Neural Networks
This repository contains the code supplements for the paper "Conservation Laws from Data Symmetry in Neural Networks"

## Requirements
To run the code, you will need [python](https://www.python.org/) with the following packages
- numpy
- matplotlib
- plotly
and their dependencies.
We tested the code under python 3.12 and the minimal versions of packages provided in 'requirements.txt'.
To install the requirements, you can install them by running
```bash
pip install requirements.txt
```
in the terminal.

## How to run the code
The scripts are self-contained and can thus easily run through the command line
```bash
polynomial_loss_landscape
python linear_conservation_law
python attention_conservation_law
```
which will generate Figure 1, 2 and 3 in the paper.

If you want to change parameters, you will have to change these in the files - we marked them.

## Polynomial loss landscape
In `polynomial_loss_landscape.py`, we plot the original and augmented training data as well as the corresponding loss landscapes for a polynomial neural network.

The original training data is given in `X_data` and `y_data` which are four points in $\mathbb{R}^2$ and $\mathbb{R}$ respectively.
`X_data` are created such that the points are not scalar multiples of each other and have different norm.
The original training data will be then augmented under the group of rotations of integer multiples of $2\pi / n$ where $n$ corresponds to `num_group_elements`.
This is done by adding a tensor dimension to `X_aug` which makes identification with `y_data` trivial, so we don't need to augment that it.

The model is defined as 

$$f(z) = 0.012 z^6 - 0.22 z^4 + z^2$$ 

where $z=px$ and $p$ are the weight parameters of the model.

By multiplying the parameter space by the training data, we generate the $z$-data corresponding the parameter space (`z_plain` and `z_aug`).
Then, we calculate the MSE loss for these parameters, i.e.

$$\mathrm{loss} = \mathbb{E}[\|f(z) - y\|^2 ]$$

with the respective $z$-data and $y$-data.

The result of those, create the loss landscape which we plot in the upper 3d plots.
Under these 3d plots, we plot the corresponding training data.

## Conservation Laws for Linear model

We train a linear model with base and $C_3$-augmented training data and then compare over multiple runs the error in the conservation law.
The result will then be plotted as in Figure 2 showing the individual runs and the mean.

For the number of base points `n_base=4`, we create random training data `X_base` and `y_base` in $\mathbb{R}^3$ and $\mathbb{R}$ respectively.
These will then be augmented by $C_3$ permutation of the indices in $\mathbb{R}^3$ and trivial action on $\mathbb{R}$ in `X_aug` and `Y_aug`.

Then, we set up exact gradient descents for the original and augmented training data for a simple linear model with MSE loss, as well as computation of the conservation laws

$$ h(W_1, W_2, W_3) = \frac{W_1 + W_2 - 2W_3}{W_1 - W_2}$$

which describes the position of $W_3$ in the range of $W_1$ and $W_2$.
Note that this is equivalent to

$$\frac{1 - \lambda}{2} W_1 + \frac{1 + \lambda}{2} W_2 -  W_3 = 0$$

for some $\lambda$.
The weights can thus only follow an affine transformation of the form

$$W(t) = a(t) W(0) + b(t) \mathrm{1}$$

for some $a(t), b(t) \in \mathbb{R}$ as expected.

Then, we calculate the deviations of $h(W)$ through time for different initializations. 
These runs and their mean will then be plotted.

## Conservation Laws for Lightning Attention model

We train a Lightning Attention model with base and $\mathrm{O}(d)$ sampled augmented training data and then compare over multiple runs the error in the conservation law.
The result will then be plotted as in Figure 3 showing the individual runs and the mean.

For the number of base points `n_base=10`, we create random training data `X_base` in $\mathbb{R}^{n\times d}$ for `n=5` and `d=8` by Gaussian initialization and scaling it with `data_scale`.
The output data `Y_base` in $\mathbb{R}^{n\times m}$ for `m=1` is created by a pass through the Lightning Attention model for random `teacher_scale` scaled weights.

These will then be augmented by $\mathrm{O}(d)$ which we sample by $QR$-decomposition.
For a random matrix $A$, we return the orthogonal matrix $Q$ in the $QR$-decomposition and then change the signs accordingly to $R$.
We sample `num_haar_pairs` samples of orthogonal matrices and also add their additive inverse to `G_samples`.
In the same way as in the linear model, we augment `X_base` and `y_base` by `G_samples` to create `X_aug` and `Y_aug`.

Then, we setup exact gradient descents for the original and augmented training data for a simple Lightning Attention model with MSE loss.
The Lightning Attention model is of the following form

$$ f_{Q, K, V}(X) = XQ K^TX^TXV $$

for $Q, K \in\mathbb{R}^{d\times r}$, $V\in\mathbb{R}^{d\times m}$ and $X \in\mathbb{R}^{n\times d}$ for `r = 1`.
We calculate the conversation law by comparing the orthogonal projection operators of $P = [Q \ K \ V]$.
Let $P = USV$ be the SVD decomposition of $P$, then the orthogonal projection to the range of $P$ is given by $UU^T$.

For our multiple training runs - here $6$ - we initialize the weights as `init_scale` scaled random weights such that their `minimum_singular_value` of `P0` is high enough.
Then, we plot the difference in the Frobenius norm in the orthogonal projection matrix to the range of $P(t)$ - and also the mean.
