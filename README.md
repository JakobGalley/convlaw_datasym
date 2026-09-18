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

For the number of base points `n_base=4`, we create random training data `X_base` and `y_base` in $\mathbb{R}^3$ and $\mathbb{R}$ respectively.
These will then be augmented by $C_3$ permutation of the indices in $\mathbb{R}^3$ and trivial action on $\mathbb{R}$ in `X_aug` and `Y_aug`.

Then we setup exact gradient descents for the original and augmented training data, as well as computation of the conservation laws

$$ \frac{W_1 + W_2 - 2W_3}{W_1 - W_2}$$

which describes the position of $W_3$ in the range of $W_1$ and $W_2$ (the Pluecker Line of $W_1$ and $W_2$).
Note that this is equivalent to

$$ W_3 = \frac{1 - \lambda}{2} W_1 + \frac{1 + \lambda}{2} W_2 $$

for some $\lambda$.
Because this holds for all permutations of $W$ with different $lambda$'s this restricts the problem such that only scaling is allowed (I presume)

Then, we just the calculate the deviations through time and their mean.
These will then be plotted.

## Conservation Laws for Lightning Attention model

TODO, but similar to above with the conservation law being different.
