SCM for MR images of MS
=======================

This repository holds code to generate counterfactual images of MR brain images for people with (and without) MS
using a *structural causal model* (SCM) [1] built in [Pyro](https://github.com/pyro-ppl/pyro).

This code was used to generate the counterfactual images in our paper ["A Structural Causal Model of MR Images of Multiple Sclerosis"](https://arxiv.org/abs/2103.03158).

Our work builds on the work of Pawlowski, Castro, and Glocker [2]. The code in
this repository is a fork of their code which can be found [here](https://github.com/biomedia-mira/deepscm).

The code for the segmentation experiment in our paper can be found [here](https://gist.github.com/jcreinhold/c9bfeb15ec4f768cd6af20e1662da8fc);
it depends on [msseg](https://github.com/jcreinhold/msseg). The exact hyperparameters and information about the data used 
for the segmentation experiment are in the docstring of that script.

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io)
of the [Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

Installation
------------

From inside this directory, run:

    python setup.py install

or (if you'd like to make updates to the package)

    python setup.py develop

Structure
---------
This repository contains code and assets structured as follows:

- `counterfactualms/`: contains the code used for running the experiments
    - `arch/`: model architectures used in experiments
    - `datasets/`: script for dataset generation and data loading used in experiments
    - `distributions/`: implementations of useful distributions or transformations
    - `experiments/`: implementation of experiments
- `assets/`: contains hyperparameters for the experiments listed in the paper

References
----------
  1. Pearl, Judea. Causality. Cambridge university press, 2009.
  2. Pawlowski, Nick, Daniel Coelho de Castro, and Ben Glocker. 
     "Deep Structural Causal Models for Tractable Counterfactual Inference."
     Advances in Neural Information Processing Systems 33 (2020).
