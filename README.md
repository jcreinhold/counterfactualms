Structural causal model for MR images of multiple sclerosis 
===========================================================

This repository holds code to create counterfactual images for MR brain images people with (and without) MS.

This code was used to implement the experiments in our paper ["A Structural Causal Model of MR Images of Multiple Sclerosis"](https://arxiv.org/abs/2103.03158).

This work builds on the work of Pawlowski, Castro, and Glocker [1]. This is a fork of their 
code which can be found [here](https://github.com/biomedia-mira/deepscm).

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
  1. Pawlowski, Nick, Daniel Coelho de Castro, and Ben Glocker. 
     "Deep Structural Causal Models for Tractable Counterfactual Inference."
     Advances in Neural Information Processing Systems 33 (2020).
