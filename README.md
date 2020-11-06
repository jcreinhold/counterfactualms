Counterfactual multiple sclerosis analysis
==========================================

This repository hold code to support counterfactual analysis for MS subjects

This work and code builds off of the work of Pawlowski et al. [1]. Their code is 
[here](https://github.com/biomedia-mira/deepscm)

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers 
of the [Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

Installation
------------

From inside this directory, run:

    python setup.py install

or (if you'd like to make updates to the package)

    python setup.py develop

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests
 
Structure
---------
This repository contains code and assets structured as follows:

- `counterfactualms/`: contains the code used for running the experiments
    - `arch/`: model architectures used in experiments
    - `datasets/`: script for dataset generation and data loading used in experiments
    - `distributions/`: implementations of useful distributions or transformations
    - `experiments/`: implementation of experiments
    - `submodules/`: git submodules
- `assets/`
    - `data/`:
        - `calabresi/`: subset of the calabresi dataset
    - `models/`: checkpoints of the trained models

References
----------
  1. Pawlowski, Nick, Daniel C. Castro, and Ben Glocker. 
     "Deep Structural Causal Models for Tractable Counterfactual Inference."
     arXiv preprint arXiv:2006.06485 (2020).

