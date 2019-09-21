---
title: What is a PPL Framework, Really?
excerpt:
tags:
  - probabilistic programming
  - pymc3
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background1.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2019-09-20
---

Probabilistic programming languages (PPLs)

## The Anatomy of a PPL Framework

1. Some language or API to specify a model
1. Some inference engine: usually MCMC (specifically, HMC) and/or VI
1. Some way to compute gradients (specifically, of the log likelihood of the
   specified model)
1. Some optimizer (usually L-BFGS)
1. Some way to monitor/analyze/diagnose inference (e.g. sampler diagnostics like
   R_hat for MCMC)

INSERT DIAGRAM HERE

### Language/API

Distributions vs Bijectors

(There are some rumblings on [normalizing
flows](https://arxiv.org/abs/1505.05770) that I am confident I do not
understand).

This is by far the hardest thing:

1. Must build the model log likelihood in some way
1. There is the idea of splitting up distributions and transformations of random
   variables (a.k.a. bijectors).
1. Be careful not to get bitten: theoretically, the model can be the same for
   sampling, inference and debugging, but in practice, the implementation needs
   to change.
1. Shapes! Shapes are hard!
   https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
   and TFD paper

### Inference engine

HMC and VI

### Autodifferentiation

Most rely on the deep learning frameworks: TensorFlow or PyTorch. Is this a good
idea?

### Optimizer

Mostly BFGS or some variant.

### Diagnostics

E.g. R hats, n_eff, etc. Very easy.

## A Zoo of PPL Frameworks

### Stan

Very easy answers to all of these questions: literally everything is implemented
by hand in C++.

[Language](https://github.com/stan-dev/stan/tree/develop/src/stan/lang)
[HMC](https://github.com/stan-dev/stan/tree/develop/src/stan/mcmc/hmc)
[VI](https://github.com/stan-dev/stan/tree/develop/src/stan/variational)
[Autodifferentiation](https://github.com/stan-dev/math/tree/develop/stan/math)
[Optimizer](https://github.com/stan-dev/stan/tree/develop/src/stan/optimization)
[Diagnostics](https://github.com/stan-dev/stan/tree/develop/src/stan/analyze/mcmc)

Because Stan builds up the log likelihood (and doesn't, e.g. build and compile a
computation graph), it doesn't have the distinction between
distributions/bijectors. WAIT... IS THIS TRUE?

> Rolling your own autodifferentiation library is the path of a crazy person.

Stan's autodiff is optimized for statistical modelling though (it doesn't rely
on the GPU).

### PyMC3 and PyMC4

### TensorFlow Probability

### Pyro

### Edward and Edward2


---

FOOTNOTES AND REFERENCES HERE
