---
title: Anatomy of a Probabilistic Programming Framework
excerpt:
tags:
  - probabilistic programming
  - pymc3
  - pymc4
  - stan
  - tensorflow probability
  - pyro
  - edward
header:
  overlay_filter: 0.3
  overlay_image: /assets/images/cool-backgrounds/cool-background2.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
toc: true
toc_sticky: true
toc_icon: "kiwi-bird"
last_modified_at: 2019-09-20
---

Probabilistic programming languages (PPLs)

I assume you know a bit about probabilistic programming and Bayesian modelling. 

https://eigenfoo.xyz/bayesian-inference-reading/

## The Anatomy of a Probabilistic Programming Framework

A probabilistic programming framework needs six things:

1. A language or API for users to specify a model
1. A library of probability distributions and transformations to build the model
   density
1. At least one inference algorithm to sample from the posterior, or obtain some
   approximation of it
1. At least one optimizer to find the mode of the model density
1. An autodifferentiation library to compute gradients required by the inference
   algorithm and optimizer
1. A suite of diagnostics to monitor and analyze the quality of inference

All those pieces come together like so:

INSERT DIAGRAM HERE

Let's break them down one at a time.

### Specifying the model: language/API

How will users specify their model?

Do you believe that Python is the best language to specify models?

Python is more hackable and you don't need to learn a new language. On the other
hand, Python forces some deep-seated abstractions onto users.

### Building the model density: distributions and transformations

Tensorflow Probability: "distributions vs bijectors"

(There are some rumblings on [normalizing
flows](https://arxiv.org/abs/1505.05770) that I am confident I do not
understand).

This is by far the hardest thing:

There is the idea of splitting up distributions and transformations of random
variables (a.k.a. bijectors).

1. Must build the model log likelihood in some way
1. Be careful not to get bitten: theoretically, the model can be the same for
   sampling, inference and debugging, but in practice, the implementation needs
   to change. E.g. Stan must be rewritten if you have missing data (cite Gelman
   paper), but PyMC3 and Pyro go to great lengths to make sure you don't need
   to.
1. Shapes! Shapes are hard!
   https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
   and TFD paper (section on shape semantics)

### Computing the posterior: inference algorithm

This is the part that actually implements inference: given a model and some
data, compute the posterior (either by sampling from it, in the case of MCMC, or
by approximating it, in the case of VI).

Most probabilistic programming frameworks out there either implement HMC (or
NUTS, or some variant thereof) or VI.

### Computing the mode: optimizer

Sometimes, instead of performing full-blown inference, it is useful to find
modes of the density specified by the model. These modes can be used as point
estimates of parameters, or as the basis of approximations to a Bayesian
posterior. Thus, this calls for an optimizer.

A simple and sensible thing to do is to use some [BFGS-based optimization
algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
(e.g. some quasi-Newton method like
[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) and call it a day.

### Computing the gradients: autodifferentiation library

Both the inference algorithm and the optimizer rely on gradients (at least, if
you're using reasonable inference algorithms and optimizers!), and so you'll
need some way to compute these gradients.

The easiest thing to do would be to rely on one of the deep learning frameworks
like TensorFlow or PyTorch.

Is this a good idea? Forces you into thinking of your model density as a
graphical model.

### Monitoring inference: diagnostics

E.g. R hats, n_eff, etc. Very easy.

## A Zoo of Probabilistic Programming Frameworks

### Stan

Very easy answers to all of these questions: literally everything is implemented
from scratch in C++.

There is a compiler for [a small domain-specific language for specifying
Bayesian models](https://github.com/stan-dev/stan/tree/develop/src/stan/lang),
libraries of [probability
distributions](https://github.com/stan-dev/math/tree/develop/stan/math/prim) and
[transforms](https://github.com/stan-dev/math/tree/develop/stan/math/prim/scal/fun),
implementations of [dynamic
HMC](https://github.com/stan-dev/stan/tree/develop/src/stan/mcmc/hmc) and
[variational
inference](https://github.com/stan-dev/stan/tree/develop/src/stan/variational),
[a separate autodifferentiation
library](https://github.com/stan-dev/math/tree/develop/stan/math), an [L-BFGS
based
optimizer](https://github.com/stan-dev/stan/tree/develop/src/stan/optimization)
and a [suite of
diagnostics](https://github.com/stan-dev/stan/tree/develop/src/stan/analyze/mcmc).

> Rolling your own autodifferentiation library is the path of a crazy person.

Stan's autodiff is optimized for statistical modelling though (it doesn't rely
on the GPU).

Contrary to popular belief, Stan does not implement NUTS:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stan implements a dynamic Hamiltonian Monte Carlo method with multinomial sampling of dynamic length trajectories, generalized termination criterion, and improved adaptation of the Euclidean metric.</p>&mdash; Dan Simpson (<a href="https://twitter.com/dan_p_simpson">@dan_p_simpson</a>) <a href="https://twitter.com/dan_p_simpson/status/1037332473175265280">September 5, 2018</a></blockquote>

And in case you're wondering what Stan _actually_ implements:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Adaptive HMC. <a href="https://twitter.com/betanalpha">@betanalpha</a> is reluctant to give it a more specific name because, to paraphrase, that’s just marketing bullshit that leads to us celebrating tiny implementation details rather than actual meaningful contributions to comp stats. This is a wide-ranging subtweet.</p>&mdash; Dan Simpson (<a href="https://twitter.com/dan_p_simpson">@dan_p_simpson</a>) <a href="https://twitter.com/dan_p_simpson/status/1034098649406554113">August 27, 2018</a></blockquote>

### TensorFlow Probability

### PyMC3

### Pyro

### Edward and Edward2

### PyMC4

---

FOOTNOTES AND REFERENCES HERE
