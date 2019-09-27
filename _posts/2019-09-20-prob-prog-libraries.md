---
title: Anatomy of a Probabilistic Programming Library — How Do They Work?
excerpt:
tags:
  - probabilistic programming
  - stan
  - pymc3
  - tensorflow probability
  - pyro
  - pymc4
header:
  overlay_filter: 0.3
  overlay_image: /assets/images/cool-backgrounds/cool-background2.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
toc: true
toc_sticky: true
toc_icon: "kiwi-bird"
last_modified_at: 2019-09-20
---

Recently, the PyMC4 development team [submitted an
abstract](https://openreview.net/forum?id=rkgzj5Za8H) to the [_Programs
Transformations for Machine Learning_ NeurIPS
workshop](https://program-transformations.github.io/). However, I realized that
despite knowing a bit about Bayesian modelling, I don't understand how
probabilistic programming library are structured or how they even work, and
therefore couldn't appreciate the design work going into PyMC4. So I dove
straight down the rabbit hole, and here's what I've got!

I assume you know a fair bit about probabilistic programming and Bayesian
modelling, and are familiar with the big players in probabilistic programming
world. If you're unsure, you can [read up
here](https://eigenfoo.xyz/bayesian-inference-reading/).

## Dissecting Probabilistic Programming Libraries

A probabilistic programming library needs to provide six things:

1. A language or API for users to specify a model
1. A library of probability distributions and transformations to build the model
   density
1. At least one inference algorithm, which samples from the posterior or
   computes some approximation of it
1. At least one optimizer, which can compute the mode of the posterior density
1. An autodifferentiation library to compute gradients required by the inference
   algorithm and optimizer
1. A suite of diagnostics to monitor and analyze the quality of inference

These six pieces come together like so:

![Flowchart illustrating how probabilistic programming
libraries](/assets/images/prob-prog-framework-flowchart.png)

Let's break this down one by one.

### Specifying the model: language/API

This is the part that users will use to specify their model. Most libraries just
let you write in some programming language (calling functions and classes in the
library, of course), but others roll their own domain-specific language.

At this point I should point out the Python bias in this post: there are plenty
of interesting and important non-Python probabilistic programming libraries out
there (e.g.  [Greta](https://greta-stats.org/) in R,
[Turing](https://turing.ml/dev/) and [Gen](https://probcomp.github.io/Gen/) in
Julia, [Figaro](https://github.com/p2t2/figaro) and
[Ranier](https://github.com/stripe/rainier) in Scala). I just don't know
anything about any of them.

At any rate, the main question here is whether or not you think
Python/R/Julia/Scala is the appropriate language.

Python is more hackable and you don't need to learn a new language. On the other
hand, Python forces some deep-seated abstractions onto users.

### Building the model density: distributions and transformations

These are the parts that your user's model calls, in order to compile/build the
model itself (whether that means a model log probability, or some loss function
to minimize depends on the particular inference algorithm you're using). By
_distributions_, I mean the probability distributions that the random variables
in your model can assume (e.g. Normal or Poisson), and by _transformations_ I
mean deterministic mathematical operations you can perform on these random
variables, while still keeping track of the derivative of these transformations
(e.g.  taking exponentials, logarithms, sines or cosines).

This is a good time to point out that the interplay between the language/API and
the distributions and transformations is (at least in my opinion) one of the
hardest problems. Here is a short list of the problems

It turns out that such transformations must be [local
diffeomorphisms](https://en.wikipedia.org/wiki/Local_diffeomorphism), and the
derivative information requires computing the log determinant of the Jacobian of
the transformation, commonly abbreviated to `log_det_jac` or something similar.

1. The library needs to keep track of every distribution and transformation that
   the model specifies, while also 

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

Most probabilistic programming libraries out there implement both MCMC
algorithms and VI algorithms, although strength of support and quality of
documentation can lean heavily one way or another. For example, Stan invests
heavily into its MCMC, whereas Pyro has the most extensive support for its SVI
algorithms.

### Computing the mode: optimizer

Sometimes, instead of performing full-blown inference, it's useful to find the
mode of the model density. These modes can be used as point estimates of
parameters, or as the basis of approximations to a Bayesian posterior. Or
perhaps you're performing VI, and you need some way to perform SGD on a loss
function. In either case, a probabilistic programming library calls for an
optimizer.

If you don't need to do VI, then a simple and sensible thing to do is to use
some [BFGS-based optimization
algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
(e.g. some quasi-Newton method like
[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) and call it a day.
However, libraries that focus on VI need to implement an optimizer commonly seen
in deep learning, such as [Adam or
RMSProp](http://docs.pyro.ai/en/stable/optimization.html#module-pyro.optim.optim).

### Computing the gradients: autodifferentiation library

Both the inference algorithm and the optimizer rely on gradients (at least, if
you're using reasonable inference algorithms and optimizers!), and so you'll
need some way to compute these gradients.

The easiest thing to do would be to rely on one of the deep learning frameworks
like TensorFlow or PyTorch.

Is this a good idea? Forces you into thinking of your model density as a
graphical model.

### Monitoring inference: diagnostics

E.g. $$\hat{R}$$s, number of effective samples, etc. Very easy.

## A Zoo of Probabilistic Programming Libraries

Having briefly outlined the deep (dark) underbelly of most probabilistic
programming libraries, it's helpful to go through several of the more popular
Python libraries as examples. I've also tried to links to the relevant source
code in the library where possible.

### Stan

Very easy answers to all of these questions: literally everything is implemented
from scratch in C++.

1. Stan has a compiler for [a small domain-specific language for specifying
   Bayesian models](https://github.com/stan-dev/stan/tree/develop/src/stan/lang)
1. Stan has libraries of [probability
   distributions](https://github.com/stan-dev/math/tree/develop/stan/math/prim)
   and
   [transforms](https://github.com/stan-dev/math/tree/develop/stan/math/prim/scal/fun)
1. Stan implements [dynamic
   HMC](https://github.com/stan-dev/stan/tree/develop/src/stan/mcmc/hmc) and
   [variational
   inference](https://github.com/stan-dev/stan/tree/develop/src/stan/variational)
1. Stan also rolls their own [autodifferentiation library](https://github.com/stan-dev/math/tree/develop/stan/math)
1. Stan has an [L-BFGS based optimizer](https://github.com/stan-dev/stan/tree/develop/src/stan/optimization)
1. Finally, Stan has a [suite of
   diagnostics](https://github.com/stan-dev/stan/tree/develop/src/stan/analyze/mcmc).

> Rolling your own autodifferentiation library is the path of a crazy person.

Stan's autodiff is optimized for statistical modelling though (it doesn't rely
on the GPU).

Note that contrary to popular belief, Stan _does not_ implement NUTS:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stan implements a dynamic Hamiltonian Monte Carlo method with multinomial sampling of dynamic length trajectories, generalized termination criterion, and improved adaptation of the Euclidean metric.</p>&mdash; Dan Simpson (<a href="https://twitter.com/dan_p_simpson">@dan_p_simpson</a>) <a href="https://twitter.com/dan_p_simpson/status/1037332473175265280">September 5, 2018</a></blockquote>

And in case you're wondering what that's called:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Adaptive HMC. <a href="https://twitter.com/betanalpha">@betanalpha</a> is reluctant to give it a more specific name because, to paraphrase, that’s just marketing bullshit that leads to us celebrating tiny implementation details rather than actual meaningful contributions to comp stats. This is a wide-ranging subtweet.</p>&mdash; Dan Simpson (<a href="https://twitter.com/dan_p_simpson">@dan_p_simpson</a>) <a href="https://twitter.com/dan_p_simpson/status/1034098649406554113">August 27, 2018</a></blockquote>

### TensorFlow Probability

1. TFP users write Python.
1. TFP implements their own
   [distributions](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions)
   and
   [transforms](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors)
   (which TensorFlow calls "bijectors"). You can read more about TensorFlow's
   distributions and bijectors in [their arXiv
   paper](https://arxiv.org/abs/1711.10604).
1. TFP implements [a ton of
   MCMC](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc)
   algorithms, and a handful of [VI
   algorithms](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi)
   as well.
1. TFP implements [several
   optimizers](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer),
   including Nelder-Mead, BFGS and L-BFGS.
1. TFP relies on TensorFlow to compute gradients (um, duh).
1. TFP uses Edward2 for diagnostics and visualization (a.k.a. model criticism)??

### PyMC3

1. PyMC3 users write Python code, using a context manager pattern (i.e. `with
   pm.Model as model`)
1. PyMC3 implements its own
   [distributions](https://github.com/pymc-devs/pymc3/tree/master/pymc3/distributions)
   and
   [transforms](https://github.com/pymc-devs/pymc3/blob/master/pymc3/distributions/transforms.py).
1. PyMC3 implements
   [NUTS](https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/hmc/nuts.py),
   (as well as [a range of other MCMC step
   methods](https://github.com/pymc-devs/pymc3/tree/master/pymc3/step_methods))
   and [several variational inference
   algorithms](https://github.com/pymc-devs/pymc3/tree/master/pymc3/variational),
   although NUTS is the default and recommended inference algorithm.
1. PyMC3 (specifically, PyMC3's `find_MAP` function) [relies on
   `scipy.optimize`](https://github.com/pymc-devs/pymc3/blob/master/pymc3/tuning/starting.py),
   which in turn implements a BFGS-based optimizer.
1. PyMC3 [relies on
   Theano](https://github.com/pymc-devs/pymc3/blob/master/pymc3/theanof.py) to
   compute gradients.
1. PyMC3 [delegates posterior visualization and
   diagnostics](https://github.com/pymc-devs/pymc3/blob/master/pymc3/plots/__init__.py).
   to [ArviZ](https://arviz-devs.github.io/arviz/).

- PyMC3's context manager pattern is an interceptor for sampling statements:
  essentially an accidental implementation of effect handlers.
- PyMC3 distributions must simply have a `random` and a `logp` method, in
  contrast to TFP and PyTorch distributions, which implement a whole bunch of
  other things like shape handling, names, parameterizations, etc.

### PyMC4

PyMC4 is still under active development (at least, at the time of writing), but
it's safe to call out the overall architecture.

1. PyMC4 users will write Python, although now with a generator pattern (e.g. `x
   = yield Normal(0, 1, "x")`), instead of the previous context manager.
1. PyMC4 will [rely on TensorFlow distributions (a.k.a.
   `tfd`)](https://github.com/pymc-devs/pymc4/tree/master/pymc4/distributions/tensorflow)
   for both distributions and transforms.
1. PyMC4 will also [rely on TensorFlow for
   MCMC](https://github.com/pymc-devs/pymc4/tree/master/pymc4/inference/tensorflow)
   (although the specifics of the exact MCMC algorithm are still fairly fluid at
   the time of writing).
1. As far as I can tell, the optimizer is still TBD.
1. Because PyMC4 relies on TensorFlow for inference, TensorFlow manages all
   gradient computation automatically.
1. PyMC4 will, like PyMC3, delegate all diagnostics and visualization
   capabilities to ArviZ.

### Pyro

Disclaimer: Pyro focuses on variational inference and normalizing flows, which
I'm not that familiar with. I may be completely off base here.

1. Pyro users write Python.
1. Pyro [relies on PyTorch
   distributions](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/__init__.py)
   ([implementing its own where
   necessary](https://github.com/pyro-ppl/pyro/tree/dev/pyro/distributions)),
   and also relies on PyTorch distributions [for its
   transforms](https://github.com/pyro-ppl/pyro/tree/dev/pyro/distributions/transforms).
1. Pyro implements [many inference
   algorithms](http://docs.pyro.ai/en/stable/inference.html) (including [HMC and
   NUTS](https://github.com/pyro-ppl/pyro/tree/dev/pyro/infer/mcmc)), but
   support for [stochastic variational inference
   (SVI)](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/svi.py) is the
   most extensive.
1. Pyro implements [its own
   optimizer](https://github.com/pyro-ppl/pyro/blob/master/pyro/optim/optim.py).
1. Pyro relies on PyTorch to compute gradients (obviously).
1. As far as I can tell, Pyro doesn't provide any diagnostic or visualization
   functionality.

