---
title: Bayesian Inference — A Reading List
excerpt:
tags:
  - bayesianism
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background7.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2018-12-15
toc: true
toc_sticky: true
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
```

A lot of people's first steps in data science start with these five packages.
It's an excellent starting point: with five import statements, you already have
a lot of machine learning firepower available to you, and if you achieve mastery
(nay, even just fluency) in these five libraries you've already amassed a
formidable skill set.

So imagine the surprise when you

This reading list isn't focused on how to use Bayesian modelling for a
_specific_ use case[^1]. It’s focused on how Bayesian inference works _in
general_: this is for people who want to really understand the engines and
workhorses of Bayesian inference, who are curious what actually happens when you
run an MCMC sampler.

## Markov-Chain Monte Carlo

### For the uninitiated

1. [MCMC Sampling for
   Dummies](https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/) by Thomas
   Wiecki. A basic introduction to MCMC with accompanying Python snippets. The
   Metropolis sampler is used an introduction to sampling.
2. [Introduction to Markov Chain Monte
   Carlo](http://www.mcmchandbook.net/HandbookChapter1.pdf) by Charles Geyer.

### Hamiltonian Monte Carlo and the No-U-Turn Sampler

1. [Hamiltonian Monte Carlo
   explained](https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html).
   A visual and intuitive explanation of HMC.
2. [A Conceptual Introduction to Hamiltonian Monte
   Carlo](https://arxiv.org/abs/1701.02434) by Michael Betancourt. An excellent
   paper to read for a solid conceptual understanding and principled intuition
   for HMC.
3. [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
   Carlo](https://arxiv.org/abs/1111.4246) by Matthew Hoffman and Andrew Gelman.
   The original NUTS paper.
4. [MCMC Using Hamiltonian
   Dynamics](http://www.mcmchandbook.net/HandbookChapter5.pdf) by Radford Neal.
5. [Hamiltonian Monte Carlo in
   PyMC3](https://colindcarroll.com/talk/hamiltonian-monte-carlo/) by Colin
   Carroll.

### Sequential Monte Carlo and particle filters

1. [An Introdution to Sequential Monte Carlo
   Methods](https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf)
   by Arnaud Doucet, Nando de Freitas and Neil Gordon. A long but detailed
   introduction.
2. [Sequential Monte Carlo Methods & Particle Filters
   Resources](http://www.stats.ox.ac.uk/~doucet/smc_resources.html) by Arnaud
   Doucet. A list of resources on SMC and particle filters: way more than you
   probably ever need to know about them.

### Other sampling methods

1. Chapter 11 (Sampling Methods) of [Pattern Recognition and Machine
   Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
   by Christopher Bishop. Covers rejection, importance, Metropolis-Hastings,
   Gibbs and slice sampling.
2. [The Markov-chain Monte Carlo Interactive
   Gallery](https://chi-feng.github.io/mcmc-demo/) by Chi Feng. A fantastic
   library of visualizations of various MCMC samplers.

## Variational Inference

### For the uninitiated

1. [Deriving
   Expectation-Maximization](http://willwolf.io/2018/11/11/em-for-lda/) by Will
   Wolf. The first blog post in a series that builds from EM all the way to VI.
   Also check out [Deriving Mean-Field Variational
   Bayes](http://willwolf.io/2018/11/23/mean-field-variational-bayes/).
2. [Variational Inference: A Review for
   Statisticians](https://arxiv.org/abs/1601.00670) by David Blei, Alp
   Kucukelbir and Jon McAuliffe. A review of variational inference (ostensibly
   for statisticians, but I don't see why non-statisticians can't read it).
3. Chapter 10 (Approximate Inference) of [Pattern Recognition and Machine
   Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
   by Christopher Bishop.

### Automatic differentiation variational inference (ADVI)

1. [Automatic Differentiation Variational
   Inference](https://arxiv.org/abs/1603.00788) by Alp Kucukelbir, Dustin Tran
   et al. The original ADVI paper.

### Operator variational inference (OPVI)

1. [Operator Variational Inference](https://arxiv.org/abs/1610.09033) by Rajesh
   Ranganath, Jaan Altosaar, Dustin Tran and David Blei. The original OPVI
   paper.

## Software for Bayesian Modelling and Inference

There are many open-source software libraries for Bayesian modelling and
inference, and it is instructive to look into the inference methods that they do
(or do not!) implement.

1. [Stan](http://mc-stan.org/)
2. [PyMC3](http://docs.pymc.io/)
3. [Pyro](http://pyro.ai/)
4. [Tensorflow Probability](https://www.tensorflow.org/probability/)
5. [Edward](http://edwardlib.org/)
6. [Greta](https://greta-stats.org/)
7. [Infer.NET](https://dotnet.github.io/infer/)
8. [BUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/)
9. [JAGS](http://mcmc-jags.sourceforge.net/)

## Further Topics

Bayesian inference doesn't stop at MCMC and VI: there is bleeding-edge research
being done on other methods of inference. While they aren't ready for real-world
use, it is interesting to see what they are.

### Approximate Bayesian computation and likelihood-free methods

1. [Likelihood-free Monte Carlo](https://arxiv.org/abs/1001.2058) by Scott
   Sisson and Yanan Fan.

### Expectation propagation

1. [Expectation propagation as a way of life: A framework for Bayesian inference
   on partitioned data](https://arxiv.org/abs/1412.4869) by Aki Vehtari, Andrew
   Gelman, et al.

---

[^1]: If that’s what you’re looking for, check out my [Bayesian modelling cookbook](https://eigenfoo.xyz/bayesian-modelling-cookbook) or [Michael Betancourt’s excellent essay on a principles Bayesian workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html).
