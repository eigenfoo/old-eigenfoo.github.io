---
title: Modern Bayesian Inference — A Reading List
excerpt:
tags:
  - bayesianism
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background6.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2018-12-15
---

<blockquote class="twitter-tweet" data-lang="en"><p lang="und" dir="ltr">▓▓▓▓▓▓▓▓▓▓▓▓▓▓░ 95%</p>&mdash; Year Progress (@year_progress) <a href="https://twitter.com/year_progress/status/1073276365720436736?ref_src=twsrc%5Etfw">December 13, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Less than 5% of 2018 left to go! This past year I’ve taken a deep dive into
Bayesian modelling and inference. It’s an incredibly deep field, but it can be
overwhelming for a newcomer.

This reading list isn’t focused on how to use Bayesian modelling for a
_specific_ use case[^1]. It’s focused on how Bayesian inference works _in
general_: this is for people who want to really understand the engines and
workhorses of Bayesian inference, who are curious what actually happens when you
run an MCMC sampler.

## Markov-Chain Monte Carlo

### For the uninitiated

1. [MCMC Sampling for
   Dummies](https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/) by Thomas
   Wiecki
2. [Introduction to Markov Chain Monte
   Carlo](http://www.mcmchandbook.net/HandbookChapter1.pdf) by Charles Geyer

### Hamiltonian Monte Carlo and the No-U-Turn Sampler

1. [Hamiltonian Monte Carlo
   explained](https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html)
2. [A Conceptual Introduction to Hamiltonian Monte
   Carlo](https://arxiv.org/abs/1701.02434) by Michael Betancourt
3. [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
   Carlo](https://arxiv.org/abs/1111.4246) by Matthew Hoffman and Andrew Gelman 
4. [MCMC Using Hamiltonian
   Dynamics](http://www.mcmchandbook.net/HandbookChapter5.pdf) by Radford Neal
5. [Hamiltonian Monte Carlo in
   PyMC3](https://colindcarroll.com/talk/hamiltonian-monte-carlo/) by Colin
   Carroll

### Sequential Monte Carlo and particle filters

1. [An Introdution to Sequential Monte Carlo
   Methods](https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf)
   by Arnaud Doucet, Nando de Freitas and Neil Gordon
2. [Sequential Monte Carlo Methods & Particle Filters
   Resources](http://www.stats.ox.ac.uk/~doucet/smc_resources.html) by Arnaud
   Doucet

### Other sampling methods

1. Chapter 11 (Sampling Methods) of [Pattern Recognition and Machine
   Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
   by Christopher Bishop
2. [The Markov-chain Monte Carlo Interactive
   Gallery](https://chi-feng.github.io/mcmc-demo/) by Chi Feng

## Variational Inference

### For the uninitiated

1. [Variational Inference: A Review for
   Statisticians](https://arxiv.org/abs/1601.00670) by David Blei, Alp
   Kucukelbir and Jon McAuliffe 
2. Chapter 10 (Approximate Inference) of [Pattern Recognition and Machine
   Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
   by Christopher Bishop

### Automatic differentiation variational inference

1. [Automatic Differentiation Variational
   Inference](https://arxiv.org/abs/1603.00788) by Alp Kucukelbir, Dustin Tran,
   Rajesh Ranganath, Andrew Gelman and David Blei

## Open-Source Packages for Bayesian Modelling/Inference

1. [Stan](http://mc-stan.org/)
2. [PyMC3](http://docs.pymc.io)
3. [Greta](https://greta-stats.org/)
4. [Pyro](http://pyro.ai)
5. [Edward](http://edwardlib.org/)
6. [Tensorflow Probability](https://www.tensorflow.org/probability/)
7. [Infer.NET](https://dotnet.github.io/infer/)
8. [BUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/)
9. [JAGS](http://mcmc-jags.sourceforge.net/)

## Further Topics

### Approximate Bayesian computation and likelihood-free methods

1. [Likelihood-free Monte Carlo](https://arxiv.org/abs/1001.2058) by Scott
   Sisson and Yanan Fan

### Expectation propagation

1. [Expectation propagation as a way of life: A framework for Bayesian inference
   on partitioned data](https://arxiv.org/abs/1412.4869)

---

[^1] If that’s what you’re looking for, check out my [Bayesian modelling cookbook](https://eigenfoo.xyz/bayesian-modelling-cookbook) or [Michael Betancourt’s excellent essay on a principles Bayesian workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html).
