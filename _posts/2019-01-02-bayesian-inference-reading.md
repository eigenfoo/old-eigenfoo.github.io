---
title: "Modern Computational Methods for Bayesian Inference — A Reading List"
excerpt: "An annotated reading list on modern computational methods for Bayesian
inference — Markov chain Monte Carlo (MCMC), variational inference (VI) and
some other (more experimental) methods."
tags:
  - bayesianism
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background12.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2019-01-02
toc: true
toc_sticky: true
---

Lately I've been troubled by how little I actually knew about how Bayesian
inference _really worked_. I could explain to you [many other machine learning
techniques](https://maria-antoniak.github.io/2018/11/19/data-science-crash-course.html),
but with Bayesian modelling... well, there's a model (which is basically the
likelihood, I think?), and then there's a prior, and then, um...

What actually happens when you run a sampler? What makes inference
"variational"? And what is this automatic differentiation doing in my
variational inference? _Cue long sleepless nights, contemplating my own
ignorance._

So to celebrate the new year[^1], I compiled a list of things to read — blog
posts, journal papers, books, anything that would help me understand (or at
least, appreciate) the math and computation that happens when I press the _Magic
Inference Button™_. Again, this reading list isn't focused on how to use
Bayesian modelling for a _specific_ use case[^2]; it’s focused on how modern
computational methods for Bayesian inference work _in general_.

So without further ado...

## Markov-Chain Monte Carlo

### For the uninitiated

1. [MCMC Sampling for
   Dummies](https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/) by Thomas
   Wiecki. A basic introduction to MCMC with accompanying Python snippets. The
   Metropolis sampler is used an introduction to sampling.
1. [Introduction to Markov Chain Monte
   Carlo](http://www.mcmchandbook.net/HandbookChapter1.pdf) by Charles Geyer.
   The first chapter of the aptly-named [_Handbook of Markov Chain Monte
   Carlo_](http://www.mcmchandbook.net/).

### Hamiltonian Monte Carlo and the No-U-Turn Sampler

1. [Hamiltonian Monte Carlo
   explained](https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html).
   A visual and intuitive explanation of HMC: great for starters.
1. [A Conceptual Introduction to Hamiltonian Monte
   Carlo](https://arxiv.org/abs/1701.02434) by Michael Betancourt. An excellent
   paper for a solid conceptual understanding and principled intuition for HMC.
1. [Exercises in Automatic Differentiation using `autograd` and
   `jax`](https://colindcarroll.com/2019/04/06/exercises-in-automatic-differentiation-using-autograd-and-jax/)
   by Colin Carroll. This is the first in a series of blog posts that explain
   HMC from the very beginning. See also [Hamiltonian Monte Carlo from
   Scratch](https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/),
   [Step Size Adaptation in Hamiltonian Monte
   Carlo](https://colindcarroll.com/2019/04/21/step-size-adaptation-in-hamiltonian-monte-carlo/),
   and [Choice of Symplectic Integrator in Hamiltonian Monte
   Carlo](https://colindcarroll.com/2019/04/28/choice-of-symplectic-integrator-in-hamiltonian-monte-carlo/).
1. [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
   Carlo](https://arxiv.org/abs/1111.4246) by Matthew Hoffman and Andrew Gelman.
   The original NUTS paper.
1. [MCMC Using Hamiltonian
   Dynamics](http://www.mcmchandbook.net/HandbookChapter5.pdf) by Radford Neal.
1. [Hamiltonian Monte Carlo in
   PyMC3](https://colindcarroll.com/talk/hamiltonian-monte-carlo/) by Colin
   Carroll.

### Sequential Monte Carlo and particle filters

1. [An Introdution to Sequential Monte Carlo
   Methods](https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf)
   by Arnaud Doucet, Nando de Freitas and Neil Gordon. This chapter from [the
   authors' textbook on SMC](https://www.springer.com/us/book/9780387951461)
   provides motivation for using SMC methods, and gives a brief introduction to
   a basic particle filter.
1. [Sequential Monte Carlo Methods & Particle Filters
   Resources](http://www.stats.ox.ac.uk/~doucet/smc_resources.html) by Arnaud
   Doucet. A list of resources on SMC and particle filters: way more than you
   probably ever need to know about them.

### Other sampling methods

1. Chapter 11 (Sampling Methods) of [Pattern Recognition and Machine
   Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
   by Christopher Bishop. Covers rejection, importance, Metropolis-Hastings,
   Gibbs and slice sampling. Perhaps not as rampantly useful as NUTS, but good
   to know nevertheless.
1. [The Markov-chain Monte Carlo Interactive
   Gallery](https://chi-feng.github.io/mcmc-demo/) by Chi Feng. A fantastic
   library of visualizations of various MCMC samplers.

## Variational Inference

### For the uninitiated

1. [Deriving
   Expectation-Maximization](http://willwolf.io/2018/11/11/em-for-lda/) by Will
   Wolf. The first blog post in a series that builds from EM all the way to VI.
   Also check out [Deriving Mean-Field Variational
   Bayes](http://willwolf.io/2018/11/23/mean-field-variational-bayes/).
1. [Variational Inference: A Review for
   Statisticians](https://arxiv.org/abs/1601.00670) by David Blei, Alp
   Kucukelbir and Jon McAuliffe. An high-level overview of variational
   inference: the authors go over one example (performing VI on GMMs) in depth.
1. Chapter 10 (Approximate Inference) of [Pattern Recognition and Machine
   Learning](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
   by Christopher Bishop.

### Automatic differentiation variational inference (ADVI)

1. [Automatic Differentiation Variational
   Inference](https://arxiv.org/abs/1603.00788) by Alp Kucukelbir, Dustin Tran
   et al. The original ADVI paper.
1. [Automatic Variational Inference in
   Stan](https://papers.nips.cc/paper/5758-automatic-variational-inference-in-stan)
   by Alp Kucukelbir, Rajesh Ranganath, Andrew Gelman and David Blei.

## Open-Source Software for Bayesian Inference

There are many open-source software libraries for Bayesian modelling and
inference, and it is instructive to look into the inference methods that they do
(or do not!) implement.

1. [Stan](http://mc-stan.org/)
1. [PyMC3](http://docs.pymc.io/)
1. [Pyro](http://pyro.ai/)
1. [Tensorflow Probability](https://www.tensorflow.org/probability/)
1. [Edward](http://edwardlib.org/)
1. [Greta](https://greta-stats.org/)
1. [Infer.NET](https://dotnet.github.io/infer/)
1. [BUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/)
1. [JAGS](http://mcmc-jags.sourceforge.net/)

## Further Topics

Bayesian inference doesn't stop at MCMC and VI: there is bleeding-edge research
being done on other methods of inference. While they aren't ready for real-world
use, it is interesting to see what they are.

### Approximate Bayesian computation (ABC) and likelihood-free methods

1. [Likelihood-free Monte Carlo](https://arxiv.org/abs/1001.2058) by Scott
   Sisson and Yanan Fan.

### Expectation propagation

1. [Expectation propagation as a way of life: A framework for Bayesian inference
   on partitioned data](https://arxiv.org/abs/1412.4869) by Aki Vehtari, Andrew
   Gelman, et al.

### Operator variational inference (OPVI)

1. [Operator Variational Inference](https://arxiv.org/abs/1610.09033) by Rajesh
   Ranganath, Jaan Altosaar, Dustin Tran and David Blei. The original OPVI
   paper.

(I've tried to include as many relevant and helpful resources as I could find,
but if you feel like I've missed something, [drop me a
line](https://twitter.com/@_eigenfoo)!)

---

[^1]: [Relevant tweet
      here.](https://twitter.com/year_progress/status/1079889949871300608)

[^2]: If that’s what you’re looking for, check out my [Bayesian modelling
      cookbook](https://eigenfoo.xyz/bayesian-modelling-cookbook) or [Michael
      Betancourt’s excellent essay on a principles Bayesian
      workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html).

