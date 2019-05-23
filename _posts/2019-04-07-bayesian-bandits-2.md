---
title: Decaying Posteriors and Contextual Bandits — Bayesian Reinforcement Learning (Part 2)
excerpt:
tags:
  - bayesian
  - reinforcement learning
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background14.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2019-06-23
---

> This is the second of a two-part series about Bayesian bandit algorithms.
> Check out the first post [here](https://eigenfoo.xyz/bayesian-bandits/).

This will be a more theoretical blog post, outlining the theoretical variations
on the Bayesian bandit algorithm.

![Cartoon of a multi-armed bandit](https://cdn-images-1.medium.com/max/1200/1*FH4t-DcuKWfLYRWvd4JIjA.png)

## Nonstationary Bandits

In the previous blog post, we concerned ourselves with stationary bandits: in
other words, we assumed that the rewards distribution did not change over time.
In the real world though, customer preferences change.

### Decaying posteriors

Thankfully, there is a nice solution to deal with

First, some notation. Suppose we have a model with parameters $$\theta$$. We
place a prior $$\pi_0(\theta)$$ on it, and, at $$t$$th time step, we observe
data $$D_t$$, compute the likelihood $$P(D_t | \theta)$$ and update the
posterior to $$\pi_t(\theta)$$.

This is the (ubiquitous) Bayesian update rule, which is given by

$$ \pi_{t+1}(\theta | D_{1:t+1}) \propto P(D_{t+1} | \theta) \cdot \pi_t (\theta | D_{1:t}) $$

However, for problems with nonstationary rewards distributions, we would like
data points observed a long time ago to have less weight than more recently
observed data points. This is only prudent in the face of a nonstationary
rewards distribution: after all, the rewards distribution may have changed
significantly between when we observed those data points and the current time.
We would rather abandon most (if not all) of our data in favor of a more
conservative posterior.

This can be achieved by making our posterior "forget" old data points.

For $$ 0 < \epsilon << 1 $$,

$$ \pi_{t+1}(\theta | D_{1:t+1}) \propto [ P(D_{t+1} | \theta) \cdot \pi_t (\theta | D_{1:t}) ]^{1-\epsilon} \cdot \pi_0(\theta)^\epsilon $$

If we stop collecting data at time $$T$$, then

$$ \pi_t(\theta | D_{1:T}) \rightarrow \pi_0(\theta) $$

- https://austinrochford.com/resources/talks/boston-bayesians-2017-bayes-bandits.slides.html#/3

## Contextual Bandits

### What even is a contextual bandit?

We first need to know what contextual bandits are, let alone Bayesian contextual
bandits. The most succinct explanation I've come across is [from John Langford's
`hunch.net`](http://hunch.net/?p=298).

Briefly, we can think of the $$k$$-armed bandit problem (as presented in [my
first blog post](https://eigenfoo.xyz/bayesian-bandits/)) as follows:

1. A policy chooses an arm $$a$$ from $$k$$ arms.
2. The world reveals the reward $$R_a$$ of the chosen arm.

However, this formulation fails to capture an important phenomenon: there is
almost always extra information that is available while making each decision.
To take advantage of this information, we might think of a different formulation
where, on each round:

1. The world announces some context information $$x$$.
2. A policy chooses an arm $$a$$ from $$k$$ arms.
3. The world reveals the reward $$R_a$$ of the chosen arm.

### Bayesian contextual bandits

- https://people.orie.cornell.edu/pfrazier/Presentations/2012.10.INFORMS.Bandit.pdf

## References

- https://en.wikipedia.org/wiki/Multi-armed_bandit
- https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
- https://github.com/david-cortes/contextualbandits
- https://arxiv.org/abs/1802.04064
