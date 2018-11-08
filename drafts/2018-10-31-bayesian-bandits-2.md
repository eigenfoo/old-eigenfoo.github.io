---
title: "Decaying Posteriors, Contextual Bandits and Bayesian Reinforcement
Learning (Part II)"
excerpt:
tags:
  - bayesianism
  - reinforcement learning
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background1.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2018-10-31
---

This is a continuation of the [first blog post on Bayesian
bandits](https://eigenfoo.xyz/bayesian-bandits/).

We introduce two extensions of the vanilla Bayesian bandit with Thompson
sampling: one to deal with nonstationary rewards, and one to deal with extra
contextual information.

## Decayed Posteriors

Previously we assumed that our rewards distribution was stationary: that is,
we might not know the average payout of each bandit, or how consistently it
delivers these payouts, but at least we know that it doesn't change in time. In
general however, this is not the case. For instance, if we were advertisers or
marketers, we know that users' preferences don't stay the same: they change in
time, possibly even with some seasonality.

Bayes' Theorem:

$$ P(\theta | D_{t+1}, D_{1:t}) \propto L(D_{t+1} | \theta) \cdot P(\theta | D_{1:t}) $$

where $\theta$ are our parameters,

With a decayed posterior, we have:

$$ \tilde{P}(\theta | D_{t+1}, D_{1:t}) \propto \left( L(D_{t+1} | \theta) \cdot \tilde{P}(\theta | D_{1:t}) \right)^{1-\epsilon} \cdot P_0{(\theta)}^{\epsilon}$$

where typically $0 < \epsilon << 1$

If we use a conjugate prior, then the decayed posterior is also conjugate.

https://austinrochford.com/resources/talks/boston-bayesians-2017-bayes-bandits.slides.html#/3/3


## Bayesian Contextual Bandits

https://people.orie.cornell.edu/pfrazier/Presentations/2012.10.INFORMS.Bandit.pdf


