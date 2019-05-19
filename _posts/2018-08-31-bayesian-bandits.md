---
title: Multi-Armed Bandits, Conjugate Models and Bayesian Reinforcement Learning
excerpt: "In this blog post I hope to show that there is more to Bayesianism than just
MCMC sampling and suffering, by demonstrating a Bayesian approach to a classic
reinforcement learning problem: the _multi-armed bandit_."
tags:
  - bayesianism
  - pymc3
  - reinforcement learning
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background13.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2018-08-31
---

Let's talk about Bayesianism. It's developed a reputation (not entirely
justified, but not entirely unjustified either) for being too mathematically
sophisticated or too computationally intensive to work at scale. For instance,
inferring from a Gaussian mixture model is fraught with computational problems
(hierarchical funnels, multimodal posteriors, etc.), and may take a seasoned
Bayesian anywhere between a day and a month to do well. On the other hand, other
blunt hammers of estimation are as easy as a maximum likelihood estimate:
something you could easily get a SQL query to do if you wanted to.

In this blog post I hope to show that there is more to Bayesianism than just
MCMC sampling and suffering, by demonstrating a Bayesian approach to a classic
reinforcement learning problem: the _multi-armed bandit_.

The problem is this: imagine a gambler at a row of slot machines (each machine
being a “one-armed bandit”), who must devise a strategy so as to maximize
rewards. This strategy includes which machines to play, how many times to play
each machine, in which order to play them, and whether to continue with the
current machine or try a different machine.

This problem is a central problem in decision theory and reinforcement learning:
the agent (our gambler) starts out in a state of ignorance, but learns through
interacting with its environment (playing slots). For more details, Cam
Davidson-Pilon has a great introduction to multi-armed bandits in Chapter 6 of
his book [_Bayesian Methods for
Hackers_](https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter6_Priorities/Ch6_Priors_PyMC3.ipynb),
and Tor Lattimore and Csaba Szepesvári cover a breathtaking amount of the
underlying theory in their book [_Bandit Algorithms_](http://banditalgs.com/).

So let's get started! I assume that you are familiar with:

- some basic probability, at least enough to know some distributions: normal,
  Bernoulli, binomial...
- some basic Bayesian statistics, at least enough to understand what a
  [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) (and
  conjugate model) is, and why one might like them.
- [Python generators and the `yield`
  keyword](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/),
  to understand some of the code I've written[^1].

Dive in!

## The Algorithm

The algorithm is straightforward. The description below is taken from Cam
Davidson-Pilon over at Data Origami[^2].

For each round,

1. Sample a random variable $$X_b$$ from the prior of bandit $$b$$, for all
   $$b$$.
2. Select the bandit with largest sample, i.e. select bandit $$B =
   \text{argmax}(X_b)$$.
3. Observe the result of pulling bandit $$B$$, and update your prior on bandit
   $$B$$ using the conjugate model update rule.
4. Repeat!

What I find remarkable about this is how dumbfoundingly simple it is! No MCMC
sampling, no $$\hat{R}$$s to diagnose, no pesky divergences... all it requires is
a conjugate model, and the rest is literally just counting.

**NB:** This algorithm is technically known as _Thompson sampling_, and is only
one of many algorithms out there. The main difference is that there are other
ways to go from our current priors to a decision on which bandit to play
next. E.g. instead of simply sampling from our priors, we could use the
upper bound of the 90% credible region, or some dynamic quantile of the
posterior (as in Bayes UCB). See Data Origami[^2] for more information.

### Stochastic (a.k.a. stationary) bandits

Let's take this algorithm for a spin! Assume we have rewards which are Bernoulli
distributed (this would be the situation we face when e.g. modelling
click-through rates). The conjugate prior for the Bernoulli distribution is the
Beta distribution (this is a special case of the Beta-Binomial model).

<script src="https://gist.github.com/eigenfoo/3d8d318f5bd8fdea24f7b12936de77b5.js"></script>

Here, `pull` returns the result of pulling on the `arm`'th bandit, and
`make_bandits` is just a factory function for `pull`.

The `bayesian_strategy` function actually implements the algorithm. We only need
to keep track of the number of times we win and the number of times we played
(`num_rewards` and `num_trials`, respectively). It samples from all current
`np.random.beta` priors (where the original prior was a $$\text{Beta}(2,
2)$$, which is symmetrix about 0.5 and explains the odd-looking `a=2+` and
`b=2+` there), picks the `np.argmax`, `pull`s that specific bandit, and updates
`num_rewards` and `num_trials`.

I've omitted the data visualization code here, but if you want to see it, check
out the [Jupyter notebook on my
GitHub](https://github.com/eigenfoo/wanderings/blob/afcf37a8c6c2a2ac38f6708c1f3dd50db2ebe71f/bayes/bayesian-bandits.ipynb)

<figure>
    <a href="/assets/images/beta-binomial.png"><img style="float: middle" src="/assets/images/beta-binomial.png" alt="Posterior distribution after several pulls for the Beta-Binomial model"></a>
</figure>

### Generalizing to conjugate models

In fact, this algorithm isn't just limited to Bernoulli-distributed rewards: it
will work for any [conjugate
model](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)!
Here I implement the Gamma-Poisson model (that is, Poisson distributed rewards,
with a Gamma conjugate prior) to illustrate how extensible this framework is.
(Who cares about Poisson distributed rewards, you ask? Anyone who worries about
returning customers, for one!)

Here's what we need to change:

- The rewards distribution on line 5 (in practice, you don't get to pick this,
  so _technically_ there's nothing to change if you're doing this in
  production!)
- The sampling from the prior on lines 17–18
- The variables you need to keep track of and update rule on lines 12–13 and
  24–25.

Without further ado:

<script src="https://gist.github.com/eigenfoo/e9a9933d94524e6dee717276c6b6f732.js"></script>

<figure>
    <a href="/assets/images/gamma-poisson.png"><img style="float: middle" src="/assets/images/gamma-poisson.png" alt="Posterior distribution after several pulls for the Gamma-Poisson model"></a>
</figure>

This really demonstrates how lean and mean conjugate models can be, especially
considering how much of a pain MCMC or approximate inference methods would be,
compared to literal _counting_. Conjugate models aren't just textbook examples:
they're _(gasp)_ actually useful!

### Generalizing to arbitrary rewards distributions

OK, so if we have a conjugate model, we can use Thompson sampling to solve the
multi-armed bandit problem. But what if our rewards distribution doesn't have a
conjugate prior, or what if we don't even _know_ our rewards distribution?

In general this problem is very difficult to solve. Theoretically, we could
place some fairly uninformative prior on our rewards, and after every pull we
could run MCMC to get our posterior, but that doesn't scale, especially for the
online algorithms that we have in mind. Luckily a recent paper by Agrawal and
Goyal[^3] gives us some help, _if we assume rewards are bounded on the interval
$$[0, 1]$$_ (of course, if we have bounded rewards, then we can just normalize
them by their maximum value to get rewards between 0 and 1).

This solutions bootstraps the first Beta-Bernoulli model to this new situation.
Here's what happens:

1. Sample a random variable $$X_b$$ from the (Beta) prior of bandit $$b$$, for
   all $$b$$.
2. Select the bandit with largest sample, i.e. select bandit $$B =
   \text{argmax}(X_b)$$.
3. Observe the reward $$R$$ from bandit $$B$$.
4. **Observe the outcome $$r$$ from a Bernoulli trial with probability of success $$R$$.**
5. Update posterior of $$B$$ with this observation $$r$$.
6. Repeat!

Here I do this for the logit-normal distribution (i.e. a random variable whose
logit is normally distributed). Note that `np.expit` is the inverse of the logit
function.

<script src="https://gist.github.com/eigenfoo/7a397fef8aaa028c5119c9f86860d72e.js"></script>

<figure>
    <a href="/assets/images/bounded.png"><img style="float: middle" src="/assets/images/bounded.png" alt="Posterior distribution after several pulls with an arbitrary reward distribution (e.g. the logit normal)"></a>
</figure>

## Final Remarks

None of this theory is new: I'm just advertising it :blush:. See Cam
Davidson-Pilon's great blog post about Bayesian bandits[^2] for a much more
in-depth treatment, and of course, read around papers on arXiv if you want to go
deeper!

Also, if you want to see all the code that went into this blog post, check out
[the notebook
here](https://github.com/eigenfoo/wanderings/blob/afcf37a8c6c2a2ac38f6708c1f3dd50db2ebe71f/bayes/bayesian-bandits.ipynb).

---

[^1]: I've hopped on board the functional programming bandwagon, and couldn't help but think that to demonstrate this idea, I didn't need a framework, a library or even a class. Just two functions!

[^2]: Davidson-Pilon, Cameron. “Multi-Armed Bandits.” DataOrigami, 6 Apr. 2013, [dataorigami.net/blogs/napkin-folding/79031811-multi-armed-bandits](https://dataorigami.net/blogs/napkin-folding/79031811-multi-armed-bandits)

[^3]: [arXiv:1111.1797](https://arxiv.org/abs/1111.1797)
