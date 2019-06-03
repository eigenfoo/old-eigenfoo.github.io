---
title: Decaying Posteriors and Contextual Bandits — Bayesian Reinforcement Learning (Part 2)
excerpt: "In this blog post, we'll take a look at two extensions to the
multi-armed bandit.  The first allows the bandit to model nonstationary rewards
distributions, whereas the second allows the bandit to model context."
tags:
  - bayesianism
  - reinforcement learning
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background14.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2019-06-02
---

> This is the second of a two-part series about Bayesian bandit algorithms.
> Check out the first post [here](https://eigenfoo.xyz/bayesian-bandits/).

[Previously](https://eigenfoo.xyz/bayesian-bandits/), I introduced the
multi-armed bandit problem, and a Bayesian approach to solving/modelling it
(Thompson sampling). We saw that conjugate models made it possible to run the
bandit algorithm online: the same is even true for non-conjugate models, so long
as the rewards are bounded.

In this follow-up blog post, we'll take a look at two extensions to the
multi-armed bandit. The first allows the bandit to model nonstationary rewards
distributions, whereas the second allows the bandit to model context. Jump in!

<figure>
    <a href="https://fsmedia.imgix.net/29/fd/a4/56/8363/4fb0/8c62/20e80649451b/the-multi-armed-bandit-determines-what-you-see-on-the-internet.jpeg?rect=0%2C34%2C865%2C432&auto=format%2Ccompress&dpr=2&w=650"><img src="https://fsmedia.imgix.net/29/fd/a4/56/8363/4fb0/8c62/20e80649451b/the-multi-armed-bandit-determines-what-you-see-on-the-internet.jpeg?rect=0%2C34%2C865%2C432&auto=format%2Ccompress&dpr=2&w=650" alt="Cartoon of a multi-armed bandit"></a>
    <figcaption>An example of a multi-armed bandit situation. Source: <a href="https://www.inverse.com/article/13762-how-the-multi-armed-bandit-determines-what-ads-and-stories-you-see-online">Inverse</a>.</figcaption>
</figure>

## Nonstationary Bandits

Up until now, we've concerned ourselves with stationary bandits: in other words,
we assumed that the rewards distribution for each arm did not change over time.
In the real world though, rewards distributions need not be stationary: customer
preferences change, trading algorithms deteriorate, and news articles rise and
fall in relevance.

Nonstationarity could mean one of two things for us:

1. either we are lucky enough to know that rewards are similarly distributed
   throughout all time (e.g. the rewards are always normally distributed, or
   always binomially distributed), and that it is merely the parameters of these
   distributions that are liable to change,
2. or we aren't so unlucky, and the rewards distributions are not only changing,
   but don't even have a nice parametric form.

Good news, though: there is a neat trick to deal with both forms of
nonstationarity!

### Decaying evidence and posteriors

But first, some notation. Suppose we have a model with parameters $$\theta$$. We
place a prior $$\color{purple}{\pi_0(\theta)}$$ on it[^1], and at the $$t$$'th
time step, we observe data $$D_t$$, compute the likelihood $$\color{blue}{P(D_t
| \theta)}$$ and update the posterior from $$\color{red}{\pi_t(\theta |
D_{1:t})}$$ to $$\color{green}{\pi_{t+1}(\theta | D_{1:t+1})}$$.

This is a quintessential application of Bayes' Theorem. Mathematically:

$$ \color{green}{\pi_{t+1}(\theta | D_{1:t+1})} \propto \color{blue}{P(D_{t+1} |
\theta)} \cdot \color{red}{\pi_t (\theta | D_{1:t})} \tag{1} \label{1} $$

However, for problems with nonstationary rewards distributions, we would like
data points observed a long time ago to have less weight than data points
observed recently. This is only prudent: in the absence of recent data, we would
like to adopt a more conservative "no-data" prior, rather than allow our
posterior to be informed by outdated data. This can be achieved by modifying the
Bayesian update to:

$$ \color{green}{\pi_{t+1}(\theta | D_{1:t+1})} \propto \color{magenta}{[}
\color{blue}{P(D_{t+1} | \theta)} \cdot \color{red}{\pi_t (\theta | D_{1:t})}
{\color{magenta}{]^{1-\epsilon}}} \cdot
\color{purple}{\pi_0(\theta)}^\color{magenta}{\epsilon} \tag{2} \label{2} $$

for some $$ 0 < \color{magenta}{\epsilon} \ll 1 $$. We can think of
$$\color{magenta}{\epsilon}$$ as controlling the rate of decay of the
evidence/posterior (i.e. how quickly we should distrust past data points).
Notice that if we stop observing data points at time $$T$$, then $$
\color{red}{\pi_t(\theta | D_{1:T})} \rightarrow \color{purple}{\pi_0(\theta)}
$$ as $$ t \rightarrow \infty $$.

Decaying the evidence (and therefore the posterior) can be used to address both
types of nonstationarity identified above. Simply use $$(\ref{2})$$ as a drop-in
replacement for $$(\ref{1})$$ when updating the hyperparameters. Whether you're
using a conjugate model or the algorithm by [Agarwal and
Goyal](https://arxiv.org/abs/1111.1797) (introduced in [the previous blog
post](https://eigenfoo.xyz/bayesian-bandits)), using $$(\ref{2})$$ will decay
the evidence and posterior, as desired.

For more information (and a worked example for the Beta-Binomial model!), check
out [Austin Rochford's talk for Boston
Bayesians](https://austinrochford.com/resources/talks/boston-bayesians-2017-bayes-bandits.slides.html#/3)
about Bayesian bandit algorithms for e-commerce.

## Contextual Bandits

We can think of the multi-armed bandit problem as follows[^2]:

1. A policy chooses an arm $$a$$ from $$k$$ arms.
2. The world reveals the reward $$R_a$$ of the chosen arm.

However, this formulation fails to capture an important phenomenon: there is
almost always extra information that is available when making each decision.
For instance, online ads occur in the context of the web page in which they
appear, and online store recommendations are given in the context of the user's
current cart contents (among other things).

To take advantage of this information, we might think of a different formulation
where, on each round:

1. The world announces some context information $$x$$.
2. A policy chooses an arm $$a$$ from $$k$$ arms.
3. The world reveals the reward $$R_a$$ of the chosen arm.

In other words, contextual bandits call for some way of taking context as input
and producing arms/actions as output.

Alternatively, if you think of regular multi-armed bandits as taking no input
whatsoever (but still producing outputs, the arms to pull), you can think of
contextual bandits as algorithms that both take inputs and produce outputs.

### Bayesian contextual bandits

Contextual bandits give us a very general framework for thinking about
sequential decision making (and reinforcement learning). Clearly, there are many
ways to make a bandit algorithm take context into account. Linear regression is
a straightforward and classic example: simply assume that the rewards depend
linearly on the context.

For a refresher on the details of Bayesian linear regression, refer to [_Pattern
Recognition and Machine
Learning_](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
by Christopher Bishop: specifically, section 3.3 on Bayesian linear regression
and exercises 3.12 and 3.13[^3]. Briefly though, if we place a Gaussian prior on
the regression weights and an inverse gamma prior on the noise parameter (i.e.,
the noise of the observations), then their joint prior will be conjugate to a
Gaussian likelihood, and the posterior predictive distribution for the rewards
will be a Student's $$t$$.

Since we need to maintain posteriors of the rewards for each arm (so that we can
do Thompson sampling), we need to run a separate Bayesian linear regression for
each arm. At every iteration we then Thompson sample from each Student's $$t$$
posterior, and select the arm with the highest sample.

However, Bayesian linear regression is a textbook example of a model that lacks
expressiveness: in most circumstances, we want something that can model
nonlinear functions as well. One (perfectly valid) way of doing this would be to
hand-engineer some nonlinear features and/or basis functions before feeding them
into a Bayesian linear regression. However, in the 21st century, the trendier
thing to do is to have a neural network learn those features for you. This is
exactly what is proposed in a [ICLR 2018 paper from Google
Brain](https://arxiv.org/abs/1802.09127). They find that this model — which they
call `NeuralLinear` — performs decently well across a variety of tasks, even
compared to other bandit algorithms. In the words of the authors:

> We believe [`NeuralLinear`'s] main strength is that it is able to
> _simultaneously_ learn a data representation that greatly simplifies the task
> at hand, and to accurately quantify the uncertainty over linear models that
> explain the observed rewards in terms of the proposed representation.

For more information, be sure to check out the [Google Brain
paper](https://arxiv.org/abs/1802.09127) and the accompanying [TensorFlow
code](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits).

## Further Reading

For non-Bayesian approaches to contextual bandits, [Vowpal
Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms)
is a great resource: [John Langford](http://hunch.net/~jl/) and the team at
[Microsoft Research](https://www.microsoft.com/research/) has [extensively
researched](https://arxiv.org/abs/1402.0555v2) contextual bandit algorithms.
They've provided blazingly fast implementations of recent algorithms and written
good documentation for them.

For the theory and math behind bandit algorithms, [Tor Lattimore and Csaba
Szepesvári's book](https://banditalgs.com/) covers a breathtaking amount of
ground.

> This is the second of a two-part series about Bayesian bandit algorithms.
> Check out the first post [here](https://eigenfoo.xyz/bayesian-bandits/).

---

[^1]: Did you know you can make [colored equations with MathJax](http://adereth.github.io/blog/2013/11/29/colorful-equations/)? Technology frightens me sometimes.

[^2]: This explanation is largely drawn from [from John Langford's `hunch.net`](http://hunch.net/?p=298).

[^3]: If you don't want to do Bishop's exercises, there's a partially complete solutions manual [on GitHub](https://github.com/GoldenCheese/PRML-Solution-Manual/) :wink:
