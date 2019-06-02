---
title: Decaying Posteriors and Contextual Bandits — Bayesian Reinforcement Learning (Part 2)
excerpt:
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

In the previous blog post, I introduced multi-armed bandits, and a Bayesian
approach to modelling it (Thompson sampling). We saw that conjugate models made
it possible to run the bandit algorithm online: the same is even true for
non-conjugate models, so long as the rewards are bounded.

In this blog post, I'll take a look at two extensions to the multi-armed bandit.
The first allows the bandit to model nonstationary rewards distributions,
whereas the second allows the bandit to model context.

Dive in!

<figure>
    <a href="https://fsmedia.imgix.net/29/fd/a4/56/8363/4fb0/8c62/20e80649451b/the-multi-armed-bandit-determines-what-you-see-on-the-internet.jpeg?rect=0%2C34%2C865%2C432&auto=format%2Ccompress&dpr=2&w=650"><img src="https://fsmedia.imgix.net/29/fd/a4/56/8363/4fb0/8c62/20e80649451b/the-multi-armed-bandit-determines-what-you-see-on-the-internet.jpeg?rect=0%2C34%2C865%2C432&auto=format%2Ccompress&dpr=2&w=650" alt="Cartoon of a multi-armed bandit"></a>
    <figcaption>An example of a multi-armed bandit situation. Source: <a href="https://www.inverse.com/article/13762-how-the-multi-armed-bandit-determines-what-ads-and-stories-you-see-online">Inverse</a>.</figcaption>
</figure>

## Nonstationary Bandits

In the previous blog post, we concerned ourselves with stationary bandits: in
other words, we assumed that the rewards distribution for each arm did not
change over time. In the real world though, rewards distributions need not be
stationary: customer preferences change, trading algorithms deteriorate, and
news articles rise and fall in relevance.

Nonstationarity could mean one of two things for us:

1. either we are lucky enough to know that rewards are similarly distributed
   throughout all time (e.g. the rewards are always normally distributed, or
   always binomially distributed), and that it is merely the parameters of these
   distributions that are liable to change,
2. or we aren't so unlucky and the rewards distributions are arbitrary and
   changing.

There is a neat trick to deal with both forms of nonstationarity, which we'll
get into next!

### Decaying evidence and posteriors

But first, some notation. Suppose we have a model with parameters $$\theta$$. We
place a prior $$\color{purple}{\pi_0(\theta)}$$ on it[^1], and at $$t$$'th time
step, we observe data $$D_t$$, compute the likelihood $$\color{blue}{P(D_t |
\theta)}$$ and update the posterior from $$\color{red}{\pi_t(\theta |
D_{1:t})}$$ to $$\color{green}{\pi_{t+1}(\theta | D_{1:t+1})}$$.

This is a (very common) application of Bayes' Theorem. Explicitly, it is given
by

$$ \color{green}{\pi_{t+1}(\theta | D_{1:t+1})} \propto \color{blue}{P(D_{t+1} |
\theta)} \cdot \color{red}{\pi_t (\theta | D_{1:t})} $$

However, for problems with nonstationary rewards distributions, we would like
data points observed a long time ago to have less weight than data points
observed more recently. This is only prudent in the face of a nonstationary
rewards distribution: in the absence of recent data, we would like to adopt a
more conservative "no-data" prior, rather than allow our posterior to be
informed by outdated data. This can be achieved by modifying the Bayesian update
to:

$$ \color{green}{\pi_{t+1}(\theta | D_{1:t+1})} \propto \color{magenta}{[}
\color{blue}{P(D_{t+1} | \theta)} \cdot \color{red}{\pi_t (\theta | D_{1:t})}
{\color{magenta}{]^{1-\epsilon}}} \cdot
\color{purple}{\pi_0(\theta)}^\color{magenta}{\epsilon} $$

for some $$ 0 < \color{magenta}{\epsilon} \ll 1 $$. We can think of
$$\color{magenta}{\epsilon}$$ as controlling the rate of decay of the
evidence/posterior (i.e. how quickly we should distrust past data points).

Notice that if we stop collecting data at time $$T$$, then $$
\color{red}{\pi_t(\theta | D_{1:T})} \rightarrow \color{purple}{\pi_0(\theta)}
$$ as $$ t \rightarrow \infty $$.

Decaying the evidence (and therefore the posterior) is a nice trick that can be
used to address both types of nonstationarity identified in the previous
section: simply use the decaying update step when updating the conjugate model
hyperparameters.

For more information on decaying evidence and posteriors, check out [Austin
Rochford's talk for Boston
Bayesians](https://austinrochford.com/resources/talks/boston-bayesians-2017-bayes-bandits.slides.html#/3)
about Bayesian bandit algorithms for e-commerce.

## Contextual Bandits

We can think of the $$k$$-armed bandit problem (as presented in [my first blog
post](https://eigenfoo.xyz/bayesian-bandits/)) as follows[^2]:

1. A policy chooses an arm $$a$$ from $$k$$ arms.
2. The world reveals the reward $$R_a$$ of the chosen arm.

However, this formulation fails to capture an important phenomenon: there is
almost always extra information that is available while making each decision.
For instance, online ads occur in the context of the web page in which they
appear, and online store recommendations are given in the context of the user's
current cart contents.

To take advantage of this information, we might think of a different formulation
where, on each round:

1. The world announces some context information $$x$$.
2. A policy chooses an arm $$a$$ from $$k$$ arms.
3. The world reveals the reward $$R_a$$ of the chosen arm.

In other words, contextual bandits call for some way of taking context as input
and producing actions as output.

Alternatively, if you think of regular multi-armed bandits as taking no input
whatsoever (but still producing outputs: actions), you can think of contextual
bandits as algorithms that both take inputs and produce outputs.

### Bayesian contextual bandits

The contextual bandit problem is an extremely general framework for thinking
about sequential decision making (or reinforcement learning). Clearly, there are
many ways to make a bandit algorithm take context into account: linear
regression is a classic example: we assume that the rewards, $$y$$, are a linear
function of the context, $$z$$.

If you're shaky on the details of Bayesian linear regression, refer to [_Pattern
Recognition and Machine
Learning_](https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
by Christopher Bishop (specifically, section 3.3 on Bayesian linear regression
and exercises 3.12 and 3.13[^3]). Briefly though, if we place a Gaussian prior
on the regression weights and an inverse gamma prior on the noise parameter
(that is, the noise of the observations), then these priors will be conjugate to
a Gaussian likelihood, and the posterior predictive distribution for the rewards
will be a Student's t.

Since we need to maintain posteriors of the rewards for each arm (so that we can
do Thompson sampling), we need to run a separate Bayesian linear regression for
each arm. At every iteration we then Thompson sample from each Student's t
posterior, and choose the arm with the highest sample.

Of course, linear regression is the textbook example of a model that offers
interpretability but lacks expressiveness. In most circumstances we would want
to be able to model nonlinear functions as well. One (perfectly valid) way of
doing this would be to hand-engineer some features and/or some basis functions
before feeding that into our Bayesian linear regression. However, in the 21st
century, the trendier thing to do is to have a neural network learn those
features for you. This is exactly what is proposed in a [2018 paper by Google
Brain](https://arxiv.org/pdf/1802.09127.pdf). They find that this setup - which
they dub NeuralLinear - works well across

However, I would criticize their characterization of the NeuralLinear algorithm:
if offers principled and interpretable uncertainty estimates for a
_non-interpretable_ representation learnt by a neural network.

For more information, be sure to check out the [Google Brain
paper](https://arxiv.org/pdf/1802.09127.pdf) and the [TensorFlow code that
accompanies
it](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits).

## Further Reading

- As always, the [Wikipedia page for contextual
  bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit#Contextual_bandit)
  is always a good place to start reading.

  https://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions_for_contextual_bandit

- For non-Bayesian approaches to contextual bandits, [Vowpal
  Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms)
  is a great resource: [John Langford](http://hunch.net/~jl/) and the team at
  [Microsoft Research](https://www.microsoft.com/research/) has [extensively
  researched](https://arxiv.org/abs/1402.0555v2) contextual bandit algorithms.
  They've provided both blazingly fast implementations of recent algorithms, and
  written good documentation for them.

---

[^1]: Did you know you can make [colored equations with MathJax](http://adereth.github.io/blog/2013/11/29/colorful-equations/)? Technology frightens me sometimes.

[^2]: This explanation is largely drawn from [from John Langford's `hunch.net`](http://hunch.net/?p=298).

[^3]: If you don't want to do Bishop's exercises, the answers are given in equations 1 and 2 of [the Google Brain paper](https://arxiv.org/abs/1802.09127) :wink:.
