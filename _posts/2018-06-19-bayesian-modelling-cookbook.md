--
title: Cookbook - Bayesian Modeling with PyMC3
excerpt:
tags:
  - bayesian
  - mcmc
  - pymc3
  - cookbook
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background5.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
toc: true
toc_label: "Recipes"
toc_icon: "utensils"
mathjax: true
last_modified_at: 2018-06-19
---

Recently I've started using [PyMC3](https://github.com/pymc-devs/pymc3) for
Bayesian modeling, and it's an absolute treat! The API only exposes as much of
heavy machinery of MCMC as you need - by which I mean, `pm.sample()` (a.k.a., as
[Thomas Wiecki](http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/) puts
it, the _Magic Inference Button™_). This really frees up your mind to think
about the data in a probabilistic/generative/"Bayesian" way!

That being said however, I quickly realized that the water gets very deep very
quickly: I thought about and explored the data, I specified a hierarchical model
that made sense to me, I hit the _Magic Inference Button™_, and... uh, what now?
I blinked at the red warnings the sampler spat out.

This is a compilation of notes, tips, tricks and recipes that I've collected
from everywhere: papers, documentation, asking questions to my [more
experienced](https://twitter.com/twiecki)
[coworkers](https://twitter.com/aseyboldt). It's still very much a work in
progress, but hopefully somebody else finds it useful!

## For the Uninitiated

- First of all, *welcome!* It's a brave new world out there - where statistics
  is cool, Bayesian and (if you're lucky) even easy. Dive in!

### Bayesian Modeling

- For an introduction to general Bayesian methods and modeling, I really liked
  [Cam Davidson Pilon's _Bayesian Methods for
  Hackers_](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/):
  it really made the whole "thinking like a Bayesian" thing click for me.

- If you're willing to spend some money, I've heard that [_Doing Bayesian Data
  Analysis_ by
  Kruschke](https://sites.google.com/site/doingbayesiandataanalysis/) (a.k.a.
  _"the puppy book"_) is for the bucket list.

### Markov Chain Monte Carlo

- For a good high-level introduction to MCMC, I liked [Michael Betancourt's
  StanCon 2017
  talk](https://www.youtube.com/watch?v=DJ0c7Bm5Djk&feature=youtu.be&t=4h40m9s):
  especially the first few minutes where he provides a motivation for MCMC, that
  really put all this math into context for me.

- For a more in-depth (and also mathematical) treatment of MCMC (and especially
  why the Metropolis-Hastings and Gibbs samplers suck), I'd check out his [paper
  on Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434).

## Formulating the Model

- Try thinking about _how_ your data would be generated: what kind of machine
  has your data as outputs?

- Try to avoid correlated variables. Some of the more robust samplers (**cough
  NUTS cough HMC cough cough**) can cope with _a posteriori_ correlated random
  variables, but sampling is much easier for everyone involved if the variables
  are uncorrelated. By the way, the bar is pretty low here: if the
  jointplot/scattergram of the two variables looks like an ellipse, thats
  usually okay. It's when the ellipse starts looking like a line that you should
  be alarmed.

## Hierarchical Models

- First off, hierarchical models are great! [The PyMC3
  docs](https://docs.pymc.io/notebooks/GLM-hierarchical.html) opine on this at
  length, so let's not waste ink.

- The poster child of a Bayesian hierarchical model is something like this:
  <img style="float: center" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/765f37f86fa26bef873048952dccc6e8067b78f4">
  <img style="float: center" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ca8c0e1233fd69fa4325c6eacf8462252ed6b00a">
  <img style="float: center" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1e56b3077b1b3ec867d6a0f2539ba9a3e79b45c1">
  This hierarchy has 3 levels (some would say it has 2 levels, since there are
  only 2 levels of parameters that we need to infer, but whatever). This is
  fine, but any more levels, and it becomes harder for to sample. Try out a
  taller hierarchy to see if it works, but err on the side of 3-level (or
  2-level, whatever) hierarchies.

- If your hierarchy is too tall, you can truncate it by introducing a
  deterministic function of your parameters somewhere (this usually turns out to
  just be a sum). For example, instead of modeling your observations are drawn
  from a 4-level hierarchy, maybe your observations can be modeled as the sum of
  three parameters, where these parameters are drawn from a 3-level hierarchy.

- More in-depth treatment here in [Betancourt and Girolami's
  paper](https://arxiv.org/abs/1312.0906). *tl;dr:* hierarchical models all but
  _require_ you use to use Hamiltonian Monte Carlo, plus some practical tips and
  goodies on how to do that stuff in the real world.

## Implementing the Model

- PyMC3 has one quirky piece of syntax, which I tripped up on for a while. It's
  described quite well in [this comment on Thomas Wiecki's
  blog](http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/#comment-2213376737).
  Basically, suppose you have several groups, and want to initialize several
  variables per group, but you want to initialize different numbers of variables
  for each group. Then you need to use the quirky `random_variables[index]`
  notation. I suggest using `scikit-learn`'s `LabelEncoder` to easily create the
  index. For example, to make normally distributed heights for the iris dataset:

``` python
# Different numbers of examples for each species
species = (48 * ['setosa'] + 52 * ['virginica'] + 63 * ['versicolor'])
# One variable per group 
heights_per_species = pm.Normal('heights_per_species',
                              mu=0, sd=1, shape=len(species))
idx = sklearn.LabelEncoder().fit_transform(species)
heights = heights_per_species[idx]
```

## MCMC: Initialization and Sampling

- Have faith in PyMC3's default sampling and initialization settings: someone
  much more experienced than us took the time to choose them!

- Never initialize the sampler with the MAP estimate! In low dimensional
  problems the MAP estimate (a.k.a. the mode of the posterior) is often quite a
  reasonable point. But in high dimensions, the MAP becomes very strange. Check
  out [Ferenc
  Huszár's](http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/)
  blog posts on high-dimensional Gaussians to see why. Besides, at the MAP the
  derivatives of the posterior are zero, and that isn't great for
  derivative-based samplers.

- If you get scary errors that describe mathematical problems (e.g. `ValueError:
  Mass matrix contains zeros on the diagonal. Some derivatives might always be
  zero.`), then you are exceptionally out of luck: those kinds of errors are
  really hard to debug. I can only point to the [Folk Theorem of Statistical
  Computing](http://andrewgelman.com/2008/05/13/the_folk_theore/):

> If you have computational problems, probably your model is wrong.

## Inspecting the Model

- You've hit the _Magic Inference Button™_, and you now have a `trace` object.
  What do you do now?

1. Check for divergences. PyMC3's sampler will spit out a warning if there are
   diverging chains, but the following code snippet may make things easier:

``` python
# Display the total number and percentage of divergent chains
diverging = trace['diverging']
print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
diverging_perc = divergent.nonzero()[0].size / len(trace) * 100
print('Percentage of Divergent Chains: {:.1f}'.format(diverging_perc))
```

2. Check the traceplot. You're looking for traceplots that look like "fuzzy
   caterpillars" (as Michael Betancourt puts it). If the trace moves into some
   region and stays there for a long time (a.k.a. there are some "sticky
   regions", that's cause for concern! That indicates that once the sampler
   moves into some region of parameter space, it gets stuck there (probably due
   to high curvature or other bad topological properties).

3. Run for both short _and_ long chains (`draws=500` and `draws=2000`,
   respectively, are good numbers, with `tune` increasing commensurately). PyMC3
   has a nice helper function to pretty-print a summary table of the trace:
   `pm.summary(long_trace).round(2)`. Look out for:
   - the $\hat{R}$ values (a.k.a. the Gelman-Rubin test statistic, a.k.a. the
     potential scale reduction factor, a.k.a. PSRF): are they all close to 1?
     If not, something is _horribly_ wrong. Consider respecifying or
     reparameterizing your model.
   - the number of effective samples _per iteration_: does it fall drastically?
     If so, this means that we are exploring less efficiently the longer we let
     our chains run. Something bad is happening. Inspect the jointplots of
     your variables, and plot the divergent samples. Do they cluster anywhere in
     parameter space?
   - the sign and magnitude of the inferred values: do they make sense, or are
     they unexpected and unreasonable? This could indicate a poorly specified
     model.

4. Run the following function (adapted from a code snippet in [the PyMC3
   docs](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html))
   to inspect your variables one at a time (if you have a plate of variables, it's
   fine to pick a couple at random). The most interesting plot at this point
   will be the jointplot/scattergram. It will tell you if the two random
   variables are correlated, and help identify any troublesome neighborhoods in
   the parameter space (divergent samples will be colored differently, and will
   cluster near such neighborhoods).

``` python
def inspect_variable(trace, var_1, var_2=None):
    # Traceplot of var_1. 
    pm.traceplot(trace, varnames=[var_1])

    # Cumulating mean of var_1.
    cum_mean = [np.mean(trace[var_1][:i])
                for i in np.arange(1, len(trace[var_1]))]
    plt.figure(figsize=(15, 4))
    plt.plot(cum_mean, lw=2.5)
    plt.xlabel('Iteration')
    plt.ylabel('MCMC mean of {}'.format(var_1))
    plt.title('MCMC estimation of {}'.format(var_1))
    plt.show()

    # Scattergram between var_1 and var_2. To identify correlations
    # and problematic neighborhoods in parameter space.
    if var_2 is not None:
        pm.pairplot(trace,
                    sub_varnames=[var_1, var_2],
                    divergences=True,
                    color='C3',
                    figsize=(10, 5),
                    kwargs_divergence={'color': 'C2'})
        plt.title('Scatter Plot between {} and {}'.format(var_1, var_2))
```

## Fixing Divergences

- Remember: if you have even _one_ diverging chain, you should be concerned.

- Increase `target_accept`: usually 0.9 is a good number (currently the default
  in PyMC3 is 0.8). This will help get rid of false positives from the test for
  divergences. However, divergences that _don't_ go away are cause for alarm.

- Consider a _non-centered_ model. This is an amazing trick: it all boils down
  to the familiar equation $ X = \sigma Z + \mu $ from STAT 101, but it honestly
  works wonders. See [Thomas Wiecki's blog
  post](http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
  on it, and [a page from the PyMC3
  documentation](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html).

