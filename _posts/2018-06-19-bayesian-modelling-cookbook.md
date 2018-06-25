---
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
toc_sticky: true
toc_label: "Recipes"
toc_icon: "utensils"
mathjax: true
last_modified_at: 2018-06-19
---

Recently I've started using [PyMC3](https://github.com/pymc-devs/pymc3) for
Bayesian modeling, and it's an amazing piece of software! The API only exposes
as much of heavy machinery of MCMC as you need - by which I mean, the
`pm.sample()` method (a.k.a., as [Thomas
Wiecki](http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/) puts it, the
_Magic Inference Button™_). This really frees up your mind to think about your
data and model, which is really the heart and soul data science!

That being said however, I quickly realized that the water gets very deep very
fast: I explored my data set, specified a hierarchical model that made sense to
me, hit the _Magic Inference Button™_, and... uh, what now?  I blinked at the
angry red warnings the sampler spat out.

So began by long, rewarding and ongoing exploration of Bayesian modeling. This
is a compilation of notes, tips, tricks and recipes that I've collected from
everywhere: papers, documentation, peppering my [more
experienced](https://twitter.com/twiecki)
[colleagues](https://twitter.com/aseyboldt) with questions. It's still very much
a work in progress, but hopefully somebody else finds it useful!

## For the Uninitiated

- First of all, _welcome!_ It's a brave new world out there - where statistics
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

## Model Formulation

- Try thinking about _how_ your data would be generated: what kind of machine
  has your data as outputs? This will help you both explore your data, as well
  as help you arrive at a reasonable model formulation.

- Try to avoid correlated variables. Some of the more robust samplers (**cough**
  NUTS **cough** HMC **cough cough**) can cope with _a posteriori_ correlated random
  variables, but sampling is much easier for everyone involved if the variables
  are uncorrelated. By the way, the bar is pretty low here: if the
  jointplot/scattergram of the two variables looks like an ellipse, thats
  usually okay. It's when the ellipse starts looking like a line that you should
  be alarmed.

### Hierarchical Models

- First off, hierarchical models are great! [The PyMC3
  docs](https://docs.pymc.io/notebooks/GLM-hierarchical.html) opine on this at
  length, so let's not waste any digital ink.

- The poster child of a Bayesian hierarchical model looks something like this
  (equations taken from
  [Wikipedia](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling)):

  <img style="float: center" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/765f37f86fa26bef873048952dccc6e8067b78f4">

  <img style="float: center" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ca8c0e1233fd69fa4325c6eacf8462252ed6b00a">

  <img style="float: center" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1e56b3077b1b3ec867d6a0f2539ba9a3e79b45c1">

  This hierarchy has 3 levels (some would say it has 2 levels, since there are
  only 2 levels of parameters to infer, but honestly whatever: by my count there
  are 3). 3 levels is fine, but add any more levels, and it becomes harder for
  to sample. Try out a taller hierarchy to see if it works, but err on the side
  of 3-level hierarchies.

- If your hierarchy is too tall, you can truncate it by introducing a
  deterministic function of your parameters somewhere (this usually turns out to
  just be a sum). For example, instead of modeling your observations are drawn
  from a 4-level hierarchy, maybe your observations can be modeled as the sum of
  three parameters, where these parameters are drawn from a 3-level hierarchy.

- More in-depth treatment here in [(Betancourt and Girolami,
  2013)](https://arxiv.org/abs/1312.0906). **tl;dr:** hierarchical models all
  but _require_ you use to use Hamiltonian Monte Carlo; also included are some
  practical tips and goodies on how to do that stuff in the real world.

## Model Implementation

- PyMC3 has one quirky piece of syntax, which I tripped up on for a while. It's
  described quite well in [this comment on Thomas Wiecki's
  blog](http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/#comment-2213376737).
  Basically, suppose you have several groups, and want to initialize several
  variables per group, but you want to initialize different numbers of variables
  for each group. Then you need to use the quirky `variables[index]`
  notation. I suggest using `scikit-learn`'s `LabelEncoder` to easily create the
  index. For example, to make normally distributed heights for the iris dataset:

  ``` python
  # Different numbers of examples for each species
  species = (48 * ['setosa'] + 52 * ['virginica'] + 63 * ['versicolor'])
  num_species = len(list(set(species)))  # 3
  # One variable per group 
  heights_per_species = pm.Normal('heights_per_species',
                                  mu=0, sd=1, shape=num_species)
  idx = sklearn.preprocessing.LabelEncoder().fit_transform(species)
  heights = heights_per_species[idx]
  ```

## MCMC Initialization and Sampling

- Have faith in PyMC3's default initialization and sampling settings: someone
  much more experienced than us took the time to choose them! NUTS is the most
  efficient MCMC sampler known to man, and `jitter+adapt diag`... well, you get
  the point.

- Never initialize the sampler with the MAP estimate. In low dimensional
  problems the MAP estimate (a.k.a. the mode of the posterior) is often quite a
  reasonable point. But in high dimensions, the MAP becomes very strange. Check
  out [Ferenc Huszár's blog
  post](http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/)
  on high-dimensional Gaussians to see why. Besides, at the MAP all the derivatives
  of the posterior are zero, and that isn't great for derivative-based samplers.

## MCMC Trace Inspection

- You've hit the _Magic Inference Button™_, and you have a `trace` object. Now
  what? First of all, make sure that your sampler didn't barf itself, and that
  your chains are safe for consumption (i.e., analysis).

1. Check for divergences. PyMC3's sampler will spit out a warning if there are
   diverging chains, but the following code snippet may make things easier:

   ``` python
   # Display the total number and percentage of divergent chains
   diverging = trace['diverging']
   print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
   diverging_perc = divergent.nonzero()[0].size / len(trace) * 100
   print('Percentage of Divergent Chains: {:.1f}'.format(diverging_perc))
   ```

2. Check the traceplot (`pm.traceplot(trace)`). You're looking for traceplots
   that look like "fuzzy caterpillars" (as Michael Betancourt puts it). If the
   trace moves into some region and stays there for a long time (a.k.a. there
   are some "sticky regions"), that's cause for concern! That indicates that
   once the sampler moves into some region of parameter space, it gets stuck
   there (probably due to high curvature or other bad topological properties).

3. In addition to the traceplot, there are [a ton of other
   plots](https://docs.pymc.io/api/plots.html) you can make with your trace:

   - `pm.plot_posterior(trace)`: check if your posteriors look reasonable.
   - `pm.forestplot(trace)`: check if your variables have reasonable credible
     intervals.
   - `pm.autocorrplot(trace)`: check if your chains are impaired by high
     autocorrelation. Also remember that thinning your chains is a waste of
     time at best, and deluding yourself at worst. See Chris Fonnesbeck's
     comment on [this GitHub
     issue](https://github.com/pymc-devs/pymc/issues/23) and [Junpeng Lao's
     reply to Michael Betancourt's
     tweet](https://twitter.com/junpenglao/status/1009748562136256512)
   - `pm.energyplot(trace)`: ideally the energy and marginal energy
     distributions should look very similar. Long tails in the distribution of
     energy levels indicates deteriorated sampler efficiency.
   - `pm.densityplot(trace)`: a souped-up version of `pm.plot_posterior`. It
     doesn't seem to be wildly useful unless you're plotting posteriors from
     multiple models.

4. Run both short _and_ long chains (`draws=500` and `draws=2000`,
   respectively, are good numbers, with `tune` increasing commensurately). PyMC3
   has a nice helper function to pretty-print a summary table of the trace:
   `pm.summary(long_trace).round(2)`. Look out for:
   - the $$\hat{R}$$ values (a.k.a. the Gelman-Rubin statistic, a.k.a. the
     potential scale reduction factor, a.k.a. PSRF): are they all close to 1?
     If not, something is _horribly_ wrong. Consider respecifying or
     reparameterizing your model.
   - the number of effective samples _per iteration_ (you may need to do the
     division yourself): does it fall drastically?  If so, this means that we
     are exploring less efficiently the longer we let our chains run. Something
     bad is happening. Inspect the jointplots of your variables, and plot the
     divergent samples. Do they cluster anywhere in parameter space?
   - the sign and magnitude of the inferred values: do they make sense, or are
     they unexpected and unreasonable? This could indicate a poorly specified
     model. (E.g. parameters of the unexpected sign that have low uncertainties
     might indicate that your model needs interaction terms.)

5. If you get scary errors that describe mathematical problems (e.g. `ValueError:
   Mass matrix contains zeros on the diagonal. Some derivatives might always be
   zero.`), then you're ~~shit out of luck~~ exceptionally unlucky: those kinds of
   errors are notoriously hard to debug. I can only point to the [Folk Theorem of
   Statistical Computing](http://andrewgelman.com/2008/05/13/the_folk_theore/):

   > If you're having computational problems, probably your model is wrong.

### Divergences

> `There were N divergences after tuning. Increase 'target_accept' or reparameterize.`
>   \- The _Magic Inference Button™_

- Remember: if you have even _one_ diverging chain, you should be concerned.

- Increase `target_accept`: usually 0.9 is a good number (currently the default
  in PyMC3 is 0.8). This will help get rid of false positives from the test for
  divergences. However, divergences that _don't_ go away are cause for alarm.

- Increasing `tune` can sometimes help as well: this gives the sampler more time
  to 1) find the typical set and 2) find good values for step sizes, scaling
  factors, etc. If you're running into divergences, it's always possible that
  the sampler just hasn't started the mixing phase and is still trying to find
  the typical set.

- Consider a _non-centered_ model. This is an amazing trick: it all boils down
  to the familiar equation $$X = \sigma Z + \mu$$ from STAT 101, but it honestly
  works wonders. See [Thomas Wiecki's blog
  post](http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
  on it, and [this page from the PyMC3
  documentation](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html).

- If that doesn't work, there may be something wrong with the way you're
  thinking about your data: consider reparameterizing your model, or
  respecifying it entirely.

### Other Common Warnings

- It's worth noting that far and away the worst warning to get is the one about
  divergences. While a divergent chain indicates that your inference may be
  flat-out _invalid_, the rest of these warnings indicate that your inference is
  merely (uh, "merely") _inefficient_.

- `The number of effective samples is smaller than XYZ for some parameters.`
  - Quoting [Junpeng Lao on
    discourse.pymc3.io](https://discourse.pymc.io/t/the-number-of-effective-samples-is-smaller-than-25-for-some-parameters/1050/3):
    "A low number of effective samples is usually an indication of strong
    autocorrelation in the chain."
  - Make sure you're using an efficient sampler like NUTS. (And not, for
    instance, Metropolis-Hastings. (I mean seriously, it's the 21st century, why
    would you ever want Metropolis-Hastings?))
  - Tweak the acceptance probability (`target_accept`) - it should be large
    enough to ensure good exploration, but small enough to not reject all
    proposals and get stuck.

- `The gelman-rubin statistic is larger than XYZ for some parameters. This
  indicates slight problems during sampling.`
  - When PyMC3 samples, it runs several chains in parallel. Loosely speaking,
    the Gelman-Rubin statistic measures how similar these chains are. Ideally it
    should be close to 1.
  - Increasing the `tune` parameter may help, for the same reasons as described
    in the _Fixing Divergences_ section.

- `The chain reached the maximum tree depth. Increase max_treedepth, increase
  target_accept or reparameterize.`
  - NUTS puts a cap on the depth of the trees that it evaluates during each
    iteration, which is controlled through the `max_treedepth`. Reaching the maximum
    allowable tree depth indicates that NUTS is prematurely pulling the plug to
    avoid excessive compute time.
  - Yeah, what the _Magic Inference Button™_ says: try increasing
    `max_treedepth` or `target_accept`.

### Model Reparameterization

- Countless warnings have told you to engage in this strange activity of
  "reparameterization". What even is that? Luckily, the [Stan User
  Manual](https://github.com/stan-dev/stan/releases/download/v2.17.1/stan-reference-2.17.1.pdf)
  (specifically the _Reparameterization and Change of Variables_ section) has
  an excellent explanation of reparameterization, and even some practical tips
  to help you do it.

## Model Inspection

- Admittedly the distinction between the previous section and this one is
  somewhat artificial (since problems with your chains indicate problems with
  your model), but I still think it's useful to make this distinction because
  these checks indicate that you're thinking about your data in the wrong way,
  (i.e. you made a poor modeling decision), and _not_ that the sampler is having
  a hard time doing its job.

1. Run the following snippet of code to inspect the pairplot of your variables
   one at a time (if you have a plate of variables, it's fine to pick a couple
   at random). It'll tell you if the two random variables are correlated, and
   help identify any troublesome neighborhoods in the parameter space (divergent
   samples will be colored differently, and will cluster near such
   neighborhoods).

   ``` python
   pm.pairplot(trace,
               sub_varnames=[variable_1, variable_2],
               divergences=True,
               color='C3',
               kwargs_divergence={'color': 'C2'})
   ```

2. Look at your posteriors (either from the traceplot, density plots or
   posterior plots). Do they even make sense? E.g. are there outliers or long
   tails that you weren't expecting? Do their uncertainties look reasonable to
   you? If you had [a plate](https://en.wikipedia.org/wiki/Plate_notation) of
   variables, are their posteriors different? Did you expect them to be that
   way? If not, what about the data made the posteriors different? You're the
   only one who knows your problem/use case, so the posteriors better look good
   to you!

3. Pick a small subset of your raw data, and see what exactly your model does
   with that data (i.e. run the model on a specific subset of your data). I find
   that a lot of problems with your model can be found this way.

4. Run [_posterior predictive
   checks_](https://docs.pymc.io/notebooks/posterior_predictive.html) (a.k.a.
   PPCs): sample from your posterior, plug it back in to your model, and
   "generate new data sets". PyMC3 even has a nice function to do all this for
   you: `pm.sample_ppc`. But what to do with these new data sets? That's a
   question only you can answer! The point of a PPC is to see if the generated
   data sets reproduce patterns you care about in the observed real data set,
   and only you know what patterns you care about. E.g. how close are the PPC
   means to the observed sample mean? What about the variance? Do you care about
   skewness or kurtosis? Outliers?
