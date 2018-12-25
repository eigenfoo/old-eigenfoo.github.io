---
title: Cookbook — Bayesian Modelling with PyMC3
excerpt: "Recently I've started using [PyMC3](https://github.com/pymc-devs/pymc3) for
Bayesian modelling, and it's an amazing piece of software! The API only exposes
as much of heavy machinery of MCMC as you need — by which I mean, just the
`pm.sample()` method."
tags:
  - bayesianism
  - pymc3
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background5.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
toc: true
toc_sticky: true
toc_label: "Recipes"
toc_icon: "utensils"
mathjax: true
last_modified_at: 2018-06-24
---

Recently I've started using [PyMC3](https://github.com/pymc-devs/pymc3) for
Bayesian modelling, and it's an amazing piece of software! The API only exposes
as much of heavy machinery of MCMC as you need — by which I mean, just the
`pm.sample()` method (a.k.a., as [Thomas
Wiecki](http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/) puts it, the
_Magic Inference Button™_). This really frees up your mind to think about your
data and model, which is really the heart and soul of data science!

That being said however, I quickly realized that the water gets very deep very
fast: I explored my data set, specified a hierarchical model that made sense to
me, hit the _Magic Inference Button™_, and… uh, what now?  I blinked at the
angry red warnings the sampler spat out.

So began by long, rewarding and ongoing exploration of Bayesian modelling. This
is a compilation of notes, tips, tricks and recipes that I've collected from
everywhere: papers, documentation, peppering my [more
experienced](https://twitter.com/twiecki)
[colleagues](https://twitter.com/aseyboldt) with questions. It's still very much
a work in progress, but hopefully somebody else finds it useful!

<img style="float: middle" width="600"
src="https://cdn.rawgit.com/pymc-devs/pymc3/master/docs/logos/svg/PyMC3_banner.svg">

## For the Uninitiated

- First of all, _welcome!_ It's a brave new world out there — where statistics
  is cool, Bayesian and (if you're lucky) even easy. Dive in!

### Bayesian modelling

- If you don't know any probability, I'd recommend [Michael
  Betancourt's](https://betanalpha.github.io/assets/case_studies/probability_theory.html)
  crash-course in practical probability theory.

- For an introduction to general Bayesian methods and modelling, I really liked
  [Cam Davidson Pilon's _Bayesian Methods for
  Hackers_](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/):
  it really made the whole “thinking like a Bayesian” thing click for me.

- If you're willing to spend some money, I've heard that [_Doing Bayesian Data
  Analysis_ by
  Kruschke](https://sites.google.com/site/doingbayesiandataanalysis/) (a.k.a.
  _“the puppy book”_) is for the bucket list.

- Here we come to a fork in the road. The central problem in Bayesian modelling
  is this: given data and a probabilistic model that we think models this data,
  how do we find the posterior distribution of the model's parameters? There are
  currently two good solutions to this problem. One is Markov-chain Monte Carlo
  sampling (a.k.a. MCMC sampling), and the other is variational inference
  (a.k.a. VI). Both methods are mathematical Death Stars: extremely powerful but
  incredibly complicated. Nevertheless, I think it's important to get at least a
  hand-wavy understanding of what these methods are. If you're new to all this,
  my personal recommendation is to invest your time in learning MCMC: it's been
  around longer, we know that there are sufficiently robust tools to help you,
  and there's a lot more support/documentation out there.

### Markov-chain Monte Carlo

- For a good high-level introduction to MCMC, I liked [Michael Betancourt's
  StanCon 2017
  talk](https://www.youtube.com/watch?v=DJ0c7Bm5Djk&feature=youtu.be&t=4h40m9s):
  especially the first few minutes where he provides a motivation for MCMC, that
  really put all this math into context for me.

- For a more in-depth (and mathematical) treatment of MCMC, I'd check out his
  [paper on Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434).

### Variational inference

- VI has been around for a while, but it was only in 2017 (2 years ago, at the
  time of writing) that _automatic differentiation variational inference_ was
  invented. As such, variational inference is undergoing a renaissance and is
  currently an active area of statistical research. Since it's such a nascent
  field, most resources on it are very theoretical and academic in nature.

- Chapter 10 (on approximate inference) in Bishop's _Pattern Recognition and
  Machine Learning_ and [this
  tutorial](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf)
  by David Blei are excellent, if a bit mathematically-intensive, resources.

- The most hands-on explanation of variational inference I've seen is the docs
  for [Pyro](http://pyro.ai/examples/svi_part_i.html), a probabilistic
  programming language developed by Uber that specializes in variational
  inference.

## Model Formulation

- Try thinking about _how_ your data would be generated: what kind of machine
  has your data as outputs? This will help you both explore your data, as well
  as help you arrive at a reasonable model formulation.

- Try to avoid correlated variables. Some of the more robust samplers (**cough**
  NUTS **cough cough**) can cope with _a posteriori_ correlated random
  variables, but sampling is much easier for everyone involved if the variables
  are uncorrelated. By the way, the bar is pretty low here: if the
  jointplot/scattergram of the two variables looks like an ellipse, thats
  usually okay. It's when the ellipse starts looking like a line that you should
  be alarmed.

- Try to avoid discrete latent variables, and discrete parameters in general.
  There is no good method to sample them in a smart way (since discrete
  parameters have no gradients); and with “naïve” samplers (i.e. those that do
  not take advantage of the gradient), the number of samples one needs to make
  good inferences generally scales exponentially in the number of parameters.
  For an instance of this, see [this example on marginal Gaussian
  mixtures](https://docs.pymc.io/notebooks/marginalized_gaussian_mixture_model.html).

- The [Stan GitHub
  wiki](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) has
  some excellent recommendations on how to choose good priors. Once you get a
  good handle on the basics of using PyMC3, I _100% recommend_ reading this wiki
  from start to end: the Stan community has fantastic resources on Bayesian
  statistics, and even though their APIs are quite different, the mathematical
  theory all translates over.

### Hierarchical models

- First of all, hierarchical models are amazing! [The PyMC3
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
  just be a sum). For example, instead of modelling your observations are drawn
  from a 4-level hierarchy, maybe your observations can be modeled as the sum of
  three parameters, where these parameters are drawn from a 3-level hierarchy.

- More in-depth treatment here in [(Betancourt and Girolami,
  2013)](https://arxiv.org/abs/1312.0906). **tl;dr:** hierarchical models all
  but _require_ you use to use Hamiltonian Monte Carlo; also included are some
  practical tips and goodies on how to do that stuff in the real world.

## Model Implementation

- At the risk of overgeneralizing, there are only two things that can go wrong
  in Bayesian modelling: either your data is wrong, or your model is wrong. And
  it is a hell of a lot easier to debug your data than it is to debug your
  model. So before you even try implementing your model, plot histograms of your
  data, count the number of data points, drop any NaNs, etc. etc.

- PyMC3 has one quirky piece of syntax, which I tripped up on for a while. It's
  described quite well in [this comment on Thomas Wiecki's
  blog](http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/#comment-2213376737).
  Basically, suppose you have several groups, and want to initialize several
  variables per group, but you want to initialize different numbers of variables
  for each group. Then you need to use the quirky `variables[index]`
  notation. I suggest using `scikit-learn`'s `LabelEncoder` to easily create the
  index. For example, to make normally distributed heights for the iris dataset:

  ```python
  # Different numbers of examples for each species
  species = (48 * ['setosa'] + 52 * ['virginica'] + 63 * ['versicolor'])
  num_species = len(list(set(species)))  # 3
  # One variable per group 
  heights_per_species = pm.Normal('heights_per_species',
                                  mu=0, sd=1, shape=num_species)
  idx = sklearn.preprocessing.LabelEncoder().fit_transform(species)
  heights = heights_per_species[idx]
  ```

- You might find yourself in a situation in which you want to use a centered
  parameterization for a portion of your data set, but a noncentered
  parameterization for the rest of your data set (see below for what these
  parameterizations are). There's a useful idiom for you here:

  ```python
  num_xs = 5
  use_centered = np.array([0, 1, 1, 0, 1])  # len(use_centered) = num_xs
  x_sd = pm.HalfCauchy('x_sd', sd=1)
  x_raw = pm.Normal('x_raw', mu=0, sd=mu_x_sd**use_centered, shape=num_xs)
  x = pm.Deterministic('x', x_sd**(1 - use_centered) * x_raw)
  ```

  You could even experiment with allowing `use_centered` to be _between_ 0 and
  1, instead of being _either_ 0 or 1!

- I prefer to use the `pm.Deterministic` function instead of simply using normal
  arithmetic operations (e.g. I'd prefer to write `x = pm.Deterministic('x', y +
  z)` instead of `x = y + z`). This means that you can index the `trace` object
  later on with just `trace['x']`, instead of having to compute it yourself with
  `trace['y'] + trace['z']`.

## MCMC Initialization and Sampling

- Have faith in PyMC3's default initialization and sampling settings: someone
  much more experienced than us took the time to choose them! NUTS is the most
  efficient MCMC sampler known to man, and `jitter+adapt_diag`… well, you get
  the point.

- However, if you're truly grasping at straws, the more powerful initialization
  setting would be `advi` or `advi+adapt_diag`, which uses variational
  inference to initialize the sampler. An even better option would be to use
  `advi+adapt_diag_grad`, which is (at the time of writing) an experimental
  feature in beta.

- Never initialize the sampler with the MAP estimate! In low dimensional
  problems the MAP estimate (a.k.a. the mode of the posterior) is often quite a
  reasonable point. But in high dimensions, the MAP becomes very strange. Check
  out [Ferenc Huszár's blog
  post](http://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/)
  on high-dimensional Gaussians to see why. Besides, at the MAP all the derivatives
  of the posterior are zero, and that isn't great for derivative-based samplers.

## MCMC Trace Diagnostics

- You've hit the _Magic Inference Button™_, and you have a `trace` object. Now
  what? First of all, make sure that your sampler didn't barf itself, and that
  your chains are safe for consumption (i.e., analysis).

1. Run the chain for as long as you have the patience or resources for. Make
   sure that the `tune` parameter increases commensurately with the `draws`
   parameter.

2. Check for divergences. PyMC3's sampler will spit out a warning if there are
   diverging chains, but the following code snippet may make things easier:

   ```python
   # Display the total number and percentage of divergent chains
   diverging = trace['diverging']
   print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
   diverging_perc = divergent.nonzero()[0].size / len(trace) * 100
   print('Percentage of Divergent Chains: {:.1f}'.format(diverging_perc))
   ```

3. Check the traceplot (`pm.traceplot(trace)`). You're looking for traceplots
   that look like “fuzzy caterpillars”. If the trace moves into some region and
   stays there for a long time (a.k.a. there are some “sticky regions”), that's
   cause for concern! That indicates that once the sampler moves into some
   region of parameter space, it gets stuck there (probably due to high
   curvature or other bad topological properties).

4. In addition to the traceplot, there are [a ton of other
   plots](https://docs.pymc.io/api/plots.html) you can make with your trace:

   - `pm.plot_posterior(trace)`: check if your posteriors look reasonable.
   - `pm.forestplot(trace)`: check if your variables have reasonable credible
     intervals, and Gelman–Rubin scores close to 1.
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

5. PyMC3 has a nice helper function to pretty-print a summary table of the
   trace: `pm.summary(trace)` (I usually tack on a `.round(2)` for my sanity).
   Look out for:
   - the $$\hat{R}$$ values (a.k.a. the Gelman–Rubin statistic, a.k.a. the
     potential scale reduction factor, a.k.a. the PSRF): are they all close to
     1? If not, something is _horribly_ wrong. Consider respecifying or
     reparameterizing your model. You can also inspect these in the forest plot.
   - the sign and magnitude of the inferred values: do they make sense, or are
     they unexpected and unreasonable? This could indicate a poorly specified
     model. (E.g. parameters of the unexpected sign that have low uncertainties
     might indicate that your model needs interaction terms.)

6. As a drastic debugging measure, try to `pm.sample` with `draws=1`,
   `tune=500`, and `discard_tuned_samples=False`, and inspect the traceplot.
   During the tuning phase, we don't expect to see friendly fuzzy caterpillars,
   but we _do_ expect to see good (if noisy) exploration of parameter space. So
   if the sampler is getting stuck during the tuning phase, that might explain
   why the trace looks horrible.

7. If you get scary errors that describe mathematical problems (e.g. `ValueError:
   Mass matrix contains zeros on the diagonal. Some derivatives might always be
   zero.`), then you're ~~shit out of luck~~ exceptionally unlucky: those kinds of
   errors are notoriously hard to debug. I can only point to the [Folk Theorem of
   Statistical Computing](http://andrewgelman.com/2008/05/13/the_folk_theore/):

   > If you're having computational problems, probably your model is wrong.

### Fixing divergences

> `There were N divergences after tuning. Increase 'target_accept' or reparameterize.`
>
>  — The _Magic Inference Button™_

- Divergences in HMC occur when the sampler finds itself in regions of extremely
  high curvature (such as the opening of the a hierarchical funnel). Broadly
  speaking, the sampler is prone to malfunction in such regions, causing the
  sampler to fly off towards to infinity. The ruins the chains by heavily
  biasing the samples.

- Remember: if you have even _one_ diverging chain, you should be worried.

- Increase `target_accept`: usually 0.9 is a good number (currently the default
  in PyMC3 is 0.8). This will help get rid of false positives from the test for
  divergences. However, divergences that _don't_ go away are cause for alarm.

- Increasing `tune` can sometimes help as well: this gives the sampler more time
  to 1) find the typical set and 2) find good values for step sizes, scaling
  factors, etc. If you're running into divergences, it's always possible that
  the sampler just hasn't started the mixing phase and is still trying to find
  the typical set.

- Consider a _noncentered_ parameterization. This is an amazing trick: it all boils down
  to the familiar equation $$X = \sigma Z + \mu$$ from STAT 101, but it honestly
  works wonders. See [Thomas Wiecki's blog
  post](http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/)
  on it, and [this page from the PyMC3
  documentation](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html).

- If that doesn't work, there may be something wrong with the way you're
  thinking about your data: consider reparameterizing your model, or
  respecifying it entirely.

### Other common warnings

- It's worth noting that far and away the worst warning to get is the one about
  divergences. While a divergent chain indicates that your inference may be
  flat-out _invalid_, the rest of these warnings indicate that your inference is
  merely (lol, “merely”) _inefficient_.

- `The number of effective samples is smaller than XYZ for some parameters.`
  - Quoting [Junpeng Lao on
    discourse.pymc3.io](https://discourse.pymc.io/t/the-number-of-effective-samples-is-smaller-than-25-for-some-parameters/1050/3):
    “A low number of effective samples is usually an indication of strong
    autocorrelation in the chain.”
  - Make sure you're using an efficient sampler like NUTS. (And not, for
    instance, Metropolis–Hastings. (I mean seriously, it's the 21st century, why
    would you ever want Metropolis–Hastings?))
  - Tweak the acceptance probability (`target_accept`) — it should be large
    enough to ensure good exploration, but small enough to not reject all
    proposals and get stuck.

- `The gelman-rubin statistic is larger than XYZ for some parameters. This
  indicates slight problems during sampling.`
  - When PyMC3 samples, it runs several chains in parallel. Loosely speaking,
    the Gelman–Rubin statistic measures how similar these chains are. Ideally it
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

### Model reparameterization

- Countless warnings have told you to engage in this strange activity of
  “reparameterization”. What even is that? Luckily, the [Stan User
  Manual](https://github.com/stan-dev/stan/releases/download/v2.17.1/stan-reference-2.17.1.pdf)
  (specifically the _Reparameterization and Change of Variables_ section) has
  an excellent explanation of reparameterization, and even some practical tips
  to help you do it (although your mileage may vary on how useful those tips
  will be to you).

- Asides from meekly pointing to other resources, there's not much I can do to
  help: this stuff really comes from a combination of intuition, statistical
  knowledge and good ol' experience. I can, however, cite some examples to give
  you a better idea.
  - The noncentered parameterization is a classic example. If you have a
    parameter whose mean and variance you are also modelling, the noncentered
    parameterization decouples the sampling of mean and variance from the
    sampling of the parameter, so that they are now independent. In this way, we
    avoid “funnels”.
  - The [_horseshoe
    distribution_](http://proceedings.mlr.press/v5/carvalho09a.html) is known to
    be a good shrinkage prior, as it is _very_ spikey near zero, and has _very_
    long tails. However, modelling it using one parameter can give multimodal
    posteriors — an exceptionally bad result. The trick is to reparameterize and
    model it as the product of two parameters: one to create spikiness at zero,
    and one to create long tails (which makes sense: to sample from the
    horseshoe, take the product of samples from a normal and a half-Cauchy).

## Model Diagnostics

- Admittedly the distinction between the previous section and this one is
  somewhat artificial (since problems with your chains indicate problems with
  your model), but I still think it's useful to make this distinction because
  these checks indicate that you're thinking about your data in the wrong way,
  (i.e. you made a poor modelling decision), and _not_ that the sampler is having
  a hard time doing its job.

1. Run the following snippet of code to inspect the pairplot of your variables
   one at a time (if you have a plate of variables, it's fine to pick a couple
   at random). It'll tell you if the two random variables are correlated, and
   help identify any troublesome neighborhoods in the parameter space (divergent
   samples will be colored differently, and will cluster near such
   neighborhoods).

   ```python
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

3. Broadly speaking, there are four kinds of bad geometries that your posterior
   can suffer from:
   - highly correlated posteriors: this will probably cause divergences or
     traces that don't look like “fuzzy caterpillars”. Either look at the
     jointplots of each pair of variables, or look at the correlation matrix of
     all variables.  Try using a centered parameterization, or reparameterize in
     some other way, to remove these correlations.
   - posteriors that form “funnels”: this will probably cause divergences. Try
     using a noncentered parameterization.
   - long tailed posteriors: this will probably raise warnings about
     `max_treedepth` being exceeded. If your data has long tails, you should
     model that with a long-tailed distribution. If your data doesn't have long
     tails, then your model is ill-specified: perhaps a more informative prior
     would help.
   - multimodal posteriors: right now this is pretty much a death blow. At the
     time of writing, all samplers have a hard time with multimodality, and
     there's not much you can do about that. Try reparameterizing to get a
     unimodal posterior. If that's not possible (perhaps you're _modelling_
     multimodality using a mixture model), you're out of luck: just let NUTS
     sample for a day or so, and hopefully you'll get a good trace.

4. Pick a small subset of your raw data, and see what exactly your model does
   with that data (i.e. run the model on a specific subset of your data). I find
   that a lot of problems with your model can be found this way.

5. Run [_posterior predictive
   checks_](https://docs.pymc.io/notebooks/posterior_predictive.html) (a.k.a.
   PPCs): sample from your posterior, plug it back in to your model, and
   “generate new data sets”. PyMC3 even has a nice function to do all this for
   you: `pm.sample_ppc`. But what do you do with these new data sets? That's a
   question only you can answer! The point of a PPC is to see if the generated
   data sets reproduce patterns you care about in the observed real data set,
   and only you know what patterns you care about. E.g. how close are the PPC
   means to the observed sample mean? What about the variance?
   - For example, suppose you were modelling the levels of radon gas in
     different counties in a country (through a hierarchical model). Then you
     could sample radon gas levels from the posterior for each county, and take
     the maximum within each county. You'd then have a distribution of maximum
     radon gas levels across counties. You could then check if the _actual_
     maximum radon gas level (in your observed data set) is acceptably within
     that distribution. If it's much larger than the maxima, then you would know
     that the actual likelihood has longer tails than you assumed (e.g. perhaps
     you should use a Student's T instead of a normal?)
   - Remember that how well the posterior predictive distribution fits the data
     is of little consequence (e.g. the expectation that 90% of the data should
     fall within the 90% credible interval of the posterior). The posterior
     predictive distribution tells you what values for data you would expect if
     we were to remeasure, given that you've already observed the data you did.
     As such, it's informed by your prior as well as your data, and it's not
     its job to adequately fit your data!
