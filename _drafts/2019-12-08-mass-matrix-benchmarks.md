---
title: Benchmarks for Mass Matrix Adaptation
excerpt: "I benchmarked various mass matrix adaptation methods in PyMC3.
Sane defaults are easy to take for granted: it's more nuanced than I initially
expected!"
tags:
  - open source
  - pymc
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background6.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2019-12-12
search: false
---

{% if page.noindex == true %}
  <meta name="robots" content="noindex">
{% endif %}

I was lucky enough to be invited to attend the [Gradient
Retreat](https://gradientretreat.com/) earlier this month. It was an entire week
on a beautiful island with some amazingly intelligent Bayesians, and no demands
on my time other than the self-set (and admittedly vague) goal of contributing
to probabilistic programming in some way.

I initially tried to implement mass matrix adaptation in Tensorflow Probability,
but I quickly readjusted my goals[^1] to something more achievable: running some
benchmarks with HMC tuning.

<figure class="half">
    <a href="/assets/images/galiano.jpg"><img src="/assets/images/galiano.jpg"></a>
    <a href="/assets/images/galiano2.jpg"><img src="/assets/images/galiano2.jpg"></a>
    <figcaption>Pictures from Galiano Island.</figcaption>
</figure>

A quick rundown for those unfamiliar: _tuning_ is what happens before sampling,
during which the goal is not to actually draw samples, but to _prepare_ to draw
samples[^2]. For HMC and its variants, this means estimating HMC parameters such
as the step size, integration time and mass matrix[^3], the last of which is
basically the covariance matrix of the model parameters. Because my life is
finite (and I assume everybody else's is too), I limited myself to mass matrix
adaptation.

The interesting thing about tuning is that there are no rules: there are no
asymptotic guarantees we can rely on and no mathematical results we can turn to
for enlightened inspiration. The only thing we care about is obtaining a decent
estimate of the mass matrix, and preferably quickly.

Accompanying this lack of understanding of mass matrix adaptation is an
commensurate lack of (apparent) scientific inquiry — there is scant literature
to look to, and for open source developers, there is precious little prior art
to draw from when writing new implementations of HMC!

So I decided to do some empirical legwork and benchmark various methods of mass
matrix adaptation. Here are the questions I was interested in answering:

1. Is the assumption that the mass matrix is diagonal (in other words, assume
   that all parameters are uncorrelated) a good assumption to make?  What are
   the implications of this assumption for the tuning time, and the number of
   effective samples per second?

1. Does the tuning schedule (i.e. the sizes of the adaptation windows) make a
   big difference? Specifically, should we have a schedule of constant
   adaptation windows, or an "expanding schedule" of exponentially growing
   adaptation windows?

1. Besides assuming the mass matrix is diagonal, are there any other ways of
   simplifying mass matrix adaptation? For example, could we approximate the
   mass matrix as low rank?

I benchmarked five different mass matrix adaptation methods:

  1. A diagonal mass matrix (`diag`)
  1. A full (a.k.a. dense) mass matrix (`full`)
  1. A diagonal mass matrix adapted on an expanding schedule (`diag_exp`)
  1. A full mass matrix adapted on an expanding schedule (`diag_exp`)
  1. A low-rank approximation to the mass matrix using [Adrian Seyboldt's `covadapt` library](https://github.com/aseyboldt/covadapt).

I benchmarked these adaptation methods against six models:

  1. A 100-dimensional multivariate normal with a non-diagonal covariance matrix (`mvnormal`)
  1. A 100-dimensional multivariate normal with a low-rank covariance matrix (`lrnormal`)
  1. A [stochastic volatility model](https://docs.pymc.io/notebooks/stochastic_volatility.html) (`stoch_vol`)
  1. The [eight schools model](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html#The-Eight-Schools-Model) (`eight`)
  1. The [baseball model](https://docs.pymc.io/notebooks/hierarchical_partial_pooling.html) (`baseball`)
  1. A [Gaussian process](https://docs.pymc.io/notebooks/GP-SparseApprox.html#Examples) (`gp`)

Without further ado, the main results are shown below. Afterwards, I make some
general observations on the benchmarks, and finally (for the readers who care or
want to contribute) I describe various shortcomeings of my experimental setup
(which, if I were more optimistic, I would call "directions for further work").

### Tuning Times

This tabulates the tuning time, in seconds, of each adaptation method for each
model.

|            |**`mvnormal`**|**`lrnormal`**|**`stoch_vol`**|**`gp`**|**`eight`**|**`baseball`**
|:-----------|-------------:|-------------:|--------------:|-------:|----------:|------------:|
|**diag**    |        365.34|        340.10|         239.59|   18.47|       2.92|         5.32|
|**full**    |          8.29|        364.07|         904.95|   14.24|       2.91|         4.93|
|**diag_exp**|        358.50|        360.91|         219.65|   16.25|       3.05|         5.08|
|**full_exp**|          8.46|        142.20|         686.58|   14.87|       3.21|         6.04|
|**covadapt**|        386.13|         89.92|         398.08|     N/A|        N/A|          N/A|

### Effective Samples per Second

This tabulates the number of effective samples drawn by each adaptation method
for each model.

|            |**`mvnormal`**|**`lrnormal`**|**`stoch_vol`**|**`gp`**|**`eight`**|**`baseball`**
|:-----------|-------------:|-------------:|--------------:|-------:|----------:|------------:|
|**diag**    |          0.02|          1.55|          11.22|   65.36|     761.82|       455.23|
|**full**    |          1.73|          0.01|           6.71|  106.30|     840.77|       495.93|
|**diag_exp**|          0.02|          1.51|           9.79|   59.89|     640.90|       336.71|
|**full_exp**|      1,799.11|      1,753.65|           0.16|  101.99|     618.28|       360.14|
|**covadapt**|          0.02|        693.87|           5.71|     N/A|        N/A|          N/A|

## Observations

> **tldr:** As is typical with these sorts of things, no one adapataion method
> uniformly outperforms the others.

- A full mass matrix can provide significant improvements over a diagonal mass
  matrix for both the tuning time and the number of effective samples per
  second. This improvement can sometimes go up to two orders of magnitude!
  - This is most noticeable in the `mvnormal` model, with heavily correlated
    parameters.
  - However, in models with less extreme correlations among parameters, this
    advantage shrinks significantly (although it doesn't go away entirely).
    Full matrices can also take longer to tune. You can see this in the baseball
    or eight schools model.
  - Nevertheless, full mass matrices never seem to perform egregiously _worse_
    than diagonal mass matrices. This makes sense theoretically: a full mass
    matrix can be estimated to be diagonal, but not vice versa.

- Having an expanding schedule for tuning can sometimes give better performance,
  but nowhere near as significant as the difference between diagonal and full
  matrices. This difference is most noticeable for the `mvnormal` and `lrnormal`
  models (probably because these models have a constant covariance matrix and so
  more careful estimates using expanding windows can provide much better
  sampling).

- `covadapt` seems to run into some numerical difficulties? While running these
  benchmarks I ran into an inscrutable and non-reproducible
  [`ArpackError`](https://stackoverflow.com/q/18436667) from SciPy.

## Experimental Setup

- All samplers were run for 2000 tuning steps and 1000 sampling steps. This is
  unusually high, but is necessary for `covadapt` to work well, and I wanted to
  use the same number of iterations across all the benchmarks.

- My expanding schedule is as follows: the first adaptation window is 100
  iterations, and each subsequent window is 1.005 times the previous window.
  These numbers give 20 updates within 2000 iterations, while maintaining an
  exponentially increasing adaptation window size.

- I didn't run `covadapt` for models with fewer than 100 model parameters.
  With so few parameters, there's no need to approximate a mass matrix as
  low-rank: you can just estimate the full mass matrix!

- I set `target_accept` (a.k.a. `adapt delta` to Stan users) to 0.9 to make all
  divergences go away.

- All of these numbers were collected by sampling once per model per adaptation
  method (yes only once, sorry) in PyMC3, running on my MacBook Pro.

## Shortcomings

- In some sense comparing tuning times is not a fair comparison: it's possible
  that some mass matrix estimates converge quicker than others, and so comparing
  their tuning times is essentially penalizing these methods for converging
  faster than others.

- It's also possible that my expanding schedule for the adaptation windows just
  sucks! There's no reason why the first window needs to be 100 iterations, or
  why 1.005 should be a good multiplier. It looks like Stan [doubles their
  adaptation window
  sizes](https://github.com/stan-dev/stan/blob/736311d88e99b997f5b902409752fb29d6ec0def/src/stan/mcmc/windowed_adaptation.hpp#L95)
  during warmup.

- These benchmarks are done only for very basic toy models: I should test more
  extensively on more models that people in The Real World™ use.

- If you are interested in taking these benchmarks further (or perhaps just want
  to fact-check me on my results), the code is [sitting in this GitHub
  repository](https://github.com/eigenfoo/mass-matrix-benchmarks)[^4].

## References and Further Reading

- [Colin Carroll's talk on HMC
  tuning](https://colcarroll.github.io/hmc_tuning_talk/)
- [Stan reference manual of HMC algorithm
  parameters](https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html)
- [Dan Foreman-Mackey's blog post on dense mass matrices in
  PyMC3](https://dfm.io/posts/pymc3-mass-matrix/)
- [Adrian Seyboldt's low-rank mass matrix
  approximations](https://github.com/aseyboldt/covadapt)

---

[^1]: To quote a (nameless) PyMC developer,
      > Every Googler I meet is rather vocal about the dumpster fire that is
      > TensorFlow.

[^2]: It's good to point out that mass matrix adaptation is to make sampling
      more efficient, not more valid. Theoretically, any mass matrix would work:
      but a good one (i.e. a good estimate of the covariance matrix of the model
      parameters) could sample orders of magnitudes more efficiently.

[^3]: uh, _*sweats and looks around nervously for differential geometers*_
      more formally called the _metric_

[^4]: There are some violin plots lying around in the notebook, a relic of a
      time when I thought that I would have the patience run each model and
      adaptation method multiple times.
