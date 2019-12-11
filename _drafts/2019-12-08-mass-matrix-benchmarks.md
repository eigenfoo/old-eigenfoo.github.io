---
title: Benchmarks for Mass Matrix Adaptation
excerpt:
tags:
  - open source
  - pymc
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background6.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2019-12-09
search: false
---

{% if page.noindex == true %}
  <meta name="robots" content="noindex">
{% endif %}

I was lucky enough to be invited to attend the [Gradient
Retreat](https://gradientretreat.com/) this past week.

## Questions

A quick low-down for those unfamiliar: _tuning_ is what happens before sampling,
during which the goal is not to actually draw samples, but to _prepare_ to draw
samples. For HMC and its variants, this usually means estimating HMC parameters
such as the step size, integration time and mass matrix[^1], the last of which
is essentially the covariance matrix of the model parameters. Because life is
finite, this post will only focus on mass matrix adaptation.

The interesting thing about tuning is that there are no rules: there are no
asymptotic guarantees we can rely on and no mathematical results we can turn to
for enlightened inspiration. The only thing we care about is obtaining decent
estimates of the mass matrix, and preferably quickly.

Accompanying this lack of understanding of mass matrix adaptation is a
commensurate lack of scientific inquiry into adaptation - there is scant
literature to draw from, and for some open source developers, there is precious
little prior art to look to when writing new implementations of HMC!

So I decided to do some empirical legwork and benchmark various methods of mass
matrix adaptation. I benchmarked five different mass matrix adaptation methods,
testing each on six different models.

## Results

Without further ado, the main results are shown below. I make some general
observations on the benchmarks, and finally (for the loving readers who care)
describe my experimental setup and possible directions for further research.

### Tuning Times

The tuning time, in seconds, of each mass matrix adaptation method for each
model.

|            |**`mvnormal`**|**`lrnormal`**|**`stoch_vol`**|**`gp`**|**`eight`**|**`baseball`**
|:-----------|-------------:|-------------:|--------------:|-------:|----------:|------------:|
|**diag**    |        365.34|        340.10|         239.59|   18.47|       2.92|         5.32|
|**full**    |          8.29|        364.07|         904.95|   14.24|       2.91|         4.93|
|**diag_exp**|        358.50|        360.91|         219.65|   16.25|       3.05|         5.08|
|**full_exp**|          8.46|        142.20|         686.58|   14.87|       3.21|         6.04|
|**covadapt**|        386.13|         89.92|         398.08|     N/A|        N/A|          N/A|

### Effective Samples per Second

The number of effective samples, drawn by each mass matrix adaptation method for
each model.

|            |**`mvnormal`**|**`lrnormal`**|**`stoch_vol`**|**`gp`**|**`eight`**|**`baseball`**
|:-----------|-------------:|-------------:|--------------:|-------:|----------:|------------:|
|**diag**    |          0.02|          1.55|          11.22|   65.36|     761.82|       455.23|
|**full**    |          1.73|          0.01|           6.71|  106.30|     840.77|       495.93|
|**diag_exp**|          0.02|          1.51|           9.79|   59.89|     640.90|       336.71|
|**full_exp**|      1,799.11|      1,753.65|           0.16|  101.99|     618.28|       360.14|
|**covadapt**|          0.02|        693.87|           5.71|     N/A|        N/A|          N/A|

## Observations

> **tldr:** there are a lot of options, but sane defaults are good!

- A full mass matrix can provide significant speedups over a diagonal mass
  matrix, sometimes up to two orders of magnitude, for both the tuning time and
  the number of effective samples per second.
  - This is most noticeable in the `mvnormal` model, with heavily correlated
    parameters.
  - However, in models with less extreme correlations among parameters, this
    advantage shrinks significantly (although it doesn't go away entirely).
    Full matrices can also take longer to tune. You can see this in the baseball
    or eight schools model.
  - Nevertheless, full mass matrices never seem to perform egregiously _worse_
    than diagonal mass matrices.

- Having an “expanding” schedule (similar to what Stan does) for tuning can
  sometimes give better performance, but nowhere near as significant as the
  difference between diagonal and full matrices. This difference is most
  noticeable for the `mvnormal` and `lrnormal` models (probably because these
  models have a constant covariance matrix and so more careful estimates using
  expanding windows can provide much better sampling).

- `covadapt` seems to run into some numerical difficulties? While running these
  benchmarks I ran into an inscrutable and non-reproducible
  [`ArpackError`](https://stackoverflow.com/q/18436667) from SciPy.

## Experimental Setup

- All samplers were run for 2000 tuning steps and 1000 sampling steps. If the
  number of tuning steps seem high, it's because it is: neither PyMC nor Stan
  have those as their defaults. However, it's necessary to have it high enough
  for `covadapt` to work.

- I benchmarked against six models:

  1. A 100-dimensional multivariate normal with a non-diagonal covariance matrix (`mvnormal`)
  1. A 100-dimensional multivariate normal with a low-rank covariance matrix (`lrnormal`)
  1. A [stochastic volatility model](https://docs.pymc.io/notebooks/stochastic_volatility.html) (`stoch_vol`)
  1. The [eight schools model](https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html#The-Eight-Schools-Model) (`eight`)
  1. The [baseball model](https://docs.pymc.io/notebooks/hierarchical_partial_pooling.html) (`baseball`)
  1. A [Gaussian process](https://docs.pymc.io/notebooks/GP-SparseApprox.html#Examples) (`gp`)

  I think this is a decent assortment of models: between the six of them, there
  are some with correlated parameters (and not), large numbers of parameters,
  hierarchical pooling...

- I set `target_accept` (a.k.a. `adapt delta` to Stan users) to 0.9 to make all
  divergences go away (I'm actually fairly concerned how `target_accept = 0.9`
  yields some "false positive divergences" with the full mass matrix... This
  doesn't happen in Stan! But that is a question for another time.)

## Shortcomings and Directions for Further Inquiry

- There is some trickiness with the tuning time comparisons: it's entirely
  possible that some mass matrix estimates converge quicker than others, and so
  forcing them to tune for longer is unfair to such mass matrix adaptation
  routines.

- It's also entirely possible that the expansion schedule for the adaptation
  windows is just suboptimal! There's no reason why 1.1 should be an obvious
  number.

- These benchmarks are done only for very basic toy models: I should test more
  extensively on more models that people in the real world use. I’m actually not
  sure how I can improve on these benchmarks, asides from grabbing more models
  to test them out on. Any feedback welcome!

## Questions

- Is the assumption that the mass matrix is diagonal a good assumption to make?
  What are its implications for the wall time of tuning, and the number of
  effective samples per second?

- Does the scheduling (i.e. the multiplier each time) make a big difference?
  Compare multipliers of 1 and 2, and evaluate similarly to above.

- Besides assuming the mass matrix is diagonal, are there any other ways of
  simplifying mass matrix adaptation? For example, could we [approximate the mass
  matrix as low rank](https://github.com/aseyboldt/covadapt)?

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

[^1]: uh, _*sweats and looks around nervously for differential geometers*_
      more formally called the _metric_
