---
title: Benchmarking Mass Matrix Adaptation
excerpt:
tags:
  - pymc
  - stan
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background4.png
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

- Does full vs diagonal mass matrices affect wall time of warmup, and time per
  effective sample?
  - The answer is probably yes.
- Does the scheduling (i.e. the multiplier each time) make a big difference?
  Compare multipliers of 1 and 2, and evaluate similarly to above.
  - Apparently? This is true for the baseball model, but not really for the
    MvNormal or GP (which also have correlated parameters)
- What about Adrian's `covadapt`?

## Results

So without further ado...

### Mean Tuning Times in Seconds

|            |**`mvnormal`**|**`lrnormal`**|**`stoch_vol`**|**`gp`**|**`eight`**|**`baseball`**
|:-----------|-------------:|-------------:|--------------:|-------:|----------:|------------:|
|**diag**    |        365.34|        340.10|         239.59|   18.47|       2.92|         5.32|
|**full**    |          8.29|        364.07|         904.95|   14.24|       2.91|         4.93|
|**diag_exp**|        358.50|        360.91|         219.65|   16.25|       3.05|         5.08|
|**full_exp**|          8.46|        142.20|         686.58|   14.87|       3.21|         6.04|
|**covadapt**|        386.13|         89.92|         398.08|     N/A|        N/A|          N/A|

### Mean Effective Samples per Second

|            |**`mvnormal`**|**`lrnormal`**|**`stoch_vol`**|**`gp`**|**`eight`**|**`baseball`**
|:-----------|-------------:|-------------:|--------------:|-------:|----------:|------------:|
|**diag**    |          0.02|          1.55|          11.22|   65.36|     761.82|       455.23|
|**full**    |          1.73|          0.01|           6.71|  106.30|     840.77|       495.93|
|**diag_exp**|          0.02|          1.51|           9.79|   59.89|     640.90|       336.71|
|**full_exp**|      1,799.11|      1,753.65|           0.16|  101.99|     618.28|       360.14|
|**covadapt**|          0.02|        693.87|           5.71|     N/A|        N/A|          N/A|

## Observations

As of the end of gradient retreat:

- A full mass matrix can provide significant speedups over a diagonal mass
  matrix, sometimes up to 100x, for both the tuning time and the number of
  effective samples per second.
  - This is most noticeable in models with heavily correlated parameters (such
    as the toy multivariate normal models).
  - However, in models with less extreme correlations among parameters, full
    mass matrices do not offer compellingly higher effective samples per second,
    and also take longer to tune. You can see this in the baseball or eight
    schools model, which have weaker correlation among parameters.
  - Nevertheless, full mass matrices never perform _worse_ than diagonal mass
    matrices.

- FIXME: Having an “expanding” schedule (similar to what Stan does) for tuning
  also gives better performance, but nowhere near as significant as the
  diagonal/full boost. The max I've seen is 2x boost on effective samples per
  second.

- There are some instances where `full_expanding` (and perhaps `covadapt`) lead
  to divergences while sampling: specifically, the stochastic volatilitye and
  low-rank multivariate normal models. I doubled the number of tuning draws
  here: so we can't compare apples to apples (a rough heuristic is to halve the
  tuning times, but this isn't perfect since tuning gets faster as tuning
  proceeds).

- I get an inscrutable [`ArpackError`](https://stackoverflow.com/q/18436667)
  with `covadapt` sometimes.

- These benchmarks are done only for very basic toy models: I should test more
  extensively on more models that people in the real world use. I’m actually not
  sure how I can improve on these benchmarks, asides from grabbing more models
  to test them out on. Any feedback welcome!

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

Caveats with this setup:

- There is some trickiness with the tuning time comparisons: it's entirely
  possible that some mass matrix estimates converge quicker than others, and so
  forcing them to tune for longer is unfair to such mass matrix adaptation
  routines.
- It's also entirely possible that the expansion schedule for the adaptation
  windows is just suboptimal! There's no reason why 1.1 should be an obvious
  number.

tldr: there are a lot of options, but sane defaults are good!

## Resources and Further Reading

- [Colin Carroll's talk on HMC
  tuning](https://colcarroll.github.io/hmc_tuning_talk/)
- [Stan reference manual of HMC algorithm
  parameters](https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html)
- [Dan Foreman-Mackey's blog post on dense mass matrices in
  PyMC3](https://dfm.io/posts/pymc3-mass-matrix/)
- [Adrian Seyboldt's low-rank mass matrix
  approximations](https://github.com/aseyboldt/covadapt)
