---
title: "Cookbook - Managing the Bias-Variance Tradeoff"
excerpt: "The bias-variance tradeoff is a unique result in machine learning: it
sits on extremely solid theoretical foundations, and has a ludicrously
far-reaching scope of applicability."
tags:
  - machine learning
  - cookbook
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background10.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
toc: true
toc_sticky: true
toc_label: "Recipes"
toc_icon: "utensils"
last_modified_at: 2018-04-13
---

The bias-variance tradeoff is a unique result in machine learning: it sits on
extremely solid theoretical foundations, and has a ludicrously far-reaching
scope of applicability. Which might explain why it's treated so...
interestingly... in introductory machine learning courses (or at least, my
introductory course). Quickly go over the proof (it's easy enough that even
undergraduates can digest it), build some intuition about what it means for
real-life predictors, and move on.

Enough has been said about what the bias-variance tradeoff (a.k.a. the
bias-variance decomposition) _is_ (read
[this excellent essay](http://scott.fortmann-roe.com/docs/BiasVariance.html) if
you're shaky). However, not that much has been said about how to _manage_ the
bias or variance of your model, which seems to me to be infinitely more
important to know. There's neat advice scattered thinly throughout
the interwebs, but I figure I'd consolidate them in one place. I'd also highly
recommend [these slides](http://cs229.stanford.edu/materials/ML-advice.pdf) from
Andrew Ng at Stanford, in which he outlines some practical advice for machine
learning models. I basically recreated some of his graphs for two of the three
images below.

<img align="middle" src="https://raw.githubusercontent.com/eigenfoo/eigenfoo.xyz/master/assets/images/bias-variance.png">

Of course, take this with a pinch of salt: the most important thing in
statistical modelling is not overbearing guidelines or rigid dogma, but the
situation you have at hand. Clear thinking is always good!

## High Variance

<img align="middle" src="https://raw.githubusercontent.com/eigenfoo/eigenfoo.xyz/master/assets/images/high-variance.png">

### Diagnostics

- The testing error looks like it continues to decrease with the
  training set size, suggesting that more data (specifically, more examples)
  will help.

- There is a significant difference between the training and testing errors.

### Remedies

- Try to obtain more examples.

- Perform some feature selection or feature engineering to get a high-quality
  set of features.

- Considering ensembling (e.g. bagging or boosting your predictors).

- Consider regularizing your model (or pruning it, or otherwise enforcing
  model parsimony).

- Consider using a less flexible model: perhaps a [parametric (or even
  linear!)](https://www.youtube.com/watch?v=68ABAU_V8qI) model will suffice for
  your purposes?

## High Bias

<img align="middle" src="https://raw.githubusercontent.com/eigenfoo/eigenfoo.xyz/master/assets/images/high-bias.png">

### Diagnostics

- Even the training error is unacceptably high.

- The training and testing errors quickly converge to a common value.

### Remedies

- Try to obtain/engineer more features (consider polynomial or interaction terms).

- Consider using a more flexible model: perhaps nonlinear or even nonparametric
  models would work better for your purposes?
