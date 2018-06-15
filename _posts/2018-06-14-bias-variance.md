---
title: "Recipes for Managing the Bias-Variance Tradeoff"
excerpt:
tags:
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background10.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2018-06-14
---

The bias-variance tradeoff is a unique result in machine learning: it is both
widely applicable and has solid mathematical foundations (its derivation is
easily digestible by undergraduates).

Enough has been said about what the bias-variance tradeoff (a.k.a. the
bias-variance decomposition) _is_ (see
[this excellent essay](http://scott.fortmann-roe.com/docs/BiasVariance.html) if
you're shaky). Not that much has been said about how to _manage_ the bias or
variance of your model. There's neat advice scattered thinly throughout the web,
but I figure I'd consolidate them in one place. I'd also highly recommend [these
slides](http://cs229.stanford.edu/materials/ML-advice.pdf) from Andrew Ng at
Stanford, which I basically replicated for two of the three images below.

<img align="middle" src="https://raw.githubusercontent.com/eigenfoo/eigenfoo.xyz/master/assets/images/bias-variance.png">

Of course, take this with a pinch of salt: the most important thing in
statistical modelling is not guidelines or dogma, but the situation you have at
hand. Clear thinking is good!

## Recipe #1: High Variance

<img align="middle" src="https://raw.githubusercontent.com/eigenfoo/eigenfoo.xyz/master/assets/images/high-variance.png">

### To Diagnose:
- Testing error seems to decrease with the training test size, suggesting that
  more data will help.
- Significant difference between the training and testing errors.

### To Remedy:
- Try to obtain more examples.
- Perform some feature selection or feature engineering to get a high-quality
  set of features.
- Considering ensembling (e.g. bagging or boosting your predictors).
- Consider regularizing your model.
- Consider using a less flexible model: perhaps a parametric or even linear
  model will suffice for your purposes?

## Recipe #2: High Bias

<img align="middle" src="https://raw.githubusercontent.com/eigenfoo/eigenfoo.xyz/master/assets/images/high-bias.png">

### To Diagnose:
- Even the training error is unacceptably high.
- The training and testing errors quickly converge to a common value.

### To Remedy:
- Try to obtain/engineer more features (consider polynomial or interaction terms).
- Consider using a more flexible model: perhaps nonlinear or even nonparametric
  models would be a better model?
