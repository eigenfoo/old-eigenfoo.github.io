---
mathjax: true
title: ~~Fruit~~ Loops and Learning - The LUPI Paradigm and SVM+
excerpt: "Learning Using Privileged Information: what it is, how to do it, and
why it's cool"
tags:
    - mathematics
    - machine learning
    - lupi
    - svm+
header:
  teaser: /assets/images/loops.jpeg
  overlay_image: /assets/images/loops.jpeg
  caption: "Photo credit: [wallpapercraze.com](http://wallpapercraze.com/images/wallpapers/fruitloops-441535.jpeg)"
  last_modified_at: 2018-01-30
---

Here's a short story you might know: you have a black box, whose name is
_Machine Learning Algorithm_. It's got two modes: training mode and testing
mode. You set it to training mode, and throw in a lot (sometimes _a lot_ a lot)
of ordered pairs $$(x_i, y_i), 1 \leq i \leq l $$. Here, the $$x_i$$ are called
the _examples_ and the $$y_i$$ are called the _targets_. Then, you set it to
testing mode and throw in some more examples, for which you don't have the
corresponding targets. You hope the $$y_i$$s that come out are in some sense
the "right" ones.

Generally speaking, this is a parable of _supervised learning_. However, Vapnik
(the inventor of the
[SVM](https://en.wikipedia.org/wiki/Support_vector_machine)) recently described
a new way to think about machine learning
([here](http://www.engr.uconn.edu/~jinbo/doc/vladimir_newparadiam.pdf) and
[here](http://jmlr.csail.mit.edu/papers/volume16/vapnik15b/vapnik15b.pdf)):
_learning using privileged information_, or _LUPI_ for short.

This post is meant to introduce the LUPI paradigm of machine learning to
people who are generally familiar with supervised learning and SVMs, and are
interested in seeing the math and intuition behind both things extended to the
LUPI paradigm.

## What is LUPI?

The main idea is that instead of two-tuples $$(x_i, y_i)$$, the black box is fed
three-tuples $$(x_i, x_i^{*}, y_i) $$, where the $$x^{*}$$s are the so-called
_privileged information_ that is only available during training, and not during
testing. The hope is that this information will train the model to better
generalize during the testing phase.

Vapnik offers many examples in which LUPI can be applied in real life: in
bioinformatics and proteomics (where advanced biological models, which the
machine might not necessarily "understand", serve as the privileged
information), in financial time series analysis (where future movements of the
time series are the unknown at prediction time, but are available
retrospectively), and in the classic MNIST dataset, where the images were
converted to a lower resolution, but each annotated with a "poetic description"
(which was available for the training data but not for the testing data).

Vapnik's team ran tests on well-known datasets in all three application areas
and found that his newly-developed LUPI methods performed noticeably better than
classical SVMs in both convergence time (i.e. the number of examples necessary
to achieve a certain degree of accuracy) and estimation of a good predictor
function.  In fact, Vapnik's proof-of-concept experiments are so whacky that
they actually [make for an entertaining read
](https://nautil.us/issue/6/secret-codes/teaching-me-softly)!

## Classical SVMs (separable and non-separable case)

There are many ways of thinking about SVMs, but I think that the one that is
most instructive here is to think of them as solving the following optimization
problem:

> Minimize $$ \frac{1}{2} \|w\|^2 $$
>
> subject to $$ y_i [ w \cdot x_i + b ] \geq 1, \; 1 \leq i \leq l $$.

Basically all this is saying is that we want to find the hyperplane that
separates our data by the maximum margin. More technically speaking, this finds
the parameters ($$w$$ and $$b$$) of the maximum margin hyperplane, with $$l_2$$
regularization.

In the non-separable case, we concede that our hyperplane may not classify all
examples perfectly (or that it may not be desireable to do so: think of
overfitting), and so we introduce a so-called _slack variable_ $$ \xi_i \geq 0
$$ for each example $$i$$, which measures the severity of misclassification of
that example. With that, the optimization becomes:

> Minimize $$ \frac{1}{2} \|w\|^2 + C\sum_{i=1}^{l}{\xi_i} $$
>
> subject to $$ y_i [ w \cdot x_i + b ] \geq 1 - \xi_i, \; \xi_i \geq 0, 1
> \leq i \leq l $$.

where $$C$$ is some regularization parameter.

This says the same thing as the previous optimization problem, but now allows
points to be (a) classified properly ($$\xi_i = 0$$), (b) within the margin but
still classified properly ($$0 < \xi_i < 1$$), or (c) misclassified
($$1 \leq \xi_i$$).

In both the separable and non-separable cases, the decision rule is simply $$
\hat{y} = \text{sign}(w \cdot x + b) $$.

An important thing to note is that, in the separable case, the SVM uses $$l$$
examples to estimate the $$n$$ components of $$w$$, whereas in the nonseparable
case, the SVM uses $$l$$ examples to estimate $$n+l$$ parameters: the $$n$$
components of $$w$$ and $$l$$ values of slacks $$\xi_i$$.  Thus, in the
non-separable case, the number of parameters to be estimated is always larger
than the number of examples: it does not matter here that most of slacks may be
equal to zero: the SVM still has to estimate all of them.

The way both optimization problems are actually _solved_ is fairly involved (they
require [Lagrange
multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier)), but in terms
of getting an intuitive feel for how SVMs work, I think that examining the
optimization problems suffice!

## What is SVM+?

In his paper introducing the LUPI paradigm, Vapnik outlines _SVM+_, a
modified form of the SVM that fits well into the LUPI paradigm, using privileged
information to improve performance. It should be emphasized that LUPI is a
paradigm - a way of thinking about machine learning - and not just a collection
of algorithms. SVM+ is just one technique that interoperates with the LUPI
paradigm.

The innovation of the SVM+ algorithm is that is uses the privileged information
to estimate the slack variables. Given the training three-tuple $$ (x, x^{*}, y)
$$, we map $$x$$ to the feature space $$Z$$, and $$x^{*}$$ to a separate feature
space $$Z^{*}$$. Then, the decision rule is $$ \hat{y} = \text{sign}(w \cdot x +
b) $$ and the slack variables are estimated by $$ \xi = w^{*} \cdot x^{*} + b^{*}
$$.

In order to find $$w$$, $$b$$, $$w^{*}$$ and $$b^{*}$$, we solve the following
optimization problem:

> Minimize $$ \frac{1}{2} (\|w\|^2 + \gamma \|w^{*}\|^2) +
> C \sum_{i=1}^{l}{(w^{*} \cdot x_i^{*} + b^{*})} $$
> 
> subject to $$ y_i [ w \cdot x_i + b ] \geq 1 - (w^{*} \cdot x^{*} + b^{*}),
> \; (w^{*} \cdot x^{*} + b^{*}) \geq 0, 1 \leq i \leq l $$.

where $$\gamma$$ indicates the extent to which the slack estimation should be
regularized in comparison to the SVM. Notice how this optimization problem is
essentially identical to the non-separable classical SVM, except the slacks
$$\xi_i$$ are now estimated with $$w^{*} \cdot x^{*} + b^{*}$$.

Again, the method of actually solving this optimization problem involves
Lagrange multipliers and quadratic programming, but I think the intuition is
captured in the optimization problem statement.

## Interpretation of SVM+

The SVM+ has a very ready interpretation. Instead of a single feature space, it
has two: one in which the non-privileged information lives (where decisions are
made), and one in which the privileged information lives (where slack variables
are estimated).

But what's the point of this second feature space? How does it help us? Vapnik
terms this problem _knowledge transfer_: it's all well and good for us to learn
from the privileged information, but it's all for naught if we can't use this
newfound knowledge in the test phase.

The way knowledge transfer is resolved here is by assuming that _examples in the
training set that are hard to separate in the privileged space, are also hard to
separate in the regular space_. Therefore, we can use the privileged information
to obtain an estimate for the slack variables.

Of course, SVMs are a technique with many possible interpretations, of which my
presentation (in terms of the optimization of $$w$$ and $$b$$) is just one. For
example, it's possible to think of SVMs in terms of kernels functions, or as
linear classifiers minimizing hinge loss. In all cases, it's possible and
worthwhile to understand that interpretation of SVMs, and how the LUPI paradigm
contributes to or extends that interpretation. I'm hoping to write a piece later
to explain these exact topics.

Vapnik also puts a great emphasis on analyzing SVM+ based on its statistical
learning theoretic properties (in particular, analyzing its rate of convergence
via the [VC dimension](https://en.wikipedia.org/wiki/VC_dimension)). Vapnik was
one of the main pioneers behind statistical learning theory, and has written an
[entire
book](https://www.amazon.com/Statistical-Learning-Theory-Vladimir-Vapnik/dp/0471030031)
on this stuff ~~which I have not read~~, so I'll leave that part aside for now. I
hope to understand this stuff one day.

## Extensions to SVM+

In his paper, Vapnik makes it clear that LUPI is a very general and abstract
paradigm, and as such there is plenty of room for creativity and innovation -
not just in researching and developing new LUPI methods and algorithms, but also
in implementing and applying them. It is unknown how to best go about supplying
privileged information so as to get good performance. How should the data be
feature engineered? How much signal should be in the privileged information?
These are all open questions.

Vapnik himself opens up three avenues to extend the SVM+ algorithm:

1. _a mixture model of slacks:_ when slacks are estimated by a mixture of a
   smooth function and some prior
2. _a model where privileged information is available only for a part of the
   training data:_ where we can only supply privileged information on a small
   subset of the training examples
3. _multiple-space privileged information:_ where the privileged information we
   can supply do not all share the same features

Clearly, there's a lot of potential in the LUPI paradigm, as well as a lot of
reasons to be skeptical. It's very much a nascent perspective of machine
learning, so I'm interested in keeping an eye on it going forward. I'm hoping
to write more posts on LUPI in the future!

**I'd love to hear any feedback on this post! Send me a tweet or an email.**
