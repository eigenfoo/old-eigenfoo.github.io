---
title: Machine Learning Mini-Syllabus
excerpt: "a.k.a. \"A Working Knowledge of Machine Learning for People Working a
Full-Time Job.\""
tags:
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background4.png
  overlay_filter: 0.05
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-01-04
mathjax: true
toc: true
toc_sticky: true
toc_label: "Yo, machines can learn?!"
toc_icon: "robot"
search: false
---

{% if page.noindex == true %}
  <meta name="robots" content="noindex">
{% endif %}

Welcome! Some notes before we begin:

- I'm assuming that you _don't_ need to learn how to actually program any of
  this (i.e. I assume that once you understand the actual ML, you can learn to
  code it yourself).
- This is mostly machine learning (not data science). [A lot of data science is
  not machine
  learning](https://veekaybee.github.io/2019/02/13/data-science-is-different/).

## Core Material

### Review of linear algebra and probability

- You should probably review in two or three passes, in increasing order of
  depth and difficulty.
- If you find yourself stumbling on some concepts, they're probably advanced and
  kind of not necessary (e.g. sufficient statistics, the exponential family, the
  formalization of probability theory with probability/measure spaces). The
  point of all of this is to get a _working knowledge_ of linear algebra and
  probability, not to rehash MA326 and MA224.
- I am skipping statistics here. Large swathes of undergraduate statistics are
  kind of irrelevant to machine learning, and the parts that are important
  (maximum likelihood methods, Bayesian statistics, linear models) are mostly
  covered somewhere below.

1. The absolute basics: literally work through anything you find online. Like,
   Khan Academy or something. You just want to know the shape of the ideas here.
   Optionally, chapters 2 and 3 of _Deep Learning_[^3] is probably a nice
   refresher.
1. Chapters 4 and 6 ("Matrix Decompositions" and "Probability and
   Distributions") of _Mathematics for Machine Learning_[^5]. If you're
   wondering why I don't prescribe Chapter 2, it's because the concepts are
   pretty pure linear algebra, and are nice to know but not _strictly_ necessary
   for machine learning, IMHO.
1. This last part is totally optional and you should frankly probably skip it.
   Chapter 2 of _Pattern Recognition and Machine Learning_[^1], which covers
   probability theory. I would consider it an accomplishment if you can
   understand most of Section 2.3 ("The Gaussian Distribution"), which would
   also be the most useful. Bishop's discussion of the Gaussian distribution
   also involves a good deal of linear algebra, so it's kind of two bird in one
   stone?

### High level overview of ML (i.e. the lay of the land)

- Chapter 1 of Bishop[^1] is a good introduction here. Read it all, even the
  math (this might be kind of Puritan, but I think the math in this chapter
  is pretty non-negotiable).

### "Classical" machine learning

- So there's a zoo of ML algorithms that you should know. This following
  menagerie should get you started.
- You can read the `scikit-learn` documentation[^2] for a good overview/guide to
  these algorithms.
- There is a heavy supervised learning bias in the following selection.

1. Linear models for regression
    - Linear regression
    - Regularization: L2 and L1
1. Linear models for classification
    - Logistic regression
    - Decision trees and gradient-boosted trees
    - $$k$$ nearest neighbors
    - Support vector machines
1. Model selection and evaluation
    - $$k$$-fold cross validation
    - Metrics for model evaluation: precision, recall, F1, ROC curve, etc.
1. Ensemble methods
    - Bootstrapping and bagging
    - Boosting

- If you have time (and maybe even if you don't), you should look through the
  following.
- There's less of a focus on supervised learning, and more

1. Feature preprocessing/engineering
1. Clustering techniques
1. Dimensionality reduction techniques
1. Novelty/outlier detection
1. Kernel methods (Bishop[^1] is the best reference on these that I've found)

### Deep learning

- I feel like most places expect you to have a basic understanding of this. And
  even if that wasn't true, I think deep learning has captured so much of the
  public discourse that you need to be able to speak their language.
- If you want to ease your way into it, you can go through a book to get a basic
  understanding of deep learning[^4].  Although most MOOCs cover this stuff
  anyway.
- After that, I think I'd recommend working though a MOOC[^6] or something
  similar.
- In particular, you should have a good grip on:

1. Stochastic gradient descent and deep learning computation
1. Multi-layered perceptrons
1. Convolutional neural networks

- Time permitting, you could also look into the following (although you could
  probably spend a lifetime fully learning these things, so be careful how you
  spend your time!):

1. Recurrent neural networks
1. Sequence models
1. Attention mechanisms
1. Tasks and models in natural language processing
1. Tasks and models in computer vision

## Selected Advanced Topics

After going through the core stuff you might want to choose one or two
"subdisciplines" within ML to specialize it, or get a broad understanding of a
lot of them. I've ended up specializing heavily in the first one, but I can
probably give some pointers for most of them.

1. Probabilistic programming and Bayesian machine learning
1. Natural language processing
1. Computer vision
1. Recommendation engines

Here are also some subdisciplines that are interesting but I don't think anybody
in the real world uses (yet):

1. Representation learning
1. Reinforcement learning

## Non-Machine Learning

Some non-machine learning stuff that will be critical to have a good grip on.

- Git
- SQL
- Python (and I mean, _really_ learn Python)
  - What are "dunder" methods?
  - Do Python classes have notions of constructors, destructors and virtual
    classes?
  - What is a generator, and how is it related to the `yield` keyword?
  - How is unit testing done in Python?
  - Can you make your own pip-installable Python package?
  - Python wasn't designed for heavy-duty numerical computation, so how/why is
    all our machine learning done in Python?
- Get familiar with popular libraries or frameworks for ML
  - NumPy, Pandas and Matplotlib are pretty universal.
  - Beyond that, what frameworks/libraries you want to learn depends on what you
    want to do.
- Containers (e.g. Docker)

## Resources

All resources I recommended above are referenced below as footnotes. Everything
is freely available online (which was important to me when writing this).

That being said, if you're willing to spend a bit of money, one of my PyMC
colleagues has [put together a nice
list](https://sedar.co/posts/build-you-a-library/) of machine learning books,
which you should check out.

---

[^1]: [Bishop's _Pattern Recognition and Machine
      Learning_](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
      (a.k.a. PRML). A classic of ML: fastidiously comprehensive, robustly
      mathematical, takes no prisoners. I found this book really useful because
      I could follow the math, but YMMV if you think math isn't the best way to
      explain concepts.

[^2]: [`scikit-learn` User
      Guide](https://scikit-learn.org/stable/user_guide.html). Super
      comprehensive dictionary for ML algorithms. The stuff here is probably
      enough to fill an intro undergrad course in ML.

[^3]: [_Deep Learning_ by Goodfellow, Bengio and
      Courville](https://www.deeplearningbook.org/). A pretty math-heavy guide
      of deep learning, but it's good. It's one of the few books that
      discussions deep learning divorced from any code (for better or for
      worse).

[^4]: [_Neural Networks and Deep Learning_ by Michael
      Nielsen](http://neuralnetworksanddeeplearning.com/). A gentler
      introduction to deep learning (stops at multilayer perceptrons, though).
      You might find this an easier introduction to deep learning, but it
      definitely won't take you far enough.

[^5]: [_Mathematics for Machine Learning_ by Deisenroth, Aldo and
      Ong](https://mml-book.github.io/). I haven't read this book, but I hear
      good things about it, and a brief skim makes me feel good about how they
      present the material.

[^6]: [fast.ai](https://www.fast.ai/) by Howard, Thomas and Gugger is probably
      one of the more famous deep learning MOOCs out there. They take a much
      more hands-on approach, covering applications of deep learning before
      going over the foundations. They've also written a whole bunch of open
      source Python libraries for deep learning or deep learning pedagogy, which
      makes me like their style :smiley:
