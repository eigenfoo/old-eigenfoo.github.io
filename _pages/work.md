---
layout: single
permalink: /work/
toc: true
toc_sticky: true
title: Work
---

## Software

Here are some open source software libraries that I've made substantial
contributions to.

### [PyMC3](https://github.com/pymc-devs/pymc3) and [PyMC4](https://github.com/pymc-devs/pymc4)

![PyMC3 logo](../assets/images/pymc-logo.png){: .align-right}

PyMC3 is a popular Python framework for Bayesian modeling and probabilistic
machine learning, focusing on Markov chain Monte Carlo (MCMC) and variational
inference (VI) algorithms. I'm a member of the core development team, and
contribute to the PyMC3 internals and documentation. I wrote [a blog post on
tips and tricks for Bayesian modelling using
PyMC3](https://www.georgeho.org/bayesian-modelling-cookbook/).

### [Pyfolio](https://github.com/quantopian/pyfolio) and [Alphalens](https://github.com/quantopian/alphalens)

<figure class="half">
    <img src="/assets/images/pyfolio-logo.png" alt="Pyfolio logo">
    <img src="/assets/images/alphalens-logo.png" alt="Alphalens logo">
</figure>

Pyfolio and Alphalens are Python libraries for risk analysis and performance
attribution of financial portfolios, and alpha factor research for algorithmic
trading. Both libraries are fully integrated into the [Quantopian
platform](http://quantopian.com/).

I developed the risk and performance attribution capabilities of Pyfolio (read
more on [my blog post here](https://www.georgeho.org/pyfolio/)), and help maintain
the library. I help develop new features, triage bug reports and troubleshoot
issues for Alphalens.

### [Knead](https://github.com/font-bakers/knead) and [Glaze](https://github.com/font-bakers/glaze)

<figure class="half">
    <img src="https://raw.githubusercontent.com/font-bakers/knead/master/docs/img/logo.png" alt="Knead logo">
    <img src="https://raw.githubusercontent.com/font-bakers/glaze/master/docs/img/logo.png" alt="Glaze logo">
</figure>

Knead and Glaze are command line tools and Python libraries for preprocessing,
manipulating, rendering and visualizing font files and algorithmically-generated
typefaces. Both libraries were written as internal tools for [The Font
Bakers](https://font-bakers.github.io/), to support their research and
development workflow. I developed and maintain both projects.

## Datasets

### [`cryptics.eigenfoo.xyz`](https://cryptics.eigenfoo.xyz/)

`cryptics.eigenfoo.xyz` is a dataset of [cryptic
crossword](https://www.newyorker.com/puzzles-and-games-dept/cryptic-crossword/reintroducing-the-new-yorkers-cryptic-crossword)
clues, indicators and charades, collected from various blogs and publicly
available digital archives.

The project scrapes several blogs and digital archives for cryptic crosswords.
Out of these collected web pages, the clues, answers, clue numbers, blogger’s
explanation and commentary, puzzle title and publication date are all parsed
and extracted into a tabular dataset. The result is _over half a million clues_
from cryptic crosswords over the past twelve years.

## Projects

Here are some interesting machine learning and data science projects I've
pursued.

### [Deep Learning for Algorithmic Type Design](https://font-bakers.github.io/)

As a research project pursued in my last year at The Cooper Union, we researched
a class- and attribute- conditional generative adversarial network capable of
producing vector graphics, with potential applications in algorithmic type
design. The generative model was to produce closed shapes, with counter spaces,
defined by a variable number of quadratic Bézier curves. At the time, no
generative model with vector graphic output has previously appeared in
literature.

### [Hate Speech on Reddit](https://www.georgeho.org/reddit-clusters/)

As part of a project on [Data Science for Social
Good](http://ee.cooper.edu/~keene/dssg.html), I ran text clustering algorithms
on well-known hateful and toxic subreddits, and collaborated with a
cross-disciplinary team of artists, architects and engineers to present the
findings at [The Cooper Union 2018 End of Year
Show](https://cooper.edu/events-and-exhibitions/exhibitions/2018-student-exhibition).
I also wrote [a blog post on my results](https://www.georgeho.org/reddit-clusters/),
and [gave a talk on the data science that went into the
project](https://www.georgeho.org/reddit-slides/).
