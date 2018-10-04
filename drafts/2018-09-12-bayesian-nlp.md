---
title: Bayesian Matrix Factorization for Text Clustering
excerpt:
tags:
  - natural language processing
  - bayesianism
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background4.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2018-09-12
---

Natural language processing is in a curious place right now. It's not
immediately obvious how successful it's been, or how close the field is to
viable, production-ready techniques (in the same way that, say, [computer vision
is](https://clarifai.com/models/)). For example, [Sebasian
Ruder](https://ruder.io) predicted that the field is [close to a watershed
moment](https://thegradient.pub/nlp-imagenet/), and that within a year we'll
have downloadable language models (that was six months ago, and, uh...).
However, [Ana Marasović](https://amarasovic.github.io/) points out that there is
[a tremendous amount of work demonstrating
that](https://thegradient.pub/frontiers-of-generalization-in-natural-language-processing/)

> "despite good performance on benchmark datasets, modern NLP techniques are
> nowhere near the skill of humans at language understanding and reasoning when
> making sense of novel natural language inputs".

I am confident that I am very bad at making lofty predictions about the future.
Instead, I'll try to talk about something I know a bit about: simple ideas and
Bayesian methods! This blog post will summarize some literature on Bayesian
matrix factorization methods, keeping an eye out for applications in text
clustering.

I'd like to talk about one application of Bayesiansm to a
specific task in NLP: text clustering. It's is exactly what it sounds like, and
there's been a good amount of success in applying text clustering to many other
NLP tasks:

- [Document organization](https://www-users.cs.umn.edu/~hanxx023/dmclass/scatter.pdf)
- [Corpus](http://jmlr.csail.mit.edu/papers/volume3/bekkerman03a/bekkerman03a.pdf)
  [summarization](https://www.cs.technion.ac.il/~rani/el-yaniv-papers/BekkermanETW01.pdf)
- [Document classification](http://www.kamalnigam.com/papers/emcat-aaai98.pdf)

So how are we 
Matrix factorizations, a severely underappreciated [technique in machine
learning](http://scikit-learn.org/stable/modules/decomposition.html).

---

One question is: why is matrix factorization a good technique to use for text
clustering? Because it is both a clustering and a feature engineering technique:
it gives us a better view of the latent space! Matrix factorization lives an
interesting double life: clustering technique by day, feature transformation
method by night. (Aggarwal and Zhai[^1] suggest that chaining matrix
factorization with some other clustering technique - e.g. agglomerative
clustering or topic modelling - is common practice and is called _concept
decomposition_, but I haven't seen any other source back this up.)

---

Some papers I read.

### Non-Negative Matrix Factorization (NMF)

A [fairly
well-known](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).
(but non-Bayesian!) [matrix
factorization](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
[technique](https://arxiv.org/abs/1401.5226).

Factorize your (entrywise non-negative) $$m \times n$$ matrix $$V$$ as
$$V = WH$$, where $$W$$ is $$m \times p$$ and $$H$$ is $$p \times n$$. $$p$$
is the dimensionality of your latent space, and each latent dimension usually
comes to measure something semantic. There are several algorithms to find this
factorization, but Lee and Seung's multiplicative update rule[^4] is most
popular.

Enough said, I think.

### Probabilistic Matrix Factorization (PMF)

Originally introduced as a paper at [NIPS
2007](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization),
_probabilistic matrix factorzation_ is  essentially the exact same model as NMF,
but with uncorrelated (a.k.a. "spherical") multivariate Gaussian priors placed
on the rows and columns of $$U$$ and $$V$$. Expressed as a graphical model, PMF
would look like this:

<figure>
    <a href="/assets/images/pmf.png"><img style="float: middle" src="/assets/images/pmf.png"></a>
</figure>

Note that the priors are placed on the _rows_ of the $$U$$ and $$V$$ matrices.

The authors then (somewhat disappointing) proceed to find the MAP estimate of
the $$U$$ and $$V$$ matrices. They show that maximizing the posterior is
equivalent to minimizing the sum-of-squared-errors loss function with two
quadratic regularization terms:

$$\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{M} {I_{ij} (R_{ij} - U_i^T V_j)^2} +
\frac{\lambda_U}{2} \sum_{i=1}^{N} |U|_{Fro}^2 +
\frac{\lambda_V}{2} \sum_{j=1}^{M} |V|_{Fro}^2 $$

where $$I_{ij}$$ is 1 if document $$i$$ contains word $$j$$, and 0 otherwise.

This loss function can be minimized via gradient descent, and implemented in
your favorite deep learning framework (e.g. Tensorflow or PyTorch).

The problem with this approach is that while the MAP estimate is often a
reasonable point in low dimensions, it becomes very strange in high dimensions,
and is usually not informative or special in any way. Read [Ferenc Huszár’s blog
post](https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/)
for more.

### Bayesian Probabilistic Matrix Factorization (BPMF)

Strictly speaking, PMF is not a Bayesian model. After all, there aren't any
priors or posteriors, only fixed hyperparameters and a MAP estimate. _Bayesian
probabilistic matrix factorization_ is a fully Bayesian treatment of PMF.

Instead of saying that the rows/columns of U and V are normally distributed with
zero mean and some precision matrix, we place hyperpriors on the mean vector and
precision matrices. The specific priors are Wishart priors on the covariance
matrices (with scale matrix $$W_0$$ and $$\nu_0$$ degrees of freedom), and
Gaussian priors on the means (with mean $$\mu_0$$ and and covariance equal to
the covariance given by the Wishart prior). Expressed as a graphical model, BPMF
would look like this:

<figure>
    <a href="/assets/images/bpmf.png"><img style="float: middle" src="/assets/images/bpmf.png"></a>
</figure>

Note that, as above, the priors are placed on the _rows_ of the $$U$$ and $$V$$
matrices, and that $$n$$ is the dimensionality of latent space (i.e. the number
of latent dimensions in the factorization).

The authors then sample from the posterior distribution of $$U$$ and $$V$$ using
a Gibbs sampler. Sampling takes several hours: somewhere between 5 to 180,
depending on how many samples you want. Nevertheless, the authors demonstrate
that BPMF can achieve more accurate and more robust results on the Netflix data
set.

I would propose several changes:
 - Use an LKJ prior on the covariance matrices instead of a Wishart prior. [This
   is known to be more numerically
   stable](https://docs.pymc.io/notebooks/LKJ.html).
 - Use a more robust sampler such as NUTS, or perhaps variational inference.
   this is so we can achieve results faster, and iterate quicker.

---


[^1]: Aggarwal, Charu C, and ChengXiang Zhai. "A Survey of Text Clustering Algorithms." Mining Text Data, Springer, 2014, pp. 77–128. ([http://charuaggarwal.net/text-cluster.pdf](http://charuaggarwal.net/text-cluster.pdf))

[^2]: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf

[^3]: https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf

[^4]: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
