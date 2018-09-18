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

A [fairly well-known](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).
[matrix factorization](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
[technique](https://arxiv.org/abs/1401.5226).

Factorize your (entrywise non-negative) $$m \times n$$ matrix $$V$$ as
$$V = WH$$, where $$W$$ is $$m \times p$$ and $$H$$ is $$p \times n$$. $$p$$
is the dimensionality of your latent space, and each latent dimension usually
comes to measure something semantic. There are several algorithms to find this
factorization, but Lee and Seung's multiplicative update rule[^4] is most
popular.

Enough said, I think.

### Probabilistic Matrix Factorization (PMF)



### Bayesian Probabilistic Matrix Factorization (BPMF)


---


[^1]: Aggarwal, Charu C, and ChengXiang Zhai. “A Survey of Text Clustering Algorithms.” Mining Text Data, Springer, 2014, pp. 77–128. ([http://charuaggarwal.net/text-cluster.pdf](http://charuaggarwal.net/text-cluster.pdf))

[^2]: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf

[^3]: https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf

[^4]: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
