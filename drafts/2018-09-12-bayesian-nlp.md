---
title: Bayesian Matrix Factorizations for Text Clustering
excerpt:
tags:
  - natural language processing
  - bayesianism
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background4.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2018-09-12
---

Natural language processing is in a remarkably curious state right now. It's not
clear to anyone how successful it's been, or how close the field is to viable,
production-ready techniques (in the same way that, say, computer vision now has
asdfasdf).

For example, [Sebasian Ruder](https://ruder.io) predicted that the field is
[close to a watershed moment](https://thegradient.pub/nlp-imagenet/), and that
within a year we'll have downloadable language models (that was six months ago,
and, uh...). However, [Ana Marasović](https://amarasovic.github.io/) points out
that there is [a tremendous amount of work demonstrating
that](https://thegradient.pub/frontiers-of-generalization-in-natural-language-processing/)

> "despite good performance on benchmark datasets, modern NLP techniques are
> nowhere near the skill of humans at language understanding and reasoning when
> making sense of novel natural language inputs".

I am confident that I know very little about all of this. Instead, I'll talk
about something I do know about:

---

- Overview of NLP: where is it now?

- Why do I think NLP can benefit from Bayesianism?

- Focus on text clustering, which is just one small face of NLP. Applications in
  information retrieval, document organization, corpus summarization, document
  categorization...

- Why is matrix factorization good for text clustering? Even if it isn't
  Bayesian???
  BECAUSE IT IS BOTH A CLUSTERING AND A FEATURE ENGINEERING TRICK: IT GIVES US A
  BETTER VIEW OF OUR LATENT SPACE!!
  Matrix factorization lives an interesting double life as a feature
  transformation / preprocessing step, and as a clustering algorithm on its own.
  Aggarwal and Zhai indicate that chaining matrix factorization with some other
  clustering technique (e.g. agglomerative clustering, topic modelling) is
  fairly common and is called _concept decomposition_, but I haven't seen any
  other source back this up.

---

Some papers I read.

- Plain NMF

- Prob MF

- Bayesian Prob MF

---

[^1]: Aggarwal, Charu C, and ChengXiang Zhai. “A Survey of Text Clustering Algorithms.” Mining Text Data, Springer, 2014, pp. 77–128. ([http://charuaggarwal.net/text-cluster.pdf](http://charuaggarwal.net/text-cluster.pdf))

[^2]: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf

[^3]: https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf
