---
title: Transformers in Natural Language Processing — A (Very) Brief Survey
excerpt: ""
tags:
  - machine learning
  - deep learning
  - natural language processing
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background4.png
  overlay_filter: 0.05
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-05-23
mathjax: true
search: false
---

{% if page.noindex == true %}
  <meta name="robots" content="noindex">
{% endif %}

I've recently had to learn a lot about natural language processing (NLP), specifically
Transformer-based NLP models.

Similar to my previous blog post on [deep autoregressive
models](https://eigenfoo.xyz/deep-autoregressive-models/), this blog post is a write-up
of my reading and research: I assume basic familiarity with deep learning, and aim to
highlight general trends in deep NLP, instead of commenting on individual NLP
architectures or systems.

This post is biased towards Transformer-based models, which seem to be the dominant
breed of NLP systems (at least, at the time of writing).

## A Brief History of Transformers in NLP

Here's an (obviously) abbreviated history of deep learning in NLP in (roughly)
chronological order.

It's also worth covering some other (non-Transformer-based) models, because they
illuminate the history of NLP.

- word2vec and GloVe
  * word2vec: Mikolov et al., Google. January-October 2013. 
  * GloVe: Pennington et al., Stanford CS. EMNLP 2014.
  * These were the first instances of word embeddings pre-trained on large amounts of
    unlabeled text. These word embeddings generalized well to most other tasks (even
    with limited amounts of labeled data), usually led to appreciable improvements in
    performance.
  * These ideas were immensely influential and have served NLP extraordinarily well.
    However, they suffer from a major limitation. They are _shallow_ representations
    that can only be used in the first layer of any network: the remainder of the
    network must still be trained from scratch.
  * There were two word2vec papers: the first one introduced the idea, whereas the
    second improved on it with techniques e.g. negative sampling.
  * Further reading
    + [word2vec arXiv paper](http://arxiv.org/abs/1301.3781)
    + [Subsequent word2vec arXiv paper](http://arxiv.org/abs/1310.4546)
    + [GloVe website](https://nlp.stanford.edu/projects/glove/)

- Broadly speaking, after word2vec/GloVe and before Transformers, there was a lot of ink
  spilled on approaches to NLP that have not been as successful, such as
  * Convolutional neural networks
  * Recurrent neural networks
  * Reinforcement learning approaches
  * Memory-augmented deep learning
- Perhaps the most famous of such models is [ELMo (Embeddings from Language
  Models)](https://allennlp.org/elmo) by the Allen Institute for Artificial Intelligence
  (a.k.a. AI2) and presented at NAACL 2018, which began NLP's fondness of Sesame Street.
- Here is [a survey paper](https://arxiv.org/abs/1708.02709) (and an [associated blog
  post](https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d))
  (published shortly after the Transformer was invented), which summarizes a lot of the
  work that was being done before the Transformer was invented.

- Transformer
  * Vaswani et al., Google Brain. December 2017.
  * The authors introduce a feed-forward network architecture, using only attention
    mechanisms and dispensing with recurrence and convolutions entirely (which were
    widespread techniques for NLP at the time).
  * It achieved state-of-the-art performance on several tasks, and (perhaps more
    importantly) was found to generalize very well to other NLP tasks, even with limited
    data.
  * Main graph is here: its feed-forward nature and the attention heads are critical to
    its success!

    ![Diagram of the Transformer](http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)

  * Further reading
    + [arXiv paper](https://arxiv.org/pdf/1706.03762.pdf)
    + [_The Illustrated Transformer_ blog post](https://jalammar.github.io/illustrated-transformer/)
    + [_The Annotated Transformer_ blog post](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

- ULMFiT (Universal Language Model Fine-tuning for Text Classification)
  * Howard and Ruder. January 2018.
  * The authors introduce an effective transfer learning method that can be applied
    to any task in NLP: this paper introduced the idea of general-domain, unsupervised
    pre-training, followed by task-specific fine-tuning. They also introduce other
    techniques that are fairly common in NLP now, such as slanted triangular learning
    rate schedules. (what some researchers now call warm-up).
  * Further reading
    + [arXiv paper](https://arxiv.org/pdf/1801.06146.pdf)

- GPT-1 and GPT-2 (Generative Pre-trained Transformers)
  * Radford et al., OpenAI. June 2018 and February 2019.
  * GPT-2 generated some controversy, as OpenAI [initially refused to open-source the
    model](https://www.theverge.com/2019/2/14/18224704/ai-machine-learning-language-models-read-write-openai-gpt2),
    citing potential misuses, but [later released the
    model](https://www.theverge.com/2019/11/7/20953040/openai-text-generation-ai-gpt-2-full-model-release-1-5b-parameters).
  * Further reading
    + [GPT-1 blog post](https://openai.com/blog/language-unsupervised/)
    + [GPT-2 blog post](https://openai.com/blog/better-language-models/)

- BERT (Bidirectional Encoder Representations from Transformers)
  * Devlin et al., Google AI Language, May 2019.
  * The authors use the Transformer encoder (and only the encoder) to pre-train deep
    bidirectional representations from unlabeled text. This pre-trained BERT model can
    then be fine-tuned with just one additional output layer to achieve state-of-the-art
    performance for many NLP tasks, without substantial task-specific architecture
    changes.
  * Further reading
    + [arXiv paper](https://arxiv.org/pdf/1810.04805.pdf)
    + [Accompanying blog post](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
    + [_The Illustrated BERT_ blog post](https://jalammar.github.io/illustrated-bert/)

- RoBERTa (Robustly Optimized BERT Approach)
  * Liu et al., Facebook AI. June 2019.
  * The scientific contributions of this paper are best quoted from its abstract:

    > We find that BERT was significantly under-trained, and can match or exceed the
    > performance of every model published after it. [...] These results highlight the
    > importance of previously overlooked design choices, and raise questions about the
    > source of recently reported improvements.

  * Identical architecture to BERT, but with different/better training settings and
    other various improvements.
  * Further reading:
    + [arXiv Paper](https://arxiv.org/abs/1907.11692)
    + [Accompanying blog post](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)

- T5 (Text-to-Text Transfer Transformer)
  * Systematic ablation study
  * Raffel et al. 10-2019 (Google)
  * Further reading
    + [arXiv paper](https://arxiv.org/pdf/1910.10683.pdf)
    + [Accompanying blog post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)

## General Trends

- Representations
  * asdfasdf
    1. Contextual vs non-contextual embeddings
       + Only word2vec and GloVe are non-contextual.
       + All other embeddings are contextual now.
    2. Unidirectional vs bidirectional embeddings
       + ELMo: bidirectional
       + ULMFiT:
       + BERT: bidirectional encoder, unidirectional decoder
       + T5: bidirectional

- Pre-training
  * Great article: https://thegradient.pub/nlp-imagenet/
  * It was never a question of _whether_ NLP systems would follow CV's model of
    fine-tuning pre-trained models, but rather _how_.
    1. What specific task and/or dataset should NLP models be pre-trained on?
       + Language modelling has really won out here.
    2. Exactly _what_ are we pre-training?
       + It used to be vectors for words, now it is an entire network, or what Sebastian
         Ruder calls _shallow to deep pre-training_.

- Feature vs fine-tuning based (strategies for applying pre-trained embeddings to
  downstream tasks)

