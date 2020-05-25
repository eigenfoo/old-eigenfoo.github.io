---
title: Transformers in Natural Language Processing — A Brief Survey
excerpt: "I've recently had to learn a lot about natural language processing —
specifically, Transformer-based models. I thought I’d write up my reading and research
and post it."
tags:
  - machine learning
  - deep learning
  - natural language processing
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background11.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-05-23
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
highlight general trends in deep NLP, instead of commenting on individual architectures
or systems.

As a disclaimer, this post by no means exhaustive and is biased towards
Transformer-based models, which seem to be the dominant breed of NLP systems (at least,
at the time of writing).

## Some Architectures and Developments

Here's an (obviously) abbreviated history of deep learning in NLP in (roughly)
chronological order.

It's also worth covering some other (non-Transformer-based) models, because they
illuminate the history of NLP.

1. word2vec and GloVe
   * word2vec: Mikolov et al., Google. January - October 2013. 
   * GloVe: Pennington et al., Stanford CS. EMNLP 2014.
   * These were the first instances of word embeddings pre-trained on large amounts of
     unlabeled text. These word embeddings generalized well to most other tasks (even
     with limited amounts of labeled data), and usually led to appreciable improvements
     in performance.
   * These ideas were immensely influential and have served NLP extraordinarily well.
     However, they suffer from a major limitation. They are _shallow_ representations
     that can only be used in the first layer of any network: the remainder of the
     network must still be trained from scratch.
   * The main appeal is well illustrated below: each word has its own vector
     representation, and there are linear vector relationships can encode common-sense
     semantic meanings of words.

   <figure class="align-center">
     <a href="https://www.tensorflow.org/images/linear-relationships.png"><img style="float: middle" src="https://www.tensorflow.org/images/linear-relationships.png" alt="Linear vector relationships in word embeddings"></a>
     <figcaption>Linear vector relationships in word embeddings. Source: <a href="https://www.tensorflow.org/images/linear-relationships.png">TensorFlow documentation</a>.</figcaption>
   </figure>

   * Further reading
     + [word2vec arXiv paper](http://arxiv.org/abs/1301.3781)
     + [Subsequent word2vec arXiv paper](http://arxiv.org/abs/1310.4546)
     + [GloVe website](https://nlp.stanford.edu/projects/glove/)

1. Broadly speaking, after word2vec/GloVe and before Transformers, a lot of ink was
   spilled on other different approaches to NLP, including (but certainly not limited
   to)
   1. Convolutional neural networks
   1. Recurrent neural networks
   1. Reinforcement learning approaches
   1. Memory-augmented deep learning
   * Perhaps the most famous of such models is [ELMo (Embeddings from Language
     Models)](https://allennlp.org/elmo) by AI2, which learned bidirectional word
     embeddings using LSTMs, and began NLP's fondness of Sesame Street.
   * I won't go into much more detail here: partly because not all of these approaches
     have held up as well as current Transformer-based models, and partly because I have
     plans for my computer that don't involve blogging about recent advances in NLP.
   * Here is [a survey paper](https://arxiv.org/abs/1708.02709) (and an [associated blog
     post](https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d))
     published shortly after the Transformer was invented, which summarizes a lot of the
     work that was being done during this period.

1. Transformer
   * Vaswani et al., Google Brain. December 2017.
   * The authors introduce a feed-forward network architecture, using only attention
     mechanisms and dispensing with convolutions and recurrence entirely (which were not
     uncommon techniques in NLP at the time).
   * It achieved state-of-the-art performance on several tasks, and (perhaps more
     importantly) was found to generalize very well to other NLP tasks, even with
     limited data.
   * Since this architecture was the progenitor of so many other NLP models, it's
     worthwhile to dig into the details a bit. The architecture is illustrated below:
     note that its feed-forward nature and multi-head self attention are critical
     aspects of this architecture!

   <figure class="align-center">
     <a href="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/06/Screenshot-from-2019-06-17-20-01-32.png"><img style="float: middle" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/06/Screenshot-from-2019-06-17-20-01-32.png" alt="Graphical representation of the Transformer"></a>
     <figcaption>Graphical representation of a Transformer block. Source: <a href="https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/">Analytics Vidhya</a>.</figcaption>
   </figure>

   * Further reading
     + [arXiv paper](https://arxiv.org/pdf/1706.03762.pdf)
     + [_The Illustrated Transformer_ blog post](https://jalammar.github.io/illustrated-transformer/)
     + [_The Annotated Transformer_ blog post](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

1. ULMFiT (Universal Language Model Fine-tuning for Text Classification)
   * Howard and Ruder. January 2018.
   * The authors introduce an effective transfer learning method that can be applied to
     any task in NLP: this paper introduced the idea of general-domain, unsupervised
     pre-training, followed by task-specific fine-tuning. They also introduce other
     techniques that are fairly common in NLP now, such as slanted triangular learning
     rate schedules. (what some researchers now call warm-up).
   * Further reading
     + [arXiv paper](https://arxiv.org/pdf/1801.06146.pdf)

1. GPT-1 and GPT-2 (Generative Pre-trained Transformers)
   * Radford et al., OpenAI. June 2018 and February 2019.
   * At the risk of peeking ahead, GPT is largely BERT but with Transformer decoder
     blocks, instead of encoder blocks. Note that in doing this, we lose the
     autoregressive/unidirectional nature of the model.
   * GPT-2 generated some controversy, as OpenAI [initially refused to open-source the
     model](https://www.theverge.com/2019/2/14/18224704/ai-machine-learning-language-models-read-write-openai-gpt2),
     citing potential malicious uses, but [ended up releasing the model
     later](https://www.theverge.com/2019/11/7/20953040/openai-text-generation-ai-gpt-2-full-model-release-1-5b-parameters).
   * Further reading
     + [GPT-1 blog post](https://openai.com/blog/language-unsupervised/)
     + [GPT-2 blog post](https://openai.com/blog/better-language-models/)
     + [_The Illustrated GPT-2_ blog post](http://jalammar.github.io/illustrated-gpt2/)

1. BERT (Bidirectional Encoder Representations from Transformers)
   * Devlin et al., Google AI Language, May 2019.
   * The authors use the Transformer encoder (and only the encoder) to pre-train deep
     bidirectional representations from unlabeled text. This pre-trained BERT model can
     then be fine-tuned with just one additional output layer to achieve
     state-of-the-art performance for many NLP tasks, without substantial task-specific
     architecture changes, as illustrated below.

   <figure class="align-center">
     <a href="https://i.pinimg.com/originals/02/95/a3/0295a3be438ae68f604e53fc88c7edb4.png"><img style="float: middle" src="https://i.pinimg.com/originals/02/95/a3/0295a3be438ae68f604e53fc88c7edb4.png" alt="Graphical representation of BERT"></a>
     <figcaption>Graphical representation of BERT. Source: <a href="https://i.pinimg.com/originals/02/95/a3/0295a3be438ae68f604e53fc88c7edb4.png">Pinterest</a>.</figcaption>
   </figure>

   * BERT was a drastic development in the NLP landscape: it became almost a cliche to
     conclude that BERT performs "surprisingly well" on whatever task or dataset you
     throw at it.
   * Further reading
     + [arXiv paper](https://arxiv.org/pdf/1810.04805.pdf)
     + [Accompanying blog post](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
     + [_The Illustrated BERT_ blog post](https://jalammar.github.io/illustrated-bert/)

1. RoBERTa (Robustly Optimized BERT Approach)
   * Liu et al., Facebook AI. June 2019.
   * The scientific contributions of this paper are best quoted from its abstract:

     > We find that BERT was significantly under-trained, and can match or exceed the
     > performance of every model published after it. [...] These results highlight the
     > importance of previously overlooked design choices, and raise questions about the
     > source of recently reported improvements.

   * The authors use an identical architecture to BERT, but propose several improvements
     to the training routine, such as changing the dataset and removing the
     next-sentence-prediction (NSP) pre-training task. Funnily enough, far and away the
     best thing the authors did to improve BERT was to do the most obvious thing: train
     it for longer!
   * Further reading:
     + [arXiv Paper](https://arxiv.org/abs/1907.11692)
     + [Accompanying blog post](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)

1. T5 (Text-to-Text Transfer Transformer)
   * Raffel et al., Google. October 2019.
   * There are two main contributions of this paper:
     1. The authors recast all NLP tasks into a text-to-text format: for example,
        instead of performing a two-way softmax for binary classification, one could
        simply teach an NLP model to output the tokens "spam" or "ham". This provides a
        unified text-to-text format for all NLP tasks.
     1. The authors systematically study and compare the effects of pre-training
        objectives, architectures, unlabeled datasets, transfer approaches, and other
        factors on dozens of canonical NLP tasks.
   * This paper (and especially the tables in the appendices!) probably cost the Google
     team an incredible amount of money, and the authors were very thorough in ablating
     what does and doesn't help for a good NLP system.
   * Further reading
     + [arXiv paper](https://arxiv.org/pdf/1910.10683.pdf)
     + [Accompanying blog post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)

## Some Thoughts and Observations

Here I comment on some general trends that I see in Transformer-based models in NLP.

1. Ever since Google developed the Transformer in 2017, most NLP contributions are not
   architectural: instead most recent advances have used the Transformer model as-is, or
   using some subset of the Transformer (e.g. BERT and GPT use exclusively Transformer
   encoder and decoder blocks, respectively). Instead, recent research has focused on
   the way NLP models are pre-trained or fine-tuned, or creating a new dataset, or
   formulating a new NLP task to measure "language understanding", etc.
   * I'm personally not sure what to make of this development: why did we collectively
     agree that architectural research wasn't worth pursuing anymore?

1. It was never a question of _whether_ NLP systems would follow computer vision's model
   of fine-tuning pre-trained models (i.e. training a model on ImageNet and then doing
   task-specific fine-tuning for downstream applications), but rather _how_.
   1. What specific task and/or dataset should NLP models be pre-trained on?
      * Language modelling has really won out here.
   1. Exactly _what_ is being learnt during pre-training?
      * It used to be vectors for words, now it is an entire network, or what Sebastian
        Ruder calls _shallow to deep pre-training_.
   * Sebastian Ruder [wrote a great article](https://thegradient.pub/nlp-imagenet/) in
     The Gradient that delves more into this topic.

1. Different NLP models learn different kinds of embeddings, and it's worth
   understanding the differences between these various learnt representations.
   1. Contextual vs non-contextual embeddings
      * The first word embeddings (that is, word2vec and GloVe) were _non-contextual_:
        each word had its own embedding, independent of the words that came before or
        after it.
      * Almost all other embeddings are _contextual_ now: when embedding a token, they
        also consider the tokens before &/ after it.
   1. Unidirectional vs bidirectional embeddings
      * When considering the context of a token, the question is whether you should
        consider the tokens both before and after it (i.e. bidirectional embeddings), or
        just the tokens that came before (i.e. unidirectional embeddings).
      * Unidirectional embeddings make the sense when generating text (i.e. text
        generation must be done in the way humans write text: in one direction). On the
        other hand, bidirectional embeddings make sense when performing sentence-level
        tasks such as summarization or rewriting.
      * The Transformer was notable in that it had bidirectional encoder blocks and
        unidirectional decoder blocks. That's why BERT [GPT-2] produces bidirectional
        [unidirectional] embeddings, since it's a stack of Transformer encoders
        [decoders].
      * Note that the unidirectional/bidirectional distinction is related to whether or
        not the model is autoregressive: autoregressive models learn unidirectional
        embeddings.

