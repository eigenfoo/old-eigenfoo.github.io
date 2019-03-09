---
title: Autoregressive Generative Models
excerpt:
tags:
  - deep learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background10.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
mathjax: true
last_modified_at: 2019-03-08
---

My current project involves working with a class of fairly niche and interesting
neural networks that aren't usually seen on a first pass through deep learning.
I thought I'd write up my reading and research and post it.

To be explicit as possible, this blog post is about _deep autoregressive
generative sequence models_. That's quite a mouthful of jargon (and two of those
words are kind of unnecessary), so let's unpack that.

- Deep
    * Well, these papers are using TensorFlow or PyTorch... so they must be
      "deep" :wink:
    * This is unnecessary word #1.
- Autoregressive
    * [Stanford has a good introduction](https://deepgenerativemodels.github.io/notes/autoregressive/)
      to autoregressive generative models, but I think a good way to explain
      these models is to compare them to recurrent neural networks (RNNs), which
      are a lot more well-known.

    <figure>
        <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"><img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png"></a>
        <figcaption>Obligatory RNN diagram. Source: <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Chris Olah</a>.</figcaption>
    </figure>

    * Like a recurrent model, an autoregressive model's output $$h_t$$ at time
      $$t$$ depends on not just $$x_t$$, but also $$x$$'s from previous time
      steps. However, _unlike_ a recurrent model, the previous $$x$$'s are not
      provided via some hidden state: they are given as just another input to
      the model.
    * The following animation of Google DeepMind's WaveNet illustrates this
      well: the $$t$$th output is generated in a _feed-forward_ fashion from
      several input $$x$$ values.[^1]

    <figure>
        <a href="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif"><img src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif"></a>
        <figcaption>WaveNet animation. Source: <a href="https://deepmind.com/blog/wavenet-generative-model-raw-audio/">Google DeepMind</a>.</figcaption>
    </figure>

    * Put simply, **an autoregressive model is merely a sequential model in
      which one predicts future values from past values.**
    * Autoregressive models offer a compelling bargain: you can have stable,
      parallel and easy-to-optimize training, faster inference time
      computations, and completely do away with [backpropagation through
      time](https://en.wikipedia.org/wiki/Backpropagation_through_time), if you
      are willing to accept a model that (by design) _cannot have_ infinite
      memory. There is [some recent
      research](http://www.offconvex.org/2018/07/27/approximating-recurrent/) to
      suggest that for many applications, this is a worthwhile tradeoff.

- Generative
    * Informally, a generative model is one that can generate new data after
      learning from the dataset.
    * More formally, a generative model models the joint distribution
      $$P(X, Y)$$ of the observation $$X$$ and the target variable $$Y$$.
      Contrast this to a discriminative model that models the conditional
      distribution $$P(X|Y)$$.
    * Generative adversarial networks (GANs) and variational autoencoders (VAEs)
      are all examples of generative models.

- Sequence model
    * Fairly self explanatory. A model that deals with sequential data, whether
      it is mapping sequences to scalars (e.g. language models), or mapping
      sequences to sequences (e.g. machine translation models).
    * Although sequence models are  sequential (duh), there has been good
      success at applying them to non-sequential data. For example, PixelRNN and
      PixelCNN (more on them below) can generate entire images, even though
      images are not sequential in nature: the model just generates a pixel at a
      time, in sequence!
    * Notice that an autoregressive model is automatically a sequence model, so
      technically it's redundant to further describe these models as sequential
      (which makes this unnecessary word #2). Still, it's a good thing to wrap
      your head around.

A good distinction is that "generative" and "sequential" describe _what_ these
models do, or what kind of data they deal with. "Autoregressive" describes _how_
these models do what they do: i.e. they describe properties of the network or
its architecture.

Despite the technicality involved in describing these models, they have seen a
good degree of success in the real world. Each model merits discussion, but
unfortunately there isn't enough space to devote to a detailed discussion about
them.

- [PixelCNN by Google DeepMind](https://arxiv.org/abs/1601.06759) was probably
  the first such model, and the progenitor of most of the other models below.
  Ironically, they spend the bulk of their paper discussing a recurrent model,
  PixelRNN, and consider PixelCNN as a "workaround" to avoid excessive
  computational burden. However, PixelCNN is probably this paper's most lasting
  contribution to the field.
- [PixelCNN++ by OpenAI](https://arxiv.org/abs/1701.05517) is, unsurprisingly,
  PixelCNN but with various improvements.
- [WaveNet by Google
  DeepMind](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) is
  heavily inspired by PixelCNN, and models raw audio, not just encoded music.
  They had to pull [a neat trick from signals
  processing](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) in order to
  cope with the sheer size of audio (high-quality audio involves at least
  16-bit precision samples, which means a 66,000-softmax per time step!)
- [Transformer, a.k.a. _the "attention is all you need" model_ by Google
  Brain](https://arxiv.org/abs/1706.03762) is (by now) a mainstay of NLP,
  performing very well at many NLP tasks and being incorporated into subsequent
  models like
  [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html).

These models have also found applications, such as [neural machine translation
(in linear time!)](https://arxiv.org/abs/1610.10099) and [modelling
video](https://arxiv.org/abs/1610.00527).

## Observations and Thoughts

Here are some general comments on autoregressive models generally.

- Given previous values $$x_1, x_2, ..., x_t$$, these models do not provide a
  _value_ for $$x_{t+1}$$, they provide a _probability distribution_ for it.
  Explicitly, they model $$P(x_{t+1} | x_1, x_2, ..., x_t)$$
    * If the $$x$$'s are discrete, then you can do this by outputting an
      $$N$$-way softmaxxed tensor, where $$N$$ is the number of discrete
      classes. This is what the original PixelCNN did, but gets messy when $$N$$
      is large (e.g. in the case of WaveNet, where $$N = 2^{16}$$).
    * If the $$x$$'s are continuous, there is still hope: you can model the
      probability distribution itself as the sum of basis functions, and having
      the model output the parameters of these basis functions. This massively
      reduces the memory footprint of the model. This was an important
      contribution of PixelCNN++.

- Autoregressive models are supervised.
    * With the success and hype of GANs and VAEs, it is easy to assume that
      all generative models are unsupervised: this is not true!
    * Way more stable than GANs, and can use all the good stuff from ML101:
      train-valid-test splits, cross validation, loss metrics, etc.

- Autoregressive sequential models work for both continuous and discrete data.
    * Autoregressive sequential models have worked for audio (WaveNet), images
      (PixelCNN++) and text (Transformer): these models are very flexible in the
      kind of data that they can model.
    * Contrast this to GANs, which (as far as I'm aware) cannot model discrete
      data.

- Autoregressive sequential models can be conditioned.
    * There are many options for conditioning! You can condition on both
      discrete and continuous variables; you can condition at multiple time
      scales; you can even condition on latent embeddings or the outputs of
      other neural networks.
        + Google DeepMind followed up their original PixelRNN paper with
          [another paper](https://arxiv.org/abs/1606.05328) that describes
          conditional generation.
        + WaveNet employs "global" and "local" conditioning: the former controls
          the identity of the speaker, while the latter controls the actual
          words that are said.
    * Note that since we do not have a discriminator (as in a GAN), it is
      difficult for an autoregressive model to achieve the kind of "unsupervised
      conditioning" that most GANs are capable of. In other words, the
      conditioning must be based on the labels of the data: you cannot expect to
      condition on random noise and have the model learn to map certain vectors
      in the noise space to latent stylistic features.

- Generating output sequences of variable length is not straightforward.
    * Neither WaveNet nor PixelCNN needed to worry about a variable output
      length: both audio and images are comprised of a fixed number of outputs
      (i.e. audio is just $$N$$ samples, and images are just $$N^2$$ pixels).
    * Text, on the other hand, is different: sentences can be of variable
      length. One would think that this is a nail in a coffin, but thankfully
      text is discrete: the standard trick is to have a "stop token" that
      indicates that the sentence is finished (i.e. model a full stop as its own
      token).
    * As far as I am aware, there is no prior literature on having both
      problems: a variable-length output of continuous values.

- Autoregressive models can model multiple timescales
    * In the case of music, there are important patterns to model at multiple
      time scales: individual musical notes drive correlations between audio
      samples at the millisecond scale, and music exhibits rhythmic patterns
      over the course of minutes. This is well illustrated by the following
      animation:

    <figure>
        <a href="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif"><img src="https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig1-Anim-160908-r01.gif"></a>
        <figcaption>Audio exhibits patterns at multiple timescales. Source: <a href="https://deepmind.com/blog/wavenet-generative-model-raw-audio/">Google DeepMind</a>.</figcaption>
    </figure>

    * There are two main ways capture these many patterns at these many
      different time scales: either make the receptive field of your model
      _extremely_ wide (e.g. through dilated convolutions), or pull some other,
      model-specific or data-specific trick.
        + WaveNet uses "context stacks".
        + Google DeepMind composed _several_ PixelRNNs to form a so-called
          "multi-scale" PixelRNN.

- How the hell can any of this work?
    * RNNs are theoretically more expressive and powerful than autoregressive
      models. However, recent work suggests that such infinite-horizon memory is
      seldom achieved in practice.
    * To quote [John Miller at the Berkeley AI Research
      lab](http://www.offconvex.org/2018/07/27/approximating-recurrent/):

    > **Recurrent models trained in practice are effectively feed-forward.** This
    > could happen either because truncated backpropagation through time
    > cannot learn patterns significantly longer than $$k$$ steps, or, more
    > provocatively, because models _trainable by gradient descent_ cannot
    > have long-term memory.

---

[^1]: There's actually a lot more nuance than meets the eye in this animation, but all I'm trying to illustrate is the feed-forward nature of autoregressive models.
