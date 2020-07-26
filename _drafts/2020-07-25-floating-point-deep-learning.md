---
title: Floating-Point and Deep Learning
excerpt: "What floating-point is, why you (a deep learner) should care, and what you (a
deep learner) can do about it."
tags:
  - deep learning
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background15.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-07-25
mathjax: true
---

1. Why should you, a deep learning practitioner, care about floating-point?
2. What even _is_ floating-point, especially these new floating-point formats made
   specifically for deep learning?
3. What practical advice is there on floating-point for deep learning?

## Floating-Point? In _My_ Deep Learning?

[It's more likely than you
think!](https://knowyourmeme.com/photos/6052-its-more-likely-than-you-think)

It's been known for quite some time that [deep neural networks can
tolerate](https://arxiv.org/abs/1502.02551) [lower numerical
precision](https://arxiv.org/abs/1502.02551). High-precision calculations turn out not
to be that useful in training or inferencing neural networks: additional precision
confers no benefit while being slower and less memory-efficient.

Surprisingly, some models can even reach a higher accuracy with lower precision, which
recent research attributes to the [regularization effects from the lower
precision](https://arxiv.org/abs/1809.00095).

## Floating-Point Formats

There are a lot more floating-point formats, but only a few have gained traction.

### IEEE Floating-Point Formats

These floating-point formats are probably what most people think of when someone says
"floating-point". They were defined by the IEEE standard 754, which sets out several
formats. For the purposes of deep learning, we are only interested three:
[FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format),
[FP32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) and
[FP64](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) (a.k.a.
half-, single- and double-precision floating-point formats)[^1].

Let's take FP32 as an example. Each FP32 number is represented as a sequence of 32 bits,
$b_{31} b_{30} ... b_{0}$. Altogether, this sequence represents the real number

$$ (-1)^{b_{31}} \cdot 2^{(b_{30} b_{29} ... b_{23}) - 127} \cdot (1.b_{22} b_{21} ... b_{0})_2 $$

Here, $b_{31}$ (the _sign bit_) determines the sign of the represented value.

$b_{30}$ through $b_{23}$ determine the magnitude or scale of the represented value
(notice that a change in any of these bits drastically changes the size of the
represented value). These bits are called the _exponent_ or _scale bits_.

Finally, $b_{22}$ through $b_{0}$ determine the precise value of the represented value.
These bits are called the _mantissa_ or _precision bits_.

The number of exponent and mantissa bits change as we go from FP16 to FP32 to FP64. When
tabulated, they look like:

|      | Sign Bits   | Exponent (Scale) Bits | Mantissa (Precision) Bits |
| :--- | ----------: | --------------------: | ------------------------: |
| FP16 | 1           | 5                     | 10                        |
| FP32 | 1           | 8                     | 23                        |
| FP64 | 1           | 11                    | 53                        |

There are some details that need to be ironed out (e.g. how to represent NaNs, positive
and negative infinities), but by and large, this is how floating point numbers work.

A lot more detail can be found on their [Wikipedia
entry](https://en.wikipedia.org/wiki/Floating-point_arithmetic#IEEE_754:_floating_point_in_modern_computers)
and of course the [latest revision of the IEEE 754
standard](https://ieeexplore.ieee.org/document/8766229) itself.

FP32 and FP64 are widely supported by software (C/C++, PyTorch, TensorFlow) and hardware
(x86 CPUs and most NVIDIA/AMD GPUs). FP16, on the other hand, is not supported in C/C++
(you need to use [a special library](http://half.sourceforge.net/)). However, since deep
learning is trending towards favoring FP16 over FP32, it has found support in the main
deep learning frameworks (e.g. `tf.float16` and `torch.float16`).

In terms of hardware support, FP16 is not supported in x86 CPUs as a distinct type, but
is well-supported on modern GPUs.

### Google bfloat16

- https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
- https://www.nextplatform.com/2018/05/10/tearing-apart-googles-tpu-3-0-ai-coprocessor/

Basically the same as half-precision floating-point format, but 3 mantissa bits become
exponent bits. In this way, bfloats can express more scale

<figure class="align-center">
  <img style="float: middle" src="https://storage.googleapis.com/gweb-cloudblog-publish/images/Three_floating-point_formats.max-700x700.png" alt="Diagram illustrating the number and type of bits in a bfloat">
  <figcaption>The number and type of bits in a bfloat. Source: <a href="https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus">Google Cloud blog</a>.</figcaption>
</figure>

### NVIDIA TensorFloat

Strictly speaking, this isn't really its own floating-point format, just an overzealous
branding of the technique that NVIDIA developed to train in mixed precision on their
Tensor Core hardware[^2].

An NVIDIA TensorFloat is just a 32-bit float that drops 13 precision bits in order to
execute on Tensor Cores. Thus, it has the precision of FP16 (10 bits), with the range of
FP32 (8 bits). However, if you're not using Tensor Cores, it's just a 32-bit float; if
you're only thinking about storage, it's just a 32-bit float.

<figure class="align-center">
  <img style="float: middle" src="https://blogs.nvidia.com/wp-content/uploads/2020/05/tf32-Mantissa-chart-hi-res-FINAL.png" alt="Diagram illustrating the number and type of bits in an NVIDIA TensorFloat">
  <figcaption>The number and type of bits in an NVIDIA TensorFloat. Source: <a href="https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/">NVIDIA blog</a>.</figcaption>
</figure>

> TODO: why should people be excited about this??

## Floating-Point Precision and Deep Learning

There are basically three ways:

1. Most GPUs: AMP
2. TPUs: bfloat
3. NVIDIA A100s: TensorFloat?

### Automatic Mixed Precision (AMP) Training

- Apex
- PyTorch uses this... where?
- What about TensorFlow?
- https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html

1. *Loss scaling:* multiply the loss by some large number, and divide the gradient
   updates by this same large number. This avoids the loss underflowing (i.e. clamping
   to zero because of the finite precision) in FP16, while still maintaining faithful
   backward propagation.
2. *FP32 master copy of weights*: store the weights themselves in FP32, but cast them to
   FP16 before doing the forward and backward propagation (to reap the performance
   benefits). During the weight update, the FP16 gradients are cast to FP32 to update
   the master copy.

You can read more about these techniques in [this paper by NVIDIA and Baidu
Research](https://arxiv.org/abs/1710.03740), or on the accompanying [blog post by
NVIDIA](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/).

### Google TPUs

If you're lucky enough to have access to Google TPUs, a good option would be to use
bfloats.

### NVIDIA A100

NVIDIA A100s are the first(?) GPUs to support TensorFloat, which might be better than
AMP?

---

Links

- https://medium.com/@moocaholic/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407
- https://www.quora.com/What-is-the-difference-between-FP16-and-FP32-when-doing-deep-learning/answer/Travis-Addair

---

[^1]: Technically speaking, there are [quadruple-](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format) and [octuple-precision](https://en.wikipedia.org/wiki/Octuple-precision_floating-point_format) floating-point formats, but those are pretty rarely used, and certainly unheard of in deep learning.

[^2]: A Tensor Core is essentially a mixed-precision FP16/FP32 core, which NVIDIA has optimized for deep learning applications.
