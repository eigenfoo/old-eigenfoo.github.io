---
title: Floating-Point Precision and Deep Learning
excerpt: ""
tags:
  - pytorch
  - deep learning
  - machine learning
header:
  overlay_image: /assets/images/cool-backgrounds/cool-background15.png
  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'
last_modified_at: 2020-07-25
mathjax: true
---

there is a trend in deep learning towards using FP16 instead of FP32 because lower
precision calculations seem to be not critical for neural networks. Additional precision
gives nothing, while being slower, takes more memory and reduces speed of communication.

## Floating-Point Formats

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

## Mixed Precision Training

### AMP

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

You can read more about these techniques in this [paper by NVIDIA and Baidu
Research](https://arxiv.org/abs/1710.03740), or on the accompanying [blog post by
NVIDIA](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/).


## Remarks and

- Has an impact on training!
- See first paragraph of
  * https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus

---

Links

- https://medium.com/@moocaholic/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407
- https://www.quora.com/What-is-the-difference-between-FP16-and-FP32-when-doing-deep-learning/answer/Travis-Addair

---

[^1]: Technically speaking, there are [quadruple-](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format) and [octuple-precision](https://en.wikipedia.org/wiki/Octuple-precision_floating-point_format) floating-point formats, but those are pretty rarely used, and certainly unheard of in deep learning.

[^2]: A Tensor Core is essentially a mixed-precision FP16/FP32 core, which NVIDIA has optimized for deep learning application.
