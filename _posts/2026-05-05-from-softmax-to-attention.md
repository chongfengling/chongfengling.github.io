---
layout: post
title: "From softmax to attention"
date: 2026-05-05
lang: zh
translation_id: softmax-0
toc:
  sidebar: left
categories:
  - CUDA
  - kernel
tags:
  - softmax
  - attention
---

# Softmax
## Why we need softmax?

For the attention formula:
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$
Where
- $Q$ is the query matrix, $Q \in \mathbb{R}^{L \times d}$
- $K$ is the key matrix, $K \in \mathbb{R}^{L \times d}$
- $V$ is the value matrix, $V \in \mathbb{R}^{L \times d}$
- $L$ is the sequence length, $d$ is the hidden dimension. 
- Apply softmax the last dimension (columns) of $QK^T$.

## Naive Softmax
### Definition

Given a vector $\mathbf{x} = [x_1, x_2, \dots, x_n]$, the standard Softmax output is defined as:

$$
\text{softmax}(\{x_1, \dots, x_N\}) = \left\{ \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} \right\}_{i=1}^N
$$

### Numerical Stability Issues

The input $x_i$ can be very large, such that the exponentials $e^{x_i}$ will result in **overflow**, leading to numerical instability. For example, the largest representable float16 value is approximately $65505 \approx e^{11}$.

To mitigate this, a common technique is to subtract the maximum value from all inputs before applying the exponential function, named **Safe Softmax**.

## Safe Softmax
### Definition
The Safe Softmax is defined as:
$$
\text{softmax}(\{x_1, \dots, x_N\}) = \left\{ \frac{e^{x_i-m}}{\sum_{j=1}^N e^{x_j-m}} \right\}_{i=1}^N, \;where\; m = \max(x_1, \dots, x_N)
$$

Easy to verify 

$$
\frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} = \frac{e^{x_i-m}}{\sum_{j=1}^N e^{x_j-m}}
$$

$$
e^{x_i-m} \in (0, 1] 
$$

### Steps (3-pass)
1. **Find Max**: Find the maximum value $m$ cross all elements in $\mathbf{x}$.
2. **Calculate Sum**: Compute $exp(x_i - m)$ for each element and sum them up.
3. **Normalize**: Divide each $exp(x_i - m)$ value by the total sum.




## Online Softmax
### Motivation

Given two sets of data, $\mathbf{x}$ and $\mathbf{y}$, maximum $m_x$ and $m_y$ and their softmax outputs separately as $P_x$ and $P_y$. Can we compute the softmax of the combined set $\mathbf{Z} = \mathbf{X} \cup \mathbf{Y}$ without pass the set $\mathbf{Z}$ again? The answer is yes, and the key is to update the maximum and the denominator in an online manner.

$$
\text{softmax}(\{z_1, \dots, z_N\}) = \left\{ \frac{e^{x_i-m}}{\sum_{j=1}^{N_x} e^{x_j-m_z} + \sum_{j=1}^{N_y} e^{y_j-m_z}} \right\}_{i=1}^N, \;where\; m = \max(x_1, \dots, x_{N_x}, y_1, \dots, y_{N_y})
$$

assume $m_x > m_y$, then we can update the maximum and the denominator as:

$$
\text{softmax}(\{z_1, \dots, z_N\}) = \left\{ \frac{e^{x_i-m}}{\sum_{j=1}^{N_x} e^{x_j-m_x} + \sum_{j=1}^{N_y} e^{y_j-m_x+m_y-m_y}} \right\}_{i=1}^N
$$

$$
\text{softmax}(\{z_1, \dots, z_N\}) = \left\{ \frac{e^{x_i-m}}{\sum_{j=1}^{N_x} e^{x_j-m_x} + e^{m_x-m_y}\sum_{j=1}^{N_y} e^{y_j-m_y}} \right\}_{i=1}^N
$$

Hence, if we record the maximum $m_x$, $m_y$ and the denominator of the previous set $\sum_{j=1}^{N_x} e^{x_j-m_x}$, $\sum_{j=1}^{N_y} e^{y_j-m_y}$, we can update the result without pass the combined set $\mathbf{Z}$ again.

### Steps (2-pass)

1. **Update Max and Denominator**: we conbine a new set of data $\mathbf{Y}$ which can only contains one element.
2. **Update results**: pass the whole set $\mathbf{Z}$ to update the results.

## Summary
TODO

---

# Attention

## Why standardattention is slow?

Back to the attention formula:
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

If we compute the attention in a naive way, 
- **FLOPs**: 
    - $O(2L^2d)$ for $QK^T$ and $O(2L^2d)$ for the softmax, about $4L^2d$.
- **Bytes**: 
    1. Read $Q$, $K$ and $V$: $O(Ld)$
    2. Read & Write $P$, $S$: $O(L^2)$
    3. Write $O$: $O(Ld)$

$$
AI \approx \frac{L^2d}{L^2 \cdot \text{sizeof(fp16)}} \approx d
$$

Hence, with format fp16, a large sequence length $L$ and a tipical head size $d=64 \text{ or } 128$, the attention mechanism is a **memory-bound** operation.




# Reference
1. [From Online Softmax to FlashAttention - By Zhao Ye](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)