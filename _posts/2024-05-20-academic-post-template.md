---
layout: post
title: "学术博文案例：数学与代码"
date: 2024-05-20
description: 这是一篇展示如何在 al-folio 中使用数学公式和代码高亮的案例文章。
tags: academic formatting math code
categories: tutorial
---

这是一篇示例文章，展示了学术写作中常用的排版功能。

### 1. 数学公式展示

你可以使用 LaTeX 语法插入行内公式，例如：$E = mc^2$。

也可以插入块级公式：

$$
\int_{a}^{b} f(x) dx = F(b) - F(a)
$$

### 2. 代码片段展示

这是一个 Python 代码块示例：

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
```

保存后运行 `bundle exec jekyll serve` 即可在本地看到效果。
