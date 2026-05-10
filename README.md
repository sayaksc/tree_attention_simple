# tree_attention_simple

Experiments for the paper:

> **Poly-attention: a general scheme for higher-order self-attention**
> Sayak Chakrabarti, Toniann Pitassi, Josh Alman
> *ICLR 2026 (Poster)*

- [OpenReview](https://openreview.net/forum?id=amivrmQyvQ)
- [arXiv](https://arxiv.org/abs/2602.02422)

---

## Overview

Standard self-attention models pairwise interactions between tokens. This paper defines **poly-attention** — a broad class of higher-order self-attention mechanisms that incorporate arbitrary tensor computations and arbitrary relationship structures between tokens. Prior higher-order alternatives (higher-order attention, Strassen attention) required superquadratic time even for simple tasks. This work:

- gives a unified framework that subsumes all prior higher-order attention proposals as special cases,
- studies the **computational complexity** of each mechanism with new algorithms and matching lower bounds,
- **tightly characterizes** which polyadic tasks each mechanism can perform, and
- introduces a new attention mechanism computable exactly in **quadratic time** that can perform function composition for any fixed number of functions — a task previously thought to require superquadratic time.

---

## This directory

`tree_attention_simple` contains simplified implementations and experiments for the **tree-structured poly-attention** variant described in the paper. The focus is on clarity and reproducibility over performance engineering.

---

## Citation

```bibtex
@inproceedings{
  chakrabarti2026polyattention,
  title     = {Poly-attention: a general scheme for higher-order self-attention},
  author    = {Sayak Chakrabarti and Toniann Pitassi and Josh Alman},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=amivrmQyvQ}
}
```

---

## License

This code is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
