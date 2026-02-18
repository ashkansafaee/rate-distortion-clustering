# Information-Theoretic Clustering via Rate-Distortion Theory

An implementation of the Blahut-Arimoto algorithm for soft clustering, grounded in Rate-Distortion Theory from information theory. 

## Problem

Standard clustering algorithms (k-means, hierarchical) minimise geometric distance. But what if you want to frame clustering as an information problem — trading off *complexity* (how much information you retain about the original data) against *distortion* (how far cluster assignments are from the true points)?

Rate-Distortion Theory gives a principled answer. This project implements it from scratch.

## Background

Given a set of 200 2D data points, the goal is to find a compressed representation T (cluster assignments) that minimises the mutual information I(X; T) — the "rate" or complexity — subject to a constraint on average distortion D.

This is formalised as minimising the Lagrangian:

```
L = I(X; T) + β · D
```

where β controls the trade-off between compression and fidelity. As β increases, the algorithm is forced to accept lower distortion, producing finer-grained (more complex) cluster structures.

## Implementation

The core of the project is a from-scratch implementation of the **Blahut-Arimoto self-consistent equations**:

```
p(t|x) = p(t) / Z(x,β) · exp[−β · d(x,t)]
p(t)   = Σ_x p(x) · p(t|x)
```

These are iterated until convergence (‖p_new − p_old‖ < 1e-12).

Key design decisions:
- Random initialisation of p(t|x) at each run to escape local optima
- Multiple runs (n=10) per (β, Nc) configuration with selection of the minimum Lagrangian
- Mutual information computed with ε-smoothing to avoid log(0) instability
- Average distortion computed as E[d(x,t)] over the joint distribution p(x,t)

## Experiments

The algorithm was run across:
- **β values**: {1, 2, 4, 8, 16, 32} (log-spaced, increasing compression pressure)
- **Cluster counts Nc**: {2, 3, 4}

Results are visualised as an **information curve** — plotting I(X;T) against −D — which traces the Pareto frontier between compression and distortion for each Nc. This directly mirrors the theoretical rate-distortion curve from information theory.

## Key Findings

- Higher β forces lower distortion at the cost of higher complexity, exactly as theory predicts
- Nc=4 achieves the best distortion-complexity frontier, suggesting 4 natural clusters in the data
- The information curve inflection points identify the β values where additional complexity yields diminishing distortion improvements — practically useful for choosing Nc

## Stack

Python · NumPy · SciPy · Matplotlib · Seaborn · Pandas

## References

- Tishby, N., Pereira, F. C., & Bialek, W. (1999). *The Information Bottleneck Method*
- Cover, T. & Thomas, J. *Elements of Information Theory*
- Stanford EE368b Rate-Distortion Theory notes

---

*Part of advanced studies in Mathematical Statistics and Machine Learning, Stockholm University.*
