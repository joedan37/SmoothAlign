# SmoothAlign

This repository contains the research code for SmoothAlign, a probabilistic
smoothing framework for correcting semantic-overlap bias in multimodal
representation learning.

## GRAM

The `GRAM/` directory contains the volume-based joint geometric alignment
implementation used for multimodal video retrieval, together with the
corresponding model, data, evaluation, and experiment configuration code.

## PCME++

The `PCME++/` directory contains the image--text matching implementations for
standard InfoNCE, PCME++, and their SmoothAlign variants, including the
probabilistic matching and distributional-prior components.

## IEMOCAP

The `IEMOCAP/` directory contains the controlled diagnostic code used to study
the correction of semantic-overlap bias and the resulting negative-repulsion
weights.

Detailed experimental contents and usage instructions will be released after
the paper is accepted.
