# Symbolic Generalization of Arithmetic with Transformers

> **2025 Inthon Datathon — 4th Place** · symbolic-generalization track

An encoder–decoder Transformer that performs multi-digit arithmetic **from raw text sequences alone**, with no symbolic preprocessing (no postfix conversion, no expression trees). The aim is *symbolic generalization*: inducing the underlying rules of arithmetic instead of memorizing seen examples, so the model extrapolates to expression lengths never observed during training (OOD).

## Problem

Standard sequence models tend to *memorize* arithmetic rather than *learn* it, and break down on expressions whose length or operator-precedence structure differs from the training distribution. The question we tackled: given only the character sequence of an expression, can a Transformer construct its own computation procedure and generalize structurally to unseen lengths?

## Approach

**1. Text-only formulation.**
The model reads the expression as a plain token sequence and emits the answer sequence — no postfix/tree preprocessing. This forces the network to internalize operator precedence and carry logic on its own, rather than offloading them to a hand-built representation.

**2. Curriculum learning — and what it actually taught us.**
We trained with a length curriculum, raising operands from 1 to 5 digits to encourage smoother convergence and better OOD length generalization. Instead we observed **catastrophic forgetting**: as difficulty increased, the model lost the basic-operation competence acquired in earlier stages. The empirical takeaway — for arithmetic reasoning, **randomized, diverse difficulty was more beneficial to generalization than a carefully ordered curriculum.**

**3. Reversed-sequence training.**
Inspired by input reversal in seq2seq learning (Sutskever et al., 2014), we reversed both the expression and the target so the model computes from the least-significant digit first, mirroring how humans carry. This made the **carry-over mechanism** easier to learn explicitly.

**4. Domain-aware loss & metrics.**
Beyond plain accuracy, we optimized against the competition's structural metrics — **TES (Target Error Symmetry)** and **EC (Expression Consistency)** — and proposed an **auxiliary loss encoding arithmetic identities** (e.g. commutativity) to push the model toward logically consistent outputs.

## Key Takeaways

- Achieved structural (length) generalization to out-of-distribution expressions from text-only input.
- Showed empirically that curriculum *ordering* can hurt arithmetic generalization via catastrophic forgetting — difficulty diversity mattered more than careful sequencing.
- The reversed-sequence formulation made carry logic explicit and improved output consistency.

## Tech Stack

Python · PyTorch

## Notes

Team entry for the 2025 Inthon Datathon.
<!-- Optional: add one line describing your specific contributions to the project. -->
