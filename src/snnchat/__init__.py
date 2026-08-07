"""snnchat -- a conversational front end for the spiking char-LM.

This package is NOT part of the Phase-1..5 research protocol. Nothing in it is
pre-registered, no number it prints is a reported figure, and it must never be
cited as evidence for or against a research candidate. It exists for one
purpose: so that the spiking model can be *talked to*.

It depends on `snn` (the research package) one way only -- it imports the
committed architectures and reuses them unchanged. It writes nothing into
`snn/`, `experiments/` or `docs/`. The model it trains is a
`snn.model.TwoCompThresholdCharLM`, byte-for-byte the arm EXP_011 measured; the
only things this package adds are a different vocabulary, a different corpus, a
different initialisation of an existing parameter, and a sampling loop.

The modules, in the order they matter:

    tokenizer   the closed alphabet and the four role markers
    sources     downloads, one reader per dataset
    persona     the authored greeting/identity exchanges
    topics      what a story is about, so it can be asked for by subject
    build_corpus  renders and packs; one .bin per source
    data        the mixture sampler, and the per-character loss weights
    model       ChatConfig, and the slow-pole spread
    train       the CUDA-graph captured, restartable training loop
    generate    sampling filters, ChatSampler, ChatSession
    rerank      draw N replies, keep the one the prompt explains
    quality     the metrics that say whether any of it helped

See docs/chat/README.md, and docs/chat/QUALITY.md for how reply quality is
measured here and what moved it.
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
