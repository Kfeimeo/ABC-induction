"""ProbLog exact-inference oracle for *tiny* grammars (optional dependency).

ProbLog never enters the production pipeline: it is only used by the test
suite (``tests/problog_reference``) to check that the sparse Inside-Outside
implementation (both backends) computes exactly the same marginal
probability, span/class posteriors and constituency posteriors as an
independent exact inference engine.  ``problog`` is imported lazily; when it
is not installed, :func:`problog_available` is False and the tests skip.

Encoding (see :func:`encode`)
-----------------------------
Every node of the top-down derivation is identified by its Gorn address
``Path`` (``[]`` root, ``[0|P]`` left child, ``[1|P]`` right child).  For
every class ``C`` and path an annotated disjunction chooses *one* expansion
of ``C`` with its probability::

    p1::expand(C, Path, rule(A, B)); p2::expand(C, Path, lex(W)); ...

so a ProbLog world fixes exactly one top-down derivation from the root
(``[]``).  Two different derivations of the same sentence always differ in the
choice at the first path where they diverge, hence they are mutually
exclusive worlds and ``P(derives the sentence) = sum over derivations`` -
exactly the PCFG sum that the inside algorithm computes.  Virtual-evidence
potentials (leaf ``lambda``, span ``psi``, both <= 1) are independent
probabilistic facts that a derivation must "pay" at the corresponding node.
"""
from __future__ import annotations

from .oracle import ProbLogOracle, encode, problog_available

__all__ = ["ProbLogOracle", "encode", "problog_available"]
