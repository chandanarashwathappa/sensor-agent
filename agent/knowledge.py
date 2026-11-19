"""Text knowledge base + FAISS semantic search (the RAG part).

Numbers come from the analytics tools. The knowledge base holds what tables
can't answer: definitions, units and how each failure mode is caused.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd

DATA_DICTIONARY = """DATA DICTIONARY - AI4I 2020 Predictive Maintenance Dataset

The dataset contains 10,000 records from a synthetic milling-machine process.

- Type: product quality variant. L = low (about 60 % of products), M = medium (30 %), H = high (10 %).
- Air temperature: ambient temperature around the machine, in Kelvin (K), around 300 K.
- Process temperature: temperature of the machining process, in Kelvin (K), about 10 K above air temperature.
- Rotational speed: spindle speed in revolutions per minute (rpm), calculated from a power of 2860 W with noise.
- Torque: torque in Newton-metres (Nm), normally distributed around 40 Nm.
- Tool wear: cumulative minutes (min) the tool has been used. H products add 5 min, M 3 min, L 2 min per process.
- Machine failure: 1 if the machine failed, else 0. A machine fails if any of the five failure modes occurs.
"""

FAILURE_MODES_TEXT = """FAILURE MODES - AI4I 2020 Predictive Maintenance Dataset

TWF (tool wear failure): the tool is replaced or fails at a randomly selected tool wear time between 200 and 240 minutes.

HDF (heat dissipation failure): heat dissipation causes a failure if the difference between process and air temperature is below 8.6 K and the rotational speed is below 1380 rpm.

PWF (power failure): the product of torque and rotational speed (in rad/s) is the power required for the process. If the power is below 3500 W or above 9000 W, the process fails.

OSF (overstrain failure): if the product of tool wear and torque exceeds 11,000 minNm for L products, 12,000 for M or 13,000 for H, the process fails by overstrain.

RNF (random failure): each process has a 0.1 % chance to fail regardless of its parameters.
"""

USAGE_NOTES = """HOW TO READ THE DATA

Each row is one machining process snapshot, not a time series, so the data shows conditions at the moment of failure rather than a warning hours ahead.
The failure-mode flags (TWF, HDF, PWF, OSF, RNF) say which failure happened; a machine can have more than one.
The dataset is synthetic: it was generated from the rules above (S. Matzka, 2020, UCI Machine Learning Repository).
"""


def build_documents(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(source, text) pairs. Statistics are computed from the real data."""
    n = len(df)
    f = int(df["Machine failure"].sum())
    stats = (f"DATASET SUMMARY - computed from the data\n\nRecords: {n}. Failed machines: {f} "
             f"({100 * f / n:.2f} % failure rate). Failure-mode counts: "
             + ", ".join(f"{m} {int(df[m].sum())}" for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]) + ".\n")
    return [("data_dictionary", DATA_DICTIONARY), ("failure_modes", FAILURE_MODES_TEXT),
            ("usage_notes", USAGE_NOTES), ("dataset_summary", stats)]


def split_paragraphs(source: str, text: str, max_chars: int = 600) -> list[tuple[str, str]]:
    """Split on blank lines (one idea per chunk), long paragraphs into max_chars pieces."""
    chunks = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        for i in range(0, len(para), max_chars):
            chunks.append((source, para[i:i + max_chars]))
    return chunks


@dataclass
class Hit:
    source: str
    text: str
    score: float


class KnowledgeBase:
    """FAISS inner-product index over normalised embeddings (= cosine similarity)."""

    def __init__(self, embeddings, documents: list[tuple[str, str]]):
        self.embeddings = embeddings
        self.chunks = [c for src, txt in documents for c in split_paragraphs(src, txt)]
        vectors = np.asarray(embeddings.embed_documents([t for _, t in self.chunks]), dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [Hit(self.chunks[i][0], self.chunks[i][1], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
