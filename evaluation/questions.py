"""20 evaluation questions with ground truth.

Expected numbers are computed here directly from the CSV with plain pandas,
independently of agent/analytics.py, so a bug in the tools shows up as a
failed question instead of being repeated in the answer key.

kind="number": the answer must contain the expected number (within tolerance).
kind="keywords": every group must be matched by at least one of its words.
"""
from __future__ import annotations

import math

import pandas as pd


def _df(path):
    return pd.read_csv(path)


def build_questions(csv_path) -> list[dict]:
    df = _df(csv_path)
    failed = df[df["Machine failure"] == 1]
    normal = df[df["Machine failure"] == 0]
    rate = lambda d: 100.0 * d["Machine failure"].mean()
    power = df["Torque"] * df["Rotational speed"] * 2 * math.pi / 60

    q = [
        ("How many records are in the dataset?", "number", len(df), 0),
        ("How many machines failed?", "number", int(df["Machine failure"].sum()), 0),
        ("What is the overall failure rate in percent?", "number", rate(df), 0.05),
        ("How many heat dissipation failures (HDF) are there?", "number", int(df["HDF"].sum()), 0),
        ("How many power failures occurred?", "number", int(df["PWF"].sum()), 0),
        ("What is the average torque of failed machines?", "number", failed["Torque"].mean(), 0.1),
        ("What is the average tool wear of machines that did not fail?", "number", normal["Tool wear"].mean(), 0.1),
        ("What is the failure rate for low-quality (L) products in percent?", "number", rate(df[df["Type"] == "L"]), 0.05),
        ("What is the failure rate of type H products?", "number", rate(df[df["Type"] == "H"]), 0.05),
        ("What is the highest rotational speed in the data?", "number", df["Rotational speed"].max(), 0),
        ("What is the median rotational speed of failed machines?", "number", failed["Rotational speed"].median(), 0.5),
        ("How many machines with tool wear above 200 minutes failed?", "number",
         int(df[(df["Tool wear"] > 200)]["Machine failure"].sum()), 0),
        ("How many type M machines failed?", "number", int(failed[failed["Type"] == "M"].shape[0]), 0),
        ("What is the average process temperature of machines with an overstrain failure?", "number",
         df[df["OSF"] == 1]["Process temperature"].mean(), 0.1),
        ("How many machines have a torque above 60 Nm?", "number", int((df["Torque"] > 60).sum()), 0),
        ("What is the average power in watts of failed machines?", "number",
         power[df["Machine failure"] == 1].mean(), 5.0),
        ("Which product type has the highest failure rate?", "keywords", [["L", "low"]], None),
        ("What causes a heat dissipation failure?", "keywords", [["temperature"], ["speed", "rpm"]], None),
        ("What does OSF stand for?", "keywords", [["overstrain", "over-strain", "over strain"]], None),
        ("What was the failure rate in the year 2019?", "keywords",
         [["not", "no ", "doesn't", "does not", "cannot", "unavailable", "isn't"]], None),
    ]
    return [{"id": i + 1, "question": text, "kind": kind, "expected": exp, "tolerance": tol}
            for i, (text, kind, exp, tol) in enumerate(q)]
