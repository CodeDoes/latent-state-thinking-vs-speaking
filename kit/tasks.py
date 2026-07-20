#!/usr/bin/env python3
"""Synthetic task generators — learnable patterns for smoke tests and ablations.

These exist so a smoke test can answer "can this architecture learn at all"
in under a minute on CPU. Tasks are token-id sequences: 0 = pad, 1 reserved
for TRIGGER, 2..257 = raw bytes.

[meta]
status: active
[/meta]
"""

import random

TASKS = {}


def task(name):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


@task("arith")
def arith(n=2000, seed=42):
    """Incrementing bytes: predict the next byte."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        start = rng.randint(0, 200)
        length = rng.randint(16, 48)
        data.append([((start + i) % 256) + 2 for i in range(length)])
    return data


@task("patch_repeat")
def patch_repeat(n=2000, seed=42):
    """Patches that repeat (the model must detect the repetition period)."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        ps = rng.choice([4, 8])
        np_ = rng.randint(3, 6)
        patch = [rng.randint(2, 257) for _ in range(ps)]
        data.append(list(patch * np_))
    return data


@task("grid")
def grid(n=500, seed=42):
    """2D spatial pattern flattened to 1D."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        gs = rng.choice([4, 8])
        data.append([(rng.randint(2, 10)) + 2 for _ in range(gs * gs)])
    return data


def load(name, **kw):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {sorted(TASKS)}")
    return TASKS[name](**kw)
