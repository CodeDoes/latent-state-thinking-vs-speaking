"""Fast synthetic multi-query world task.

One context (a short move chain with distractors) + several single-token
questions. Answers are always a single vocabulary symbol, so training and
exact-match are cheap and unambiguous. Designed for tiny CPU-runnable models.

Tasks:
  WHERE E    -> final location of E           (long-horizon last-mention)
  AT   E L   -> YES/NO: is E at location L?    (membership / aggregation)
  SAME E     -> YES/NO: does another entity share E's location? (aggregation)
"""

import random


def build_vocab():
    cats = {
        "name": [f"N{i}" for i in range(10)],
        "loc": [f"L{i}" for i in range(10)],
        "rel": ["WHERE", "AT", "SAME", "YES", "NO"],
    }
    syms = [s for v in cats.values() for s in v]
    return syms, cats


def gen_world(cats, rng, max_events=8):
    locs = cats["loc"]
    names = rng.sample(cats["name"], rng.randint(3, 5))
    loc_of = {nm: rng.choice(locs) for nm in names}

    events = []
    n_moves = rng.randint(max(3, max_events // 2), max_events)
    for _ in range(n_moves):
        it = rng.choice(names)
        newloc = rng.choice(locs)
        events.append((it, newloc))
        loc_of[it] = newloc
        if rng.random() < 0.5:  # distractor move (real entity/loc, misleads)
            it2 = rng.choice([x for x in names if x != it])
            l2 = rng.choice(locs)
            events.append((it2, l2))
            loc_of[it2] = l2

    context = [t for (it, l) in events for t in (it, l)]

    # guarantee at least one co-located pair so SAME has non-NO answers
    if not any(loc_of[x] == loc_of[y] for x in names for y in names if x != y):
        b = rng.choice(names)
        a = rng.choice([x for x in names if x != b])
        loc_of[a] = loc_of[b]

    queries = []
    K = rng.randint(4, 8)
    for _ in range(K):
        qt = rng.choices(["WHERE", "AT", "SAME"], weights=[0.4, 0.3, 0.3])[0]
        if qt == "WHERE":
            e = rng.choice(names)
            queries.append((["WHERE", e], loc_of[e]))
        elif qt == "AT":
            e = rng.choice(names)
            lk = rng.choice(locs)
            ans = "YES" if loc_of[e] == lk else "NO"
            queries.append((["AT", e, lk], ans))
        else:  # SAME
            e = rng.choice(names)
            ans = "YES" if any(x != e and loc_of[x] == loc_of[e] for x in names) else "NO"
            queries.append((["SAME", e], ans))
    return {"context": context, "queries": queries, "loc_of": loc_of, "names": names}


def generate_dataset(n=2000, seed=42, max_events=8):
    syms, cats = build_vocab()
    rng = random.Random(seed)
    return [gen_world(cats, rng, max_events) for _ in range(n)]
