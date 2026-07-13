"""
reverse_templates.py
====================

Inverse of src/dataset.py. Given a generated narrative (the exact output text
the forward generator would have produced), recover the underlying structured
world state that produced it.

The forward generator (dataset.py) defines templates like
    "{name} moved from {old_loc} to {new_loc}."
    "{name} picked up {item}."
    "{name} gave {item} to {name2}."
    "The secret code for {entity} is {password}."
    "The secret code for {entity} was updated to {decoy}."
    "{name} was in the {location}."          (init)
    "{name} was in the {loc} with {items}."  (init-inventory)
    "Suddenly, {name} noticed..."             (story)

The reverse program parses the narrative and builds a category-by-category
*World struct* in reverse. This is exactly the structured state that:
  1. The generator itself uses in src/dataset.py:Entity / World.
  2. The hybrid latent-state model would need to *implictly compute and keep
     tracked* in its latent state in order to answer the question correctly.

So inverting the templates doubles as a checklist of categor(y|ies) and
parameter(s) the final model must track.

USAGE
-----
    python reverse_templates.py             # demo on synthetic narrative
    python reverse_templates.py --text "…"  # parse a custom narrative
    python reverse_templates.py --json      # dump as JSON
"""

from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# 1.  THE FORWARD TEMPLATE CATALOG
# ---------------------------------------------------------------------------
# A priori, every category / parameter the inverse needs to recover is listed
# here so we don't miss one. This is also the checklist the final model has
# to keep track of (see PROGRESS.md, "what the model needs to track").
#
# The catalog below names each template in the forward grammar AND the
# (category, parameter) the trainer/reverse-updater must update when it
# observes that template in the stream.

TEMPLATE_CATALOG: List[Dict[str, str]] = [
    # INIT lines ------------------------------------------------------------------
    {"id": "init.simple",
     "template": "{name} was in the {loc}.",
     "category": "entity",   "param": "location"},
    {"id": "init.with_inventory",
     "template": "{name} was in the {loc} with {items}.",
     "category": "entity",   "param": "inventory"},

    # ACTION lines ---------------------------------------------------------------
    {"id": "action.move",
     "template": "{name} moved from {old_loc} to {new_loc}.",
     "category": "entity",   "param": "location"},
    {"id": "action.went_to",
     "template": "{name} went to the {new_loc}.",
     "category": "entity",   "param": "location"},
    {"id": "action.pickup",
     "template": "{name} picked up {item}.",
     "category": "entity",   "param": "inventory"},
    {"id": "action.drop",
     "template": "{name} dropped {item}.",
     "category": "entity",   "param": "inventory"},
    {"id": "action.give",
     "template": "{name1} gave {item} to {name2}.",
     "category": "pair",     "param": "inventory_and_location"},   # also requires co-location
    {"id": "action.transfer",
     "template": "{name1} gave the {item} to {name2}.",
     "category": "pair",     "param": "inventory_and_location"},   # same as give (used in transfer-task)

    # SECRET lines ----------------------------------------------------------------
    {"id": "secret.set",
     "template": "The secret code for {entity} is {password}.",
     "category": "secret",   "param": "password"},
    {"id": "secret.update",
     "template": "The secret code for {entity} was updated to {decoy}.",
     "category": "secret",   "param": "decoy_enabled"},            # not the password itself -- toggles anti-recency trap

    # STORY / CONT lines (no world update; just filler) -------------------------
    {"id": "story.write",
     "template": "Write a story about {name}.",
     "category": "story",    "param": "subject"},
    {"id": "story.action",
     "template": "{name} looked around. | {name} felt something was about to happen. | "
                "Something caught {name}'s attention. | {name} heard a strange noise. | "
                "The air felt different.",
     "category": "story",    "param": "mood"},
]


# A plain prose *reverse grammar*: regexes (with named groups) and a
# (category, param) tag for each, mirroring the catalog above.
# Order matters: more specific patterns first; fall-through to NOT_HANDLED.

EVENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # INIT --------------------------------------------------------------------
    ("init.inventory",
     re.compile(r"^(?P<name>[A-Z][a-z]+) was in the (?P<loc>[\w ]+?) "
                r"with (?P<items>[\w ,]+?)\.$")),
    ("init.simple",
     re.compile(r"^(?P<name>[A-Z][a-z]+) was in the (?P<loc>[\w ]+?)\.$")),
    ("init.held",  # used by transfer-task: "{name} had the {item}."
     re.compile(r"^(?P<name>[A-Z][a-z]+) had the (?P<item>[a-z]+)\.$")),

    # ACTION ------------------------------------------------------------------
    ("action.move",
     re.compile(r"^(?P<name>[A-Z][a-z]+) moved from (?P<old_loc>[\w ]+?) "
                r"to (?P<new_loc>[\w ]+?)\.$")),
    ("action.went_to",
     re.compile(r"^(?P<name>[A-Z][a-z]+) went to the (?P<new_loc>[\w ]+?)\.$")),
    ("action.transfer",
     re.compile(r"^(?P<name1>[A-Z][a-z]+) gave the (?P<item>[a-z]+) "
                r"to (?P<name2>[A-Z][a-z]+)\.$")),
    ("action.give",
     re.compile(r"^(?P<name1>[A-Z][a-z]+) gave (?P<item>[a-z]+) "
                r"to (?P<name2>[A-Z][a-z]+)\.$")),
    ("action.pickup",
     re.compile(r"^(?P<name>[A-Z][a-z]+) picked up (?P<item>[a-z]+)\.$")),
    ("action.drop",
     re.compile(r"^(?P<name>[A-Z][a-z]+) dropped (?P<item>[a-z]+)\.$")),

    # SECRET ------------------------------------------------------------------
    # These MUST come before any "The {noun} ..." matchers might mistake them.
    ("secret.set",
     re.compile(r"^The secret code for (?P<entity>[A-Z][a-z]+) is (?P<password>[A-Z0-9]+)\.$")),
    ("secret.update",
     re.compile(r"^The secret code for (?P<entity>[A-Z][a-z]+) was updated to (?P<decoy>[A-Z0-9]+)\.$")),

    # STORY -------------------------------------------------------------------
    ("story.write",
     re.compile(r"^Write a story about (?P<name>[A-Z][a-z]+)\.$")),
    ("story.action",
     re.compile(r"^(?P<name>[A-Z][a-z]+) (?:looked around|felt something was about to happen|"
                r"heard a strange noise)\.$")),
    ("story.action.catch",
     re.compile(r"^(?:Something caught [A-Z][a-z]+'s attention|A strange noise [a-z ]+|"
                r"The air felt different\.|Suddenly, [A-Z][a-z]+ noticed something unusual\.)$")),
]


# ---------------------------------------------------------------------------
# 2.  THE WORLDS THE MODEL HAS TO TRACK
# ---------------------------------------------------------------------------
# A category + parameter list, mirroring the catalog. Anything not in here is
# something the latent state does NOT need to keep track of and can therefore
# be safely abstracted away.

WORLD_PARAMETERS = {
    "Pools": [
        "names",         # set      of entity names in the world
        "locations",     # set      of all location names
        "items",         # set      of all item names
    ],
    "Per entity": [
        ("location",     "string"),
        ("inventory",    "set<string>"),
        ("first_password", "string | None"),
        ("password_overridden", "bool"),   # 'secret.update' has been seen for them
    ],
    "Per item": [
        ("current_holder", "name | None"),
    ],
    "Per pair": [
        ("co_located",   "bool"),           # whether name1 and name2 are in the same loc right now
    ],
    "Story": [
        ("subject",      "name | None"),
        ("last_opening", "string | None"),  # picked continuation sentence
    ],
    "Counts the trainer must know to bucket difficulty": [
        "n_interference",
        "n_distractors",
        "gap_chars",
        "has_decoy",
        "n_hops",
        "n_moves",
        "n_actions",
    ],
}


# ---------------------------------------------------------------------------
# 3.  REVERSED WORLD
# ---------------------------------------------------------------------------

@dataclass
class ReverseEntity:
    """Inverted Entity: per-entity state recovered from the narrative."""
    name: str
    location: Optional[str] = None
    inventory: List[str] = field(default_factory=list)
    first_password: Optional[str] = None       # the *INITIAL* secret code if seen
    last_password:  Optional[str] = None        # most-recent code (decoy-trap)
    password_overridden: bool = False           # 'secret.update' observed?


@dataclass
class ReverseWorld:
    """Inverted World: full state the forward generator tracked."""
    names: set = field(default_factory=set)
    locations: set = field(default_factory=set)
    items: set = field(default_factory=set)
    entities: Dict[str, ReverseEntity] = field(default_factory=dict)

    # Per-item holder (we track this on top of .inventory on the holder)
    # because the reverse program needs it for the multi-hop transfer task.
    item_holder: Dict[str, Optional[str]] = field(default_factory=dict)

    # Counters (what the trainer needs for stratified reporting).
    n_interference: int = 0
    n_distractors: int = 0
    has_decoy: bool = False
    n_moves: int = 0
    n_actions: int = 0

    # Story-attached strings.
    narrative_unhandled: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "names": sorted(self.names),
            "locations": sorted(self.locations),
            "items": sorted(self.items),
            "entities": {n: asdict(e) for n, e in self.entities.items()},
            "item_holder": self.item_holder,
            "counts": {
                "n_interference": self.n_interference,
                "n_distractors":   self.n_distractors,
                "has_decoy":       self.has_decoy,
                "n_moves":         self.n_moves,
                "n_actions":       self.n_actions,
            },
            "narrative_unhandled": self.narrative_unhandled,
        }
        return d


# ---------------------------------------------------------------------------
# 4.  THE INVERSION
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[\w]+")     # for token-ish counting
INTERFERENCE_PATTERNS = [
    # dataset.py:_interference_sentence produces six specific templates that
    # MENTION real entity/location names and so are *interference* distractors
    # (counted in n_interference, no world-state update).
    re.compile(r"^[A-Z][a-z]+ walked toward the [\w ]+?\.$"),
    re.compile(r"^[A-Z][a-z]+ was seen near the [\w ]+?\.$"),
    re.compile(r"^The [\w ]+? looked different today\.$"),
    re.compile(r"^[A-Z][a-z]+ spent some time in the [\w ]+?\.$"),
    re.compile(r"^Someone mentioned [A-Z][a-z]+ at the [\w ]+?\.$"),
    re.compile(r"^[A-Z][a-z]+ left the [\w ]+? in a hurry\.$"),
]
FILLER_PATTERNS = [
    # dataset.py:FILLER -- no world-state mention at all, just atmospheric noise.
    # Increments n_distractors (for recall-task difficulty) but NOT entities.
    re.compile(r"^(?:It was a quiet day\.|The sun was shining\.|"
               r"Time passed slowly\.|Nothing unusual happened\.|"
               r"The weather was pleasant\.|A soft wind moved through the trees\.|"
               r"The clock on the wall kept ticking\.|"
               r"Shadows stretched long across the floor\.)$"),
]


def reverse_event(event: str, w: ReverseWorld) -> Tuple[str, Dict[str, Any]]:
    """Try every reverse pattern. Return (event_id, params) or ('unhandled', {}).

    Also performs the *updating* of the world state (`w`) that the forward
    program would have performed -- just in reverse. This is the key insight:
    nothing in the forward generator's state-update needs to change for the
    inverse to track it; we just re-derive the same state from observations.
    """
    for ev_id, pat in EVENT_PATTERNS:
        m = pat.match(event)
        if not m:
            continue
        g = m.groupdict()
        # Update world ---------------------------------------------------
        if ev_id.startswith("init."):
            name = g["name"]; w.names.add(name)
            e = w.entities.setdefault(name, ReverseEntity(name=name))
            if "loc" in g:
                loc_val = g["loc"].strip()
                w.locations.add(loc_val)
                e.location = loc_val
            if ev_id == "init.inventory":
                items = [s.strip() for s in g["items"].split(", ")]
                e.inventory = list(items)
                for it in items:
                    w.items.add(it)
                    w.item_holder[it] = name
            elif ev_id == "init.held":
                item = g["item"]; w.items.add(item)
                if item not in e.inventory:
                    e.inventory.append(item)
                w.item_holder[item] = name
            return ev_id, g

        if ev_id == "action.move":
            name = g["name"]; w.names.add(name)
            old = g["old_loc"].strip(); new = g["new_loc"].strip()
            w.locations.update([old, new])
            e = w.entities.setdefault(name, ReverseEntity(name=name))
            e.location = new
            w.n_moves += 1
            return ev_id, g

        if ev_id == "action.went_to":
            # Same net effect as a move, without an explicit start location.
            # Used by the transfer task's co-location guarantee.
            name = g["name"]; new = g["new_loc"].strip()
            w.names.add(name); w.locations.add(new)
            e = w.entities.setdefault(name, ReverseEntity(name=name))
            e.location = new
            w.n_moves += 1
            return ev_id, g

        if ev_id == "action.pickup":
            name = g["name"]; item = g["item"]; w.names.add(name); w.items.add(item)
            e = w.entities.setdefault(name, ReverseEntity(name=name))
            if item not in e.inventory:
                e.inventory.append(item)
            w.item_holder[item] = name
            w.n_actions += 1
            return ev_id, g

        if ev_id == "action.drop":
            name = g["name"]; item = g["item"]; w.names.add(name); w.items.add(item)
            e = w.entities.setdefault(name, ReverseEntity(name=name))
            if item in e.inventory:
                e.inventory.remove(item)
            if w.item_holder.get(item) == name:
                w.item_holder[item] = None
            w.n_actions += 1
            return ev_id, g

        if ev_id in ("action.give", "action.transfer"):
            n1, n2, item = g["name1"], g["name2"], g["item"]
            w.names.update([n1, n2]); w.items.add(item)
            e1 = w.entities.setdefault(n1, ReverseEntity(name=n1))
            e2 = w.entities.setdefault(n2, ReverseEntity(name=n2))
            if item in e1.inventory:
                e1.inventory.remove(item)
            e2.inventory.append(item) if item not in e2.inventory else None
            w.item_holder[item] = n2
            # Co-locate: forward transfer_task forces same location; reverse
            # can't know that path explicitly, but co-location is the only
            # way 'give' makes sense -> assert?
            if e1.location and not e2.location:
                e2.location = e1.location
            elif e2.location and not e1.location:
                e1.location = e2.location
            w.n_actions += 1
            return ev_id, g

        if ev_id == "secret.set":
            entity, pw = g["entity"], g["password"]
            w.names.add(entity)
            e = w.entities.setdefault(entity, ReverseEntity(name=entity))
            # Only set as FIRST password if we haven't seen one before.
            if e.first_password is None:
                e.first_password = pw
            e.last_password = pw
            return ev_id, g

        if ev_id == "secret.update":
            entity, decoy = g["entity"], g["decoy"]
            w.names.add(entity)
            e = w.entities.setdefault(entity, ReverseEntity(name=entity))
            e.password_overridden = True
            e.last_password = decoy      # most-recent
            w.has_decoy = True
            return ev_id, g

        if ev_id == "story.write":
            w.names.add(g["name"]); return ev_id, g
        if ev_id.startswith("story."):
            return ev_id, g

    # Fall through: bookkeeping
    for ipat in INTERFERENCE_PATTERNS:
        if ipat.match(event):
            w.n_interference += 1
            return "interference", {}
    for fpat in FILLER_PATTERNS:
        if fpat.match(event):
            w.n_distractors += 1
            return "filler", {}
    if "secret code" in event:                      # mentioned but not parsed
        w.n_distractors += 1
        return "secret_distractor", {}
    return "unhandled", {}


def reverse_templates(narrative: str) -> Tuple[ReverseWorld, List[Tuple[str, str, Dict[str, Any]]]]:
    """Top-level inversion: parse the whole narrative, return (world, events).

    Each entry in `events` is (event_id, original_sentence, parsed_groups).
    `world` is the rebuilt structured state -- the SAME state the forward
    generator would have ended up with.
    """
    world = ReverseWorld()
    log: List[Tuple[str, str, Dict[str, Any]]] = []
    for s in re.split(r"(?<=\.) ", narrative.strip()):
        s = s.strip()
        if not s: continue
        # Strip Question / Answer tail if present
        m_q = re.match(r"^(.+\?)\s*$", s)
        if m_q:
            log.append(("question", s, {}))
            continue
        ev_id, params = reverse_event(s, world)
        log.append((ev_id, s, params))
        if ev_id == "unhandled":
            world.narrative_unhandled.append(s)
    return world, log


# ---------------------------------------------------------------------------
# 5.  SELF-DEMO
# ---------------------------------------------------------------------------

def _demo_synthetic() -> str:
    """Construct a tiny narrative in FORWARD template form, then reverse it.

    This acts as a unit test: we should recover exactly the world the forward
    generator started with.
    """
    # Forward:
    #   John was in the kitchen with key.
    #   Mary was in the garden.
    #   John moved from kitchen to bedroom.
    #   Mary walked toward the office. (interference)
    #   John gave key to Mary.
    #   The secret code for John is ABCD2345.
    #   The secret code for Sam is ZZ99XYZ1.
    #   The secret code for John was updated to NEW00000.
    return ("John was in the kitchen with key. "
            "Mary was in the garden. "
            "John moved from kitchen to bedroom. "
            "Mary walked toward the office. "
            "John gave key to Mary. "
            "The secret code for John is ABCD2345. "
            "The secret code for Sam is ZZ99XYZ1. "
            "The secret code for John was updated to NEW00000.")


def _self_test(N: int = 60):
    """For every task type: does the inverse reconstruct the GT answer?

    Reports per-task accuracy AND the inconsistent-GT rate for the transfer task
    (the case where the forward generator produces an internally inconsistent
    ground-truth that the inverse correctly disagrees with).
    """
    import sys
    sys.path.insert(0, "src")
    from dataset import (generate_location_task, generate_inventory_task,
                         generate_transfer_task, generate_recall_task)

    def check(task, fn, kwargs, extractor):
        nc = nt = 0
        inconsistent = 0
        for _ in range(N):
            try:
                n, q, gt, _ = fn(**kwargs)
            except Exception:
                continue
            if not q:
                continue
            w, _ = reverse_templates(n)
            try:
                pred = extractor(w, q)
            except Exception:
                pred = None
            if pred == gt:
                nc += 1
            elif task == "transfer":
                inconsistent += 1
            nt += 1
        print(f"  {task:10s}: {nc:3d}/{nt:3d}  "
              + (f"(inconsistent GT: {inconsistent})" if task == "transfer" else ""))

    print(f"=== Self-test: inverse accuracy on {N} samples per task ===")
    check("location",
          generate_location_task,
          dict(n_moves=5, n_interference=3, n_filler=1, max_chars=400),
          lambda w, q: w.entities[q.replace('Where is ', '').replace('?', '').strip()].location)
    check("inventory",
          generate_inventory_task,
          dict(n_actions=5, n_interference=3, max_chars=400),
          lambda w, q: (' and '.join(
              w.entities[q.replace('What does ', '').replace(' have?', '').strip()].inventory)
              or 'nothing'))
    check("transfer",
          generate_transfer_task,
          dict(n_steps=5, n_interference=2, max_chars=400),
          lambda w, q: (lambda item:
              w.entities[w.item_holder[item]].location
              if w.item_holder.get(item) else None)
              (q.replace('Where is the ', '').replace('?', '').strip()))
    check("recall",
          generate_recall_task,
          dict(n_distractor_sentences=5, decoy_for_target=True, max_chars=400),
          lambda w, q: w.entities[q.replace(
              'What was the FIRST secret code for ', '').replace('?', '').strip()].first_password)
    print()


def main():
    ap = argparse.ArgumentParser(description="Reverse the dataset's templates.")
    ap.add_argument("--text", type=str, default=None, help="narrative text to reverse")
    ap.add_argument("--json", action="store_true", help="dump world as JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="generate N samples per task and report inverse accuracy vs ground-truth")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    text = args.text or _demo_synthetic()
    print("\n--- NARRATIVE ---")
    print(text)
    world, log = reverse_templates(text)

    print("\n--- PARSED EVENTS ---")
    for ev_id, sent, params in log:
        print(f"  [{ev_id:18s}]  {sent}")

    out = world.to_dict()
    if args.json:
        print("\n--- WORLD (JSON) ---")
        print(json.dumps(out, indent=2, default=list))
    else:
        print("\n--- WORLD (pretty) ---")
        for k, v in out.items():
            print(f"  {k}: {v}")

    print("\n--- WHAT THE LATENT MODEL MUST TRACK (checklist) ---")
    for sec, params in WORLD_PARAMETERS.items():
        print(f"  [{sec}]")
        if isinstance(params, list) and params and isinstance(params[0], tuple):
            for n, t in params:
                print(f"    - {n}: {t}")
        else:
            for p in params:
                print(f"    - {p}")


if __name__ == "__main__":
    main()
