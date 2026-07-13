"""
Toy world generator for synthetic reasoning tasks.

Tasks:
  1. location_tracking - where is X after a long, *interfering* sequence of moves?
  2. inventory_tracking - what does X hold after pickups/drops?
  3. transfer          - multi-hop: where is the item after it changed hands + moved?
  4. exact_recall      - remember a password seen far earlier amid decoys (anti-recency)
  5. story_continuation

Design notes (see AGENTS.md: "too easy / low loss but useless output"):
  * Interfering distractors use REAL entity names and locations, so a model that
    just grabs "the last location word it saw" or attends to the most recent
    mention will be actively misled. This makes low loss require real tracking.
  * Every QA sample is emitted in a strict "Answer: <X>" slot. Evaluation can
    therefore do EXACT MATCH on the generated answer rather than a lazy substring
    check, which is what previously hid useless output behind a low loss.
  * Each sample carries a `meta` dict (difficulty/structure) so evaluation can
    break accuracy down by CONDITION (interference level, recall gap, decoy
    present, multi-hop depth) -- far more informative than a single number.
"""

import random
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# --- shared pools ---------------------------------------------------------

FILLER = [
    "It was a quiet day.",
    "The sun was shining.",
    "Time passed slowly.",
    "Nothing unusual happened.",
    "The weather was pleasant.",
    "A soft wind moved through the trees.",
    "The clock on the wall kept ticking.",
    "Shadows stretched long across the floor.",
]

# Literal label used to anchor the answer slot in training and evaluation.
ANSWER_LABEL = "Answer:"


def _interference_sentence(world) -> str:
    """Distractor that mentions a REAL entity/location to create interference."""
    name = random.choice(list(world.entities.keys()))
    loc = random.choice(world.locations)
    templates = [
        f"{name} walked toward the {loc}.",
        f"{name} was seen near the {loc}.",
        f"The {loc} looked different today.",
        f"{name} spent some time in the {loc}.",
        f"Someone mentioned {name} at the {loc}.",
        f"{name} left the {loc} in a hurry.",
    ]
    return random.choice(templates)


@dataclass
class Entity:
    name: str
    location: str
    inventory: List[str] = field(default_factory=list)


@dataclass
class World:
    entities: Dict[str, Entity] = field(default_factory=dict)
    locations: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    def __init__(self, names=None, locations=None, items=None):
        self.locations = locations or [
            "kitchen", "bedroom", "garden", "garage", "bathroom",
            "living room", "office", "basement", "attic", "hallway"
        ]
        self.items = items or [
            "apple", "book", "key", "phone", "cup", "pen",
            "wallet", "watch", "bag", "umbrella"
        ]
        self.names = names or [
            "John", "Mary", "Alex", "Sam", "Emma", "Leo",
            "Zoe", "Max", "Lily", "Tom"
        ]
        self.entities = {}
        self.history = []
        self._init_world()

    def _init_world(self):
        names_subset = random.sample(self.names, k=random.randint(3, 5))
        for i, name in enumerate(names_subset):
            loc = random.choice(self.locations)
            if i == 0:
                inv = random.sample(self.items, k=random.randint(1, 2))
            else:
                inv = random.sample(self.items, k=random.randint(0, 2))
            self.entities[name] = Entity(name=name, location=loc, inventory=inv)

    def _narrate(self, sentence: str):
        self.history.append(sentence)

    def move_entity(self, name: str) -> str:
        entity = self.entities[name]
        new_loc = random.choice([l for l in self.locations if l != entity.location])
        old_loc = entity.location
        entity.location = new_loc
        sentence = f"{name} moved from {old_loc} to {new_loc}."
        self._narrate(sentence)
        return sentence

    def pickup_item(self, name: str) -> str:
        entity = self.entities[name]
        available = [i for i in self.items if i not in entity.inventory]
        if not available:
            return ""
        item = random.choice(available)
        entity.inventory.append(item)
        sentence = f"{name} picked up {item}."
        self._narrate(sentence)
        return sentence

    def drop_item(self, name: str) -> str:
        entity = self.entities[name]
        if not entity.inventory:
            return ""
        item = random.choice(entity.inventory)
        entity.inventory.remove(item)
        sentence = f"{name} dropped {item}."
        self._narrate(sentence)
        return sentence

    def give_item(self, name1: str, name2: str) -> str:
        e1 = self.entities[name1]
        e2 = self.entities[name2]
        if not e1.inventory or e1.location != e2.location:
            return ""
        item = random.choice(e1.inventory)
        e1.inventory.remove(item)
        e2.inventory.append(item)
        sentence = f"{name1} gave {item} to {name2}."
        self._narrate(sentence)
        return sentence

    def transfer_item(self, item: str, from_name: str, to_name: str) -> str:
        """Force a transfer (co-locating if needed) and narrate it.

        Used by the multi-hop transfer task where the answer depends on the
        item's final holder's location.
        """
        if item not in self.entities[from_name].inventory:
            return ""
        if self.entities[from_name].location != self.entities[to_name].location:
            self.entities[to_name].location = self.entities[from_name].location
        self.entities[from_name].inventory.remove(item)
        self.entities[to_name].inventory.append(item)
        sentence = f"{from_name} gave the {item} to {to_name}."
        self._narrate(sentence)
        return sentence

    def generate_password(self) -> Tuple[str, str]:
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        password = "".join(random.choices(chars, k=8))
        entity = random.choice(list(self.entities.keys()))
        sentence = f"The secret code for {entity} is {password}."
        self._narrate(sentence)
        return password, entity

    def _interference(self) -> str:
        return _interference_sentence(self)


def generate_location_task(
    n_moves: int = 10,
    n_interference: int = 8,
    n_filler: int = 3,
    max_chars: int = 600,
) -> Tuple[str, str, str, dict]:
    """Location tracking with interfering mentions of other entities/places."""
    world = World()
    sentences = [
        f"{name} was in the {entity.location}."
        for name, entity in world.entities.items()
    ]

    names = list(world.entities.keys())
    n_interf = 0
    for _ in range(n_moves):
        name = random.choice(names)
        action = random.choice(["move", "pickup", "drop"])
        if action == "move":
            sentences.append(world.move_entity(name))
        elif action == "pickup":
            s = world.pickup_item(name)
            if s:
                sentences.append(s)
        else:
            s = world.drop_item(name)
            if s:
                sentences.append(s)
        if random.random() < 0.5:
            sentences.append(world._interference())
            n_interf += 1
        if len(" ".join(sentences)) > max_chars:
            break

    for _ in range(n_filler):
        sentences.append(random.choice(FILLER))
        if len(" ".join(sentences)) > max_chars:
            break

    narrative = " ".join(s for s in sentences if s)
    target = random.choice(names)
    question = f"Where is {target}?"
    answer = world.entities[target].location
    meta = {"n_moves": n_moves, "n_interference": n_interf}
    return narrative, question, answer, meta


def generate_inventory_task(
    n_actions: int = 6,
    n_interference: int = 4,
    max_chars: int = 600,
) -> Tuple[str, str, str, dict]:
    """Inventory tracking with interfering mentions."""
    world = World()
    sentences = []

    for name, entity in world.entities.items():
        if entity.inventory:
            items_str = ", ".join(entity.inventory)
            sentences.append(f"{name} was in the {entity.location} with {items_str}.")
        else:
            sentences.append(f"{name} was in the {entity.location}.")

    names = list(world.entities.keys())
    n_interf = 0
    for _ in range(n_actions):
        name = random.choice(names)
        action = random.choice(["move", "pickup", "drop"])
        if action == "move":
            sentences.append(world.move_entity(name))
        elif action == "pickup":
            s = world.pickup_item(name)
            if s:
                sentences.append(s)
        else:
            s = world.drop_item(name)
            if s:
                sentences.append(s)
        if random.random() < 0.4:
            sentences.append(world._interference())
            n_interf += 1
        if len(" ".join(sentences)) > max_chars:
            break

    narrative = " ".join(s for s in sentences if s)
    entities_with_items = [n for n in names if world.entities[n].inventory]
    target = random.choice(entities_with_items) if entities_with_items else random.choice(names)
    question = f"What does {target} have?"
    answer = " and ".join(world.entities[target].inventory) if world.entities[target].inventory else "nothing"
    meta = {"n_actions": n_actions, "n_interference": n_interf}
    return narrative, question, answer, meta


def generate_transfer_task(
    n_steps: int = 7,
    n_interference: int = 6,
    max_chars: int = 600,
) -> Tuple[str, str, str, dict]:
    """Multi-hop: track an item as it changes hands and its holder moves.

    The answer (the item's final location) requires combining two reasoning
    steps: who currently holds the item, and where that holder ended up.

    CO-LOCATION GUARANTEE: every "give"/"transfer" event must be preceded in
    the narrative by an explicit sentence that places giver and recipient in
    the SAME room. If the call to `world.transfer_item(...)` triggered the
    auto-co-location branch (i.e. they were originally in different rooms),
    we *also* emit a "{recipient} went to the {giver_location}." sentence so
    the narrative is self-consistent -- otherwise the question
    "Where is the {item}?" is ill-posed (the recipient's location is
    under-determined by the narrative). This eliminates the ambiguous-Answer
    class that previously caused inverse vs. GT disagreements.
    """
    world = World()
    sentences = [
        f"{name} was in the {entity.location}."
        for name, entity in world.entities.items()
    ]

    names = list(world.entities.keys())
    item = random.choice(world.items)
    holder = random.choice(names)
    world.entities[holder].inventory.append(item)
    sentences.append(f"{holder} had the {item}.")

    n_interf = 0
    for _ in range(n_steps):
        r = random.random()
        if r < 0.45:
            recip = random.choice([n for n in names if n != holder])
            # Look up the giver's location BEFORE the transfer mutates state.
            giver_loc = world.entities[holder].location
            recip_loc = world.entities[recip].location
            s = world.transfer_item(item, holder, recip)
            if s:
                # If the transfer silently re-located the recipient (different
                # rooms before), narrate that move FIRST so the narrative
                # fully determines the answer.
                if recip_loc != giver_loc:
                    reloc = f"{recip} went to the {giver_loc}."
                    sentences.append(reloc)
                sentences.append(s)
                holder = recip
            else:
                sentences.append(world.move_entity(holder))
        else:
            sentences.append(world.move_entity(holder))
        if random.random() < 0.5:
            sentences.append(world._interference())
            n_interf += 1
        if len(" ".join(sentences)) > max_chars:
            break

    narrative = " ".join(s for s in sentences if s)
    question = f"Where is the {item}?"
    answer = world.entities[holder].location
    meta = {"n_hops": n_steps, "n_interference": n_interf}
    return narrative, question, answer, meta


def generate_recall_task(
    n_distractor_sentences: int = 40,
    decoy_for_target: bool = True,
    max_chars: int = 600,
) -> Tuple[str, str, str, dict]:
    """Exact recall with a long gap, decoy codes, and an anti-recency trap.

    The target code is shown early. The distractor stream contains many other
    codes (for other entities) plus interfering filler. If decoy_for_target is
    set, the SAME entity's code is later "updated", and the question asks for
    the FIRST code -- so a model relying on recency will be wrong.

    Distractors are added until the narrative reaches `max_chars`, which keeps
    the whole sample (narrative + question + answer) inside a small char-level
    context window instead of being silently truncated.
    """
    world = World()
    sentences = [
        f"{name} was in the {entity.location}."
        for name, entity in world.entities.items()
    ]

    password, entity_name = world.generate_password()
    sentences.append(f"The secret code for {entity_name} is {password}.")

    other_entities = [n for n in world.entities if n != entity_name]
    n_added = 0
    for _ in range(n_distractor_sentences):
        r = random.random()
        if r < 0.3 and other_entities:
            p2, e2 = world.generate_password()
            sentences.append(f"The secret code for {e2} is {p2}.")
        elif r < 0.6:
            sentences.append(world._interference())
        else:
            sentences.append(random.choice(FILLER))
        n_added += 1
        if len(" ".join(sentences)) > max_chars:
            break

    gap_chars = len(" ".join(sentences))  # distance from code to question

    if decoy_for_target:
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        decoy = "".join(random.choices(chars, k=8))
        sentences.append(f"The secret code for {entity_name} was updated to {decoy}.")
        question = f"What was the FIRST secret code for {entity_name}?"
    else:
        question = f"What is the secret code for {entity_name}?"

    answer = password
    narrative = " ".join(s for s in sentences if s)
    meta = {
        "n_distractors": n_added,
        "gap_chars": gap_chars,
        "has_decoy": decoy_for_target,
    }
    return narrative, question, answer, meta


def generate_story_prompt(
    n_setup_sentences: int = 3,
) -> Tuple[str, str, dict]:
    """Story continuation prompt + ground-truth opening line."""
    world = World()
    sentences = []

    name = list(world.entities.keys())[0]
    entity = world.entities[name]

    sentences.append(f"Write a story about {name}.")
    sentences.append(f"{name} was in the {entity.location}.")

    for _ in range(n_setup_sentences - 1):
        actions = [
            f"{name} looked around.",
            f"{name} felt something was about to happen.",
            f"Something caught {name}'s attention.",
            f"{name} heard a strange noise.",
            f"The air felt different.",
        ]
        sentences.append(random.choice(actions))

    prompt = " ".join(sentences)

    continuations = [
        f"Suddenly, {name} noticed something unusual.",
        f"Then {name} decided to explore further.",
        f"Out of nowhere, a mysterious figure appeared.",
        f"At that moment, everything changed.",
    ]
    ground_truth = random.choice(continuations)
    meta = {}
    return prompt, ground_truth, meta


# --- formatting helpers ---------------------------------------------------
# These guarantee train/eval use the SAME surface form so the model learns to
# emit its answer in the "Answer:" slot, enabling strict exact-match scoring.

def format_for_training(sample: dict) -> str:
    """Full text the model is trained on (narrative + question + answer)."""
    if sample["task_type"] == "story":
        return f"{sample['narrative']}\n{sample['answer']}"
    return (
        f"{sample['narrative']}\n"
        f"Question: {sample['question']}\n"
        f"{ANSWER_LABEL} {sample['answer']}"
    )


def build_prompt(sample: dict) -> str:
    """Prompt fed at inference: narrative + question + 'Answer: '.

    Ends with a trailing space so the surface form exactly matches
    format_for_training (which writes 'Answer: <ans>'); otherwise the model
    sees a context at inference it never encountered during training.
    """
    if sample["task_type"] == "story":
        return sample["narrative"] + "\n"
    return (
        f"{sample['narrative']}\n"
        f"Question: {sample['question']}\n"
        f"{ANSWER_LABEL} "
    )


def parse_answer(generated: str) -> str:
    """Extract the predicted answer from generated text (before any newline)."""
    text = generated.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if text.lower().startswith(ANSWER_LABEL.lower()):
        text = text[len(ANSWER_LABEL):].strip()
    return text


def _bucket(sample: dict) -> str:
    """Coarse difficulty bucket for stratified accuracy reporting."""
    t = sample["task_type"]
    m = sample.get("meta", {})
    if t in ("location", "inventory", "transfer"):
        ni = m.get("n_interference", 0)
        if ni == 0:
            return "interf=0"
        if ni <= 2:
            return "interf=1-2"
        return "interf>=3"
    if t == "recall":
        if m.get("has_decoy"):
            return "decoy=yes"
        return "decoy=no"
    return "all"


def generate_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    task_weights: Optional[Dict[str, float]] = None,
    location_max_chars: int = 600,
    inventory_max_chars: int = 600,
    transfer_max_chars: int = 600,
    recall_max_chars: int = 600,
) -> List[Dict]:
    """Generate a mixed dataset of reasoning tasks.

    `*-max-chars` controls narrative length per task type (so the longest
    sample still fits the model's context window). For a quick CPU run you can
    pass smaller values (e.g. recall_max_chars=400) to keep context short.
    """
    random.seed(seed)

    if task_weights is None:
        task_weights = {
            "location": 0.3,
            "inventory": 0.2,
            "transfer": 0.2,
            "recall": 0.2,
            "story": 0.1,
        }

    tasks = list(task_weights.keys())
    weights = list(task_weights.values())

    dataset = []
    for _ in range(n_samples):
        task_type = random.choices(tasks, weights=weights, k=1)[0]

        if task_type == "location":
            narrative, question, answer, meta = generate_location_task(max_chars=location_max_chars)
        elif task_type == "inventory":
            narrative, question, answer, meta = generate_inventory_task(max_chars=inventory_max_chars)
        elif task_type == "transfer":
            narrative, question, answer, meta = generate_transfer_task(max_chars=transfer_max_chars)
        elif task_type == "recall":
            narrative, question, answer, meta = generate_recall_task(max_chars=recall_max_chars)
        elif task_type == "story":
            narrative, answer, meta = generate_story_prompt()
            question = ""
        else:
            continue

        dataset.append({
            "narrative": narrative,
            "question": question,
            "answer": answer,
            "task_type": task_type,
            "meta": meta,
        })

    return dataset


if __name__ == "__main__":
    dataset = generate_dataset(n_samples=8, seed=42)
    for i, sample in enumerate(dataset):
        print(f"\n=== Sample {i+1} [{sample['task_type']}] meta={sample['meta']} ===")
        print(f"Narrative: {sample['narrative']}")
        if sample["question"]:
            print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"-- train text --\n{format_for_training(sample)}")
