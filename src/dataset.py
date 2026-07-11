"""
Toy world generator for synthetic reasoning tasks.

Generates narratives from a simulated world state and questions
that test whether the model tracks entity locations, inventory,
and actions over time.

Task types:
  1. location_tracking - where is X after a series of moves?
  2. inventory_tracking - what does X have after interactions?
  3. exact_recall - remember a password/token seen earlier
  4. story_continuation - continue a coherent narrative
"""

import random
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


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
        names_subset = random.sample(self.names, k=random.randint(2, 4))
        for i, name in enumerate(names_subset):
            loc = random.choice(self.locations)
            # Guarantee at least the first entity has some items
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

    def generate_password(self) -> Tuple[str, str]:
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        password = "".join(random.choices(chars, k=8))
        entity = random.choice(list(self.entities.keys()))
        sentence = f"The secret code for {entity} is {password}."
        self._narrate(sentence)
        return password, entity


def generate_location_task(
    n_moves: int = 5,
    n_distractors: int = 3,
) -> Tuple[str, str, str]:
    """
    Generate a location tracking task.

    Returns: (narrative, question, answer)
    """
    world = World()
    sentences = []

    # Initial state
    for name, entity in world.entities.items():
        sentences.append(f"{name} was in the {entity.location}.")

    # Random moves
    names = list(world.entities.keys())
    for _ in range(n_moves):
        name = random.choice(names)
        action = random.choice(["move", "pickup", "drop"])
        if action == "move":
            sentences.append(world.move_entity(name))
        elif action == "pickup":
            s = world.pickup_item(name)
            if s:
                sentences.append(s)
        elif action == "drop":
            s = world.drop_item(name)
            if s:
                sentences.append(s)

    # Add distractor sentences
    for _ in range(n_distractors):
        distractors = [
            f"It was a quiet day.",
            f"The sun was shining.",
            f"Time passed slowly.",
            f"Nothing unusual happened.",
            f"The weather was pleasant.",
        ]
        sentences.insert(random.randint(0, len(sentences)), random.choice(distractors))

    narrative = " ".join(s for s in sentences if s)

    # Question
    target = random.choice(names)
    question = f"Where is {target}?"
    answer = world.entities[target].location

    return narrative, question, answer


def generate_inventory_task(
    n_actions: int = 4,
) -> Tuple[str, str, str]:
    """
    Generate an inventory tracking task.

    Returns: (narrative, question, answer)
    """
    world = World()
    sentences = []

    for name, entity in world.entities.items():
        if entity.inventory:
            items_str = " and ".join(entity.inventory)
            sentences.append(f"{name} was in the {entity.location} with {items_str}.")
        else:
            sentences.append(f"{name} was in the {entity.location}.")

    names = list(world.entities.keys())
    for _ in range(n_actions):
        name = random.choice(names)
        action = random.choice(["move", "pickup", "drop"])
        if action == "move":
            sentences.append(world.move_entity(name))
        elif action == "pickup":
            s = world.pickup_item(name)
            if s:
                sentences.append(s)
        elif action == "drop":
            s = world.drop_item(name)
            if s:
                sentences.append(s)

    narrative = " ".join(s for s in sentences if s)

    # Question about inventory
    entities_with_items = [n for n in names if world.entities[n].inventory]
    target = random.choice(entities_with_items) if entities_with_items else random.choice(names)
    question = f"What does {target} have?"
    answer = " and ".join(world.entities[target].inventory) if world.entities[target].inventory else "nothing"

    return narrative, question, answer


def generate_recall_task(
    n_distractor_sentences: int = 20,
) -> Tuple[str, str, str]:
    """
    Generate an exact recall task.
    Model must remember a password/code seen earlier.

    Returns: (narrative, question, answer)
    """
    world = World()
    sentences = []

    # Initial state
    for name, entity in world.entities.items():
        sentences.append(f"{name} was in the {entity.location}.")

    # Insert password
    password, entity_name = world.generate_password()

    # Distractor sentences
    distractors = [
        f"It was a quiet day.",
        f"The sun was shining brightly.",
        f"Time passed slowly in the house.",
        f"Nothing unusual happened.",
        f"The weather was pleasant outside.",
        f"Birds were singing in the garden.",
        f"A clock ticked on the wall.",
        f"The air felt warm and still.",
    ]

    for _ in range(n_distractor_sentences):
        sentences.append(random.choice(distractors))

    narrative = " ".join(s for s in sentences if s)

    question = f"What is the secret code for {entity_name}?"
    answer = password

    return narrative, question, answer


def generate_story_prompt(
    n_setup_sentences: int = 3,
) -> Tuple[str, str]:
    """
    Generate a story continuation task.

    Returns: (prompt, ground_truth_continuation)
    """
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

    return prompt, ground_truth


def generate_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    task_weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Generate a mixed dataset of reasoning tasks.

    Args:
        n_samples: number of samples to generate
        seed: random seed for reproducibility
        task_weights: dict of task type -> weight. Default equal weights.

    Returns:
        List of dicts with keys: narrative, question, answer, task_type
    """
    random.seed(seed)

    if task_weights is None:
        task_weights = {
            "location": 0.3,
            "inventory": 0.25,
            "recall": 0.25,
            "story": 0.2,
        }

    tasks = list(task_weights.keys())
    weights = list(task_weights.values())

    dataset = []
    for _ in range(n_samples):
        task_type = random.choices(tasks, weights=weights, k=1)[0]

        if task_type == "location":
            narrative, question, answer = generate_location_task()
        elif task_type == "inventory":
            narrative, question, answer = generate_inventory_task()
        elif task_type == "recall":
            narrative, question, answer = generate_recall_task()
        elif task_type == "story":
            narrative, answer = generate_story_prompt()
            question = ""
        else:
            continue

        dataset.append({
            "narrative": narrative,
            "question": question,
            "answer": answer,
            "task_type": task_type,
        })

    return dataset


if __name__ == "__main__":
    # Quick test
    import pprint

    dataset = generate_dataset(n_samples=5, seed=42)
    for i, sample in enumerate(dataset):
        print(f"\n=== Sample {i+1} [{sample['task_type']}] ===")
        print(f"Narrative: {sample['narrative'][:200]}...")
        if sample["question"]:
            print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
