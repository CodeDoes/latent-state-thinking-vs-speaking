"""Synthdata generator for the logic needle-in-a-haystack task.

Format:
    instruction + noise + needle + noise + needle-action-transformation
    + noise + repeat-X-times + ask-questions-about-needle-transformations

The generator is parametrised so you can dial difficulty without changing
the training loop. Overfitting is solved by infinite procedural variation.
"""

import random
from typing import Optional

# ── Noise sentence templates ──────────────────────────────────────────────
NOISE_SUBJECTS = [
    "the sky", "cats", "dogs", "the sun", "the moon", "trees",
    "the ocean", "birds", "the wind", "mountains", "rivers", "stars",
    "the earth", "fire", "the rain", "clouds", "fish", "flowers",
]
NOISE_VERBS = [
    "is", "are", "can be", "might be", "could be", "will be",
    "appear", "seem", "remain", "become", "stay",
]
NOISE_ADJECTIVES = [
    "blue", "warm", "cold", "bright", "dark", "soft", "hard",
    "wet", "dry", "fast", "slow", "big", "small", "round",
    "tall", "short", "sweet", "sour", "loud", "quiet",
]
NOISE_COMPLEMENTS = [
    "during the night", "in the morning", "after the rain",
    "before sunset", "on a clear day", "in the distance",
    "under the surface", "above the clouds", "near the shore",
    "through the valley", "across the field", "beyond the hills",
    "", "", "",  # sometimes omit
]


def random_noise_sentence(rng: random.Random) -> str:
    subj = rng.choice(NOISE_SUBJECTS)
    verb = rng.choice(NOISE_VERBS)
    adj = rng.choice(NOISE_ADJECTIVES)
    comp = rng.choice(NOISE_COMPLEMENTS)
    if comp:
        return f"{subj} {verb} {adj} {comp}."
    return f"{subj} {verb} {adj}."


# ── Operations ────────────────────────────────────────────────────────────

def apply_op(op: str, var_name: str, current_val: int) -> str:
    """Apply an operation and return the display text + new value."""
    if op == "add":
        delta = random.randint(1, 20)  # note: uses global random in templates
        return f"Add {delta} to {var_name}", current_val + delta
    elif op == "subtract":
        delta = random.randint(1, 20)
        return f"Subtract {delta} from {var_name}", current_val - delta
    elif op == "multiply":
        factor = random.randint(2, 5)
        return f"Multiply {var_name} by {factor}", current_val * factor
    elif op == "divide":
        factor = random.randint(2, 5)
        # keep it integer-clean
        new_val = current_val // factor
        return f"Divide {var_name} by {factor}", new_val
    elif op == "add_twice":
        d1, d2 = random.randint(1, 10), random.randint(1, 10)
        return f"Add {d1} to {var_name} then add {d2}", current_val + d1 + d2
    elif op == "set":
        new_val = random.randint(0, 100)
        return f"Set {var_name} to {new_val}", new_val
    return f"Do nothing to {var_name}", current_val


OPERATIONS = ["add", "subtract", "multiply", "divide", "set"]


# ── Generator ─────────────────────────────────────────────────────────────

class LogicNiiahGenerator:
    """Yields (text, answer_positions) pairs for the logic niiah task.

    `answer_positions` is a list of (start_char, end_char) spans in `text`
    that contain the answer tokens. The solver uses these to mask the loss.
    """

    def __init__(
        self,
        var_names: Optional[list[str]] = None,
        operations: Optional[list[str]] = None,
        *,
        seed: int = 42,
    ):
        self.var_names = var_names or ["A", "B", "C", "D", "E", "X", "Y", "Z"]
        self.operations = operations or OPERATIONS
        self.rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def generate(
        self,
        num_vars: int = 3,
        min_transforms: int = 2,
        max_transforms: int = 6,
        noise_min: int = 1,
        noise_max: int = 4,
        value_range: tuple[int, int] = (0, 100),
    ) -> dict:
        """Generate one logic niiah example.

        Returns a dict with:
            text: str — the full formatted text
            answer_spans: list[(start, end)] — character spans in `text`
                that contain the answers
            answers: list[str] — the correct answer strings
            metadata: dict — params used + ground-truth variable map
        """
        rng = self.rng
        chosen_names = rng.sample(self.var_names, min(num_vars, len(self.var_names)))
        values = {name: rng.randint(*value_range) for name in chosen_names}
        history = {name: [(values[name], "initial")] for name in chosen_names}

        lines = []
        # ── Instruction ──
        lines.append("Task: Track the variable values through the text and "
                     "answer the questions at the end.")
        lines.append("")

        # ── Noise block ──
        for _ in range(rng.randint(noise_min, noise_max)):
            lines.append(random_noise_sentence(rng))
        lines.append("")

        # ── Needles + transformations ──
        # Shuffle the variable order and interleave
        order = chosen_names.copy()
        rng.shuffle(order)
        for name in order:
            n_transforms = rng.randint(min_transforms, max_transforms)
            # Initial assignment (needle)
            lines.append(f"Let {name} = {values[name]}")
            for _ in range(rng.randint(noise_min, noise_max)):
                lines.append(random_noise_sentence(rng))
            # Transformations (needle-action-transformation)
            for _ in range(n_transforms):
                op = rng.choice(self.operations)
                display, new_val = apply_op(op, name, values[name])
                values[name] = new_val
                history[name].append((new_val, op))
                lines.append(display)
                for _ in range(rng.randint(noise_min, noise_max)):
                    lines.append(random_noise_sentence(rng))
            lines.append("")

        # ── Questions ──
        lines.append("Questions:")
        answer_lines_indices = []
        for name in chosen_names:
            lines.append(f"  What is the final value of {name}?")
            answer_line = f"  Answer: {values[name]}"
            answer_lines_indices.append(len(lines))
            lines.append(answer_line)

        text = "\n".join(lines)

        # Compute answer character spans
        answer_spans = []
        answers = []
        char_pos = 0
        for li, line in enumerate(lines):
            if li in answer_lines_indices:
                # line is like "  Answer: 42"
                answer_str = line.split(": ", 1)[1]
                start = char_pos + len(line) - len(answer_str)
                end = char_pos + len(line)
                answer_spans.append((start, end))
                answers.append(answer_str)
            char_pos += len(line) + 1  # +1 for the newline

        return {
            "text": text,
            "answer_spans": answer_spans,
            "answers": answers,
            "metadata": {
                "num_vars": num_vars,
                "num_transforms": [len(history[n]) - 1 for n in chosen_names],
                "final_values": {n: values[n] for n in chosen_names},
                "history": history,
            },
        }

    def generate_batch(
        self,
        batch_size: int = 8,
        **gen_kwargs,
    ) -> list[dict]:
        """Generate a batch of examples, each as a dict from generate()."""
        return [self.generate(**gen_kwargs) for _ in range(batch_size)]


# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = LogicNiiahGenerator()
    ex = gen.generate(num_vars=2, min_transforms=2, max_transforms=4)
    print(ex["text"])
    print(f"\nAnswer spans (char offsets): {ex['answer_spans']}")
    print(f"Answers: {ex['answers']}")
    print(f"Final values: {ex['metadata']['final_values']}")
