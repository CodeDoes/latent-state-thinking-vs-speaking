"""Synthdata generator with LEARNABLE RULES.

Each task has a fixed underlying rule that the model can discover.
The rule is consistent across examples, but inputs vary.
"""

import random
from typing import Optional, List, Dict


# ── Rule-based tasks ──────────────────────────────────────────────────────

class RuleGenerator:
    """Generates examples following a fixed rule."""

    def __init__(self, rule_name: str, seed: int = 42):
        self.rule_name = rule_name
        self.rng = random.Random(seed)

    def generate(self) -> Dict:
        raise NotImplementedError


class SumThresholdRule(RuleGenerator):
    """Sum of numbers in sequence >= threshold -> 1 else 0."""
    def __init__(self, threshold: int = 50, **kwargs):
        super().__init__("sum_threshold", **kwargs)
        self.threshold = threshold

    def generate(self) -> Dict:
        length = self.rng.randint(8, 20)
        seq = [self.rng.randint(1, 9) for _ in range(length)]
        total = sum(seq)
        label = 1 if total >= self.threshold else 0
        text = " ".join(str(d) for d in seq) + f" = {label}"
        return {
            'text': text,
            'answer_spans': [(len(text) - 1, len(text))],
            'answers': [str(label)],
        }


class VowelMajorityRule(RuleGenerator):
    """Vowels > consonants -> 1 else 0."""
    def __init__(self, **kwargs):
        super().__init__("vowel_majority", **kwargs)
        self.vowels = set('aeiouAEIOU')

    def generate(self) -> Dict:
        length = self.rng.randint(8, 20)
        seq = [self.rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length)]
        vowel_count = sum(1 for c in seq if c in self.vowels)
        label = 1 if vowel_count > length - vowel_count else 0
        text = " ".join(seq) + f" = {label}"
        return {
            'text': text,
            'answer_spans': [(len(text) - 1, len(text))],
            'answers': [str(label)],
        }


class EndpointMatchRule(RuleGenerator):
    """First char == last char -> 1 else 0."""
    def __init__(self, **kwargs):
        super().__init__("endpoint_match", **kwargs)

    def generate(self) -> Dict:
        length = self.rng.randint(8, 20)
        seq = [self.rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length)]
        label = 1 if seq[0] == seq[-1] else 0
        text = " ".join(seq) + f" = {label}"
        return {
            'text': text,
            'answer_spans': [(len(text) - 1, len(text))],
            'answers': [str(label)],
        }


class CountTriggerRule(RuleGenerator):
    """Count of target char > threshold -> 1 else 0."""
    def __init__(self, target: str = 'x', threshold: int = 3, **kwargs):
        super().__init__("count_trigger", **kwargs)
        self.target = target
        self.threshold = threshold

    def generate(self) -> Dict:
        length = self.rng.randint(8, 20)
        seq = [self.rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length)]
        count = sum(1 for c in seq if c == self.target)
        label = 1 if count > self.threshold else 0
        text = " ".join(seq) + f" = {label}"
        return {
            'text': text,
            'answer_spans': [(len(text) - 1, len(text))],
            'answers': [str(label)],
        }


class ParityRule(RuleGenerator):
    """Sum of digits % 2 == 0 -> 1 else 0."""
    def __init__(self, **kwargs):
        super().__init__("parity", **kwargs)

    def generate(self) -> Dict:
        length = self.rng.randint(8, 20)
        seq = [self.rng.randint(0, 9) for _ in range(length)]
        label = 1 if sum(seq) % 2 == 0 else 0
        text = " ".join(str(d) for d in seq) + f" = {label}"
        return {
            'text': text,
            'answer_spans': [(len(text) - 1, len(text))],
            'answers': [str(label)],
        }


class ModuloRule(RuleGenerator):
    """Sum of digits % 3 == 0 -> 1 else 0."""
    def __init__(self, mod: int = 3, **kwargs):
        super().__init__("modulo", **kwargs)
        self.mod = mod

    def generate(self) -> Dict:
        length = self.rng.randint(8, 20)
        seq = [self.rng.randint(0, 9) for _ in range(length)]
        label = 1 if sum(seq) % self.mod == 0 else 0
        text = " ".join(str(d) for d in seq) + f" = {label}"
        return {
            'text': text,
            'answer_spans': [(len(text) - 1, len(text))],
            'answers': [str(label)],
        }


# ── Fixed rule instances (same rule across all examples) ──────────────────

RULES = {
    'sum_threshold': SumThresholdRule(threshold=50),
    'vowel_majority': VowelMajorityRule(),
    'endpoint_match': EndpointMatchRule(),
    'count_trigger': CountTriggerRule(target='x', threshold=3),
    'parity': ParityRule(),
    'modulo3': ModuloRule(mod=3),
}


def generate_batch(rule_name: str, batch_size: int, seed: int = 42) -> List[Dict]:
    """Generate a batch using a fixed rule with varied RNG."""
    if rule_name not in RULES:
        raise ValueError(f"Unknown rule: {rule_name}")
    rng = random.Random(seed)
    rule = RULES[rule_name]
    rule.rng = rng  # override with our RNG
    return [rule.generate() for _ in range(batch_size)]


# ── Tokenizer for character/space sequences ───────────────────────────────

CHARS = ['\n', ' ', '!', ',', '-', '.', ':', '=', '=',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         '=']
SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
VOCAB = SPECIAL + CHARS
char_to_id = {c: i for i, c in enumerate(VOCAB)}
id_to_char = {i: c for c, i in char_to_id.items()}
PAD_ID = char_to_id['<PAD>']
UNK_ID = char_to_id['<UNK>']


def encode(text: str) -> List[int]:
    return [char_to_id.get(c, UNK_ID) for c in text]


def decode(ids: List[int]) -> str:
    return ''.join(id_to_char.get(i, '<UNK>') for i in ids)


if __name__ == '__main__':
    # Quick test
    for name in RULES:
        batch = generate_batch(name, 5, seed=42)
        print(f"{name}:")
        for ex in batch[:2]:
            print(f"  {ex['text'][:80]}... -> {ex['answers']}")
        print()