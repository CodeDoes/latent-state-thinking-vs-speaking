"""Multilingual TinyStories data loader.

Downloads and serves stories from Dxniz/TinyStories-Multilingual (29 languages).
Provides both world-tokenized and byte-tokenized batches.
"""

import json
import random
import torch
from pathlib import Path
from typing import Optional

from domains.rwkv.hf_rwkv_tokenizer import RWKV_TOKENIZER
from domains.byte.byte_vocab import BYTE_TO_ID, PAD_ID as BYTE_PAD, VOCAB_SIZE as BYTE_VOCAB_SIZE

WORLD_VOCAB_PATH = Path(__file__).parent / "rwkv_vocab_v20230424.txt"
world_tokenizer = RWKV_TOKENIZER(str(WORLD_VOCAB_PATH))
WORLD_PAD_ID = 0


# ── Languages available ──
ALL_LANGUAGES = [
    "en", "tr", "fr", "de", "es", "pt", "it", "ru", "zh", "ja", "ko",
    "ar", "hi", "nl", "pl", "sv", "uk", "cs", "ro", "hu", "el", "vi",
    "id", "fa", "da", "no", "sk", "sr", "bg",
]


def download_stories(cache_dir: Optional[Path] = None, max_stories: int = 50000):
    """Download multilingual TinyStories from Hugging Face, return list of dicts.

    Each dict: {'language_code': str, 'output': str, 'score': float}
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "multilingual_tinystories"
    cache_path = cache_dir / "stories.jsonl"

    if cache_path.exists():
        # Load from cache
        stories = []
        with open(cache_path, "r") as f:
            for line in f:
                stories.append(json.loads(line))
                if len(stories) >= max_stories:
                    break
        print(f"Loaded {len(stories)} stories from cache ({cache_path})")
        return stories

    # Download
    import requests
    url = "https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual/resolve/main/tinystories.jsonl"
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8'

    cache_dir.mkdir(parents=True, exist_ok=True)
    stories = []
    with open(cache_path, "w", encoding="utf-8") as f_out:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                f_out.write(line + "\n")
                stories.append(data)
                if len(stories) >= max_stories:
                    break
    print(f"Downloaded {len(stories)} stories to {cache_path}")
    return stories


def filter_by_language(stories, languages: list[str]):
    """Filter stories to only include specified languages."""
    return [s for s in stories if s.get("language_code") in languages]


def filter_by_score(stories, min_score: float = 7.0):
    """Filter stories by quality score (0-10)."""
    return [s for s in stories if s.get("score", 0) >= min_score]


class TinyStoriesDataset:
    """Iterable dataset over multilingual TinyStories.

    Provides both world-tokenized and byte-tokenized views of the same text.
    """

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        min_score: float = 7.0,
        max_stories: int = 50000,
        cache_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        stories = download_stories(cache_dir, max_stories)
        if languages:
            stories = filter_by_language(stories, languages)
        if min_score > 0:
            stories = filter_by_score(stories, min_score)
        self.stories = stories
        self.rng = random.Random(seed)
        print(f"TinyStoriesDataset: {len(stories)} stories, "
              f"{len(set(s['language_code'] for s in stories))} languages")

    def __len__(self):
        return len(self.stories)

    def get_world_batch(
        self, batch_size: int, max_len: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-tokenized batch: (input_ids, targets, mask)."""
        input_ids_list, target_list, mask_list = [], [], []
        for _ in range(batch_size):
            text = self.rng.choice(self.stories)["output"]
            tokens = world_tokenizer.encodeBytes(text.encode("utf-8"))
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [WORLD_PAD_ID] * (max_len - len(tokens))
            ids = torch.tensor(tokens, dtype=torch.long)
            tgt = torch.roll(ids, shifts=-1)
            tgt[-1] = WORLD_PAD_ID
            msk = torch.zeros(max_len, dtype=torch.float)
            label_pos = min(len(text.encode("utf-8")) - 1, max_len - 2)
            msk[label_pos] = 1.0
            input_ids_list.append(ids)
            target_list.append(tgt)
            mask_list.append(msk)
        return torch.stack(input_ids_list), torch.stack(target_list), torch.stack(mask_list)

    def get_byte_batch(
        self, batch_size: int, max_len: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Byte-tokenized batch: (input_ids, targets, mask)."""
        input_ids_list, target_list, mask_list = [], [], []
        for _ in range(batch_size):
            text = self.rng.choice(self.stories)["output"]
            raw = text.encode("utf-8")
            tokens = [BYTE_TO_ID[b] for b in raw]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [BYTE_PAD] * (max_len - len(tokens))
            ids = torch.tensor(tokens, dtype=torch.long)
            tgt = torch.roll(ids, shifts=-1)
            tgt[-1] = BYTE_PAD
            msk = torch.zeros(max_len, dtype=torch.float)
            label_pos = min(len(raw) - 1, max_len - 2)
            msk[label_pos] = 1.0
            input_ids_list.append(ids)
            target_list.append(tgt)
            mask_list.append(msk)
        return torch.stack(input_ids_list), torch.stack(target_list), torch.stack(mask_list)


# ── Quick test ──
if __name__ == "__main__":
    ds = TinyStoriesDataset(languages=["en", "fr", "de", "ja"], max_stories=1000)
    print(f"Dataset: {len(ds)} stories")

    # Test world batch
    ids, tgt, msk = ds.get_world_batch(2, 64)
    print(f"World batch: ids={ids.shape}, targets={tgt.shape}, mask={msk.shape}")

    # Test byte batch
    ids, tgt, msk = ds.get_byte_batch(2, 128)
    print(f"Byte batch: ids={ids.shape}, targets={tgt.shape}, mask={msk.shape}")

    # Show example
    story = ds.stories[0]
    print(f"\nExample: [{story['language_code']}] score={story['score']}")
    print(f"  {story['output'][:100]}...")
