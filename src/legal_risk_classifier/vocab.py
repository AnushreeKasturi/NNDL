"""Word vocabulary for the CNN baseline.

The transformer models bring their own subword tokenizers. The CNN has no
pretrained vocabulary, so it needs one built from the training split only —
building it over the whole corpus would leak test vocabulary into training.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"
PAD_INDEX, UNK_INDEX = 0, 1

# Keep alphanumeric words; legal text is full of section numbers and
# cross-references, and "10.2" carries more signal than the punctuation around it.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-/]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Vocabulary:
    def __init__(self, tokens: list[str]) -> None:
        self.itos: list[str] = [PAD, UNK] + [t for t in tokens if t not in (PAD, UNK)]
        self.stoi: dict[str, int] = {token: i for i, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_tokens: int) -> list[int]:
        """Token ids, truncated and right-padded to a fixed width."""
        ids = [self.stoi.get(token, UNK_INDEX) for token in tokenize(text)[:max_tokens]]
        return ids + [PAD_INDEX] * (max_tokens - len(ids))

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2, max_size: int = 50_000) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        kept = [token for token, n in counts.most_common(max_size) if n >= min_freq]
        return cls(kept)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
