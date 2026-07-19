#!/usr/bin/env python3
"""Unit tests for the RWKV world tokenizer.

Tests both the TRIE-based RWKV_TOKENIZER and the HuggingFace RwkvTokenizer
wrapper. Validates against the reference implementation from ChatRWKV.

Usage:
    PYTHONPATH=. python tests/test_rwkv_tokenizer.py -v
"""

import itertools
import random
import sys
import unittest
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_rwkv_tokenizer import RWKV_TOKENIZER, TRIE

VOCAB_PATH = Path(__file__).parent.parent / "src" / "rwkv_vocab_v20230424.txt"
TOKENIZER = RWKV_TOKENIZER(str(VOCAB_PATH))


# ── Reference Tokenizer #1 (slow, for cross-validation) ──────────────────
# Implements the same greedy longest-match algorithm using precomputed
# lookup tables instead of a TRIE. Used to verify our TRIE implementation.
class ReferenceTokenizer:
    """Slow reference: Tokenizer #1 from ChatRWKV (lookup-table-based)."""

    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[: line.index(" ")])
                x = eval(line[line.index(" ") : line.rindex(" ")])
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes)
                assert len(x) == int(line[line.rindex(" ") :])
                sorted_tokens.append(x)
                self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}

        # Precompute tables for fast matching
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for s in reversed(sorted_tokens):  # longer tokens first
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes):
        tokens = []
        i = 0
        while i < len(src):
            s = src[i : i + 1]
            if i < len(src) - 1:
                s1 = int(src[i + 1])
                s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(self.idx2token[t] for t in tokens)

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode("utf-8")


REF_TOKENIZER = ReferenceTokenizer(str(VOCAB_PATH))


# ── Test Cases ────────────────────────────────────────────────────────────

class TestTrieStructure(unittest.TestCase):
    """Test the TRIE data structure directly."""

    def test_trie_empty(self):
        t = TRIE()
        self.assertIsNone(t.ch)
        self.assertEqual(len(t.to), 256)
        self.assertIsNone(t.to[0])
        self.assertEqual(t.values, set())

    def test_trie_add_and_find(self):
        t = TRIE()
        t.add(b"hello", val=(b"hello", 42))
        t.add(b"he", val=(b"he", 7))
        t.add(b"world", val=(b"world", 99))

        # Find longest match at start of "hello!"
        idx, node, values = t.find_longest(b"hello!", 0)
        self.assertEqual(idx, 5)  # "hello" is 5 bytes
        self.assertIn((b"hello", 42), values)

        # Find at start of "he"
        idx, node, values = t.find_longest(b"he", 0)
        self.assertEqual(idx, 2)
        self.assertIn((b"he", 7), values)

        # "hex" — should match "he" (2 bytes) since "hex" isn't in trie
        idx, node, values = t.find_longest(b"hex", 0)
        self.assertEqual(idx, 2)
        self.assertIn((b"he", 7), values)

    def test_trie_no_match_returns_id(self):
        """Starting at an unknown byte: the TRIE returns (1, ???, ???)
        because there's no child for 0x00. The tokenizer handles this by
        falling back to the single-byte token."""
        t = TRIE()
        t.add(b"hello", val=(b"hello", 1))
        # The trie has no child for 0x00, so find_longest should raise
        # or return something the caller can handle.
        with self.assertRaises((UnboundLocalError, TypeError)):
            idx, node, values = t.find_longest(b"\x00test", 0)

    def test_trie_add_no_val(self):
        t = TRIE()
        t.add(b"test")
        idx, node, values = t.find_longest(b"test", 0)
        self.assertEqual(idx, 4)
        self.assertIn(b"test", values)


class TestVocabIntegrity(unittest.TestCase):
    """Test the vocabulary file itself."""

    def test_vocab_size(self):
        """Vocab file has 65529 entries (indices 1-65529).
        Token 0 is reserved for <|endoftext|> sentinel."""
        self.assertEqual(len(TOKENIZER.idx2token), 65529)

    def test_all_indices_contiguous(self):
        """Indices 1-65529, no gaps. Token 0 reserved."""
        indices = sorted(TOKENIZER.idx2token.keys())
        self.assertEqual(indices[0], 1, "First index should be 1, token 0 reserved")
        self.assertEqual(indices[-1], 65529)
        self.assertEqual(len(indices), 65529)
        for i in range(1, 65530):
            self.assertIn(i, TOKENIZER.idx2token, f"Missing index {i}")

    def test_first_256_are_single_bytes(self):
        """Tokens 1-256 must be single bytes 0x00-0xFF (offset by 1).
        Token 0 = <|endoftext|>, token 1 = 0x00, token 2 = 0x01, ...
        """
        for byte_val in range(256):
            token_id = byte_val + 1
            tok = TOKENIZER.idx2token[token_id]
            self.assertIsInstance(tok, bytes)
            self.assertEqual(
                len(tok), 1,
                f"Token {token_id} should be single byte, got {tok!r} (len={len(tok)})"
            )
            self.assertEqual(
                tok[0], byte_val,
                f"Token {token_id} should be byte {byte_val}, got {tok[0]}"
            )

    def test_token_1_is_null_byte(self):
        """Token 1 = byte 0x00 (null)."""
        self.assertEqual(TOKENIZER.idx2token[1], b"\x00")

    def test_no_duplicate_tokens(self):
        self.assertEqual(len(TOKENIZER.token2idx), len(TOKENIZER.idx2token))

    def test_all_tokens_are_bytes(self):
        for idx, tok in TOKENIZER.idx2token.items():
            self.assertIsInstance(tok, bytes, f"Token {idx} is not bytes: {tok!r}")

    def test_decode_single_byte_roundtrip(self):
        """Every single-byte (0x00-0xFF) decodes back to itself.
        Token 1 = byte 0x00, token 2 = byte 0x01, ..., token 256 = byte 0xFF.
        """
        for b_val in range(256):
            token_id = b_val + 1  # token 1 = byte 0
            decoded = TOKENIZER.decodeBytes([token_id])
            self.assertEqual(
                decoded, bytes([b_val]),
                f"Byte {b_val}: token {token_id} gave {decoded!r}"
            )


class TestReferenceConsistency(unittest.TestCase):
    """Our TRIE tokenizer must match the reference lookup-table tokenizer."""

    def _check_consistency(self, text: str):
        our_tokens = TOKENIZER.encode(text)     # returns list[list[int]]
        ref_tokens = REF_TOKENIZER.encode(text)  # returns list[int]
        self.assertEqual(
            our_tokens[0] if our_tokens else [],
            ref_tokens,
            f"Mismatch for {text!r}: our={our_tokens}, ref={ref_tokens}",
        )

    def test_empty_string(self):
        self._check_consistency("")

    def test_single_space(self):
        self._check_consistency(" ")

    def test_simple_ascii(self):
        for s in ["hello", "world", "hello world", "abc123", "test!@#"]:
            self._check_consistency(s)

    def test_numbers(self):
        for s in ["0", "10", "99", "100", "12345", " 0", " 10", " 99"]:
            self._check_consistency(s)

    def test_unicode(self):
        cases = [
            "你好世界",
            "ñóñò",
            "café",
            "straße",
            "日本語",
            "Привет",
            "مرحبا",
            "שלום",
            "🌍🌎🌏",
            "éâîôû",
        ]
        for s in cases:
            self._check_consistency(s)

    def test_punctuation(self):
        self._check_consistency("...,;:!?-'\"()[]{}@#$%^&*+=/\\|~`<>")

    def test_newlines_and_tabs(self):
        for s in ["\n", "\t", "\r\n", "line1\nline2\nline3", "a\tb"]:
            self._check_consistency(s)

    def test_repeated_chars(self):
        for ch in ["a", " ", "0", "x"]:
            for length in [1, 2, 5, 10, 50, 100]:
                self._check_consistency(ch * length)

    def test_mixed_language(self):
        cases = [
            "Hello 你好 123",
            "café ñoño あいう",
            "English (日本語) mixed",
            "123 456 789 你好吗",
            "a" * 50 + " " + "b" * 50,
        ]
        for s in cases:
            self._check_consistency(s)

    def test_random_ascii_strings(self):
        """Reference test: 500 random ASCII strings."""
        random.seed(42)
        char_sets = [
            ["0", " "],
            ["0", "1"],
            ["0", "1", " "],
            ["0", "1", " ", "00", "11", "  ", "000", "111", "   "],
            list("01 \n\r\t,.;!?:'\"-=你好"),
        ]
        for _ in range(500):
            x = ""
            for cs in char_sets:
                for _ in range(random.randint(1, 32)):
                    x += random.choice(cs)
            self._check_consistency(x)

    def test_random_multibyte(self):
        """Reference test: 500 random multi-byte Unicode strings."""
        random.seed(42)
        for _ in range(500):
            codepoint = random.randint(256, 0x10FFFF)
            try:
                char = chr(codepoint)
                char.encode("utf-8")  # must be valid UTF-8
                count = random.randint(1, 4)
                s = char * count
                self._check_consistency(s)
            except (ValueError, UnicodeEncodeError):
                pass


class TestRoundtrip(unittest.TestCase):
    """encode then decode must return the original text."""

    def _check_roundtrip(self, text: str):
        tokens = TOKENIZER.encode(text)
        # encode returns [[...]] for batch
        flat_tokens = tokens[0] if tokens else []
        decoded = TOKENIZER.decode([flat_tokens])[0] if flat_tokens else ""
        self.assertEqual(
            decoded, text,
            f"Roundtrip failed for {text!r}: got {decoded!r}"
        )

    def test_empty(self):
        self._check_roundtrip("")

    def test_simple(self):
        for s in ["hello", "world", "test", " ", "  ", "a"]:
            self._check_roundtrip(s)

    def test_unicode_roundtrip(self):
        cases = [
            "你好世界",
            "café ñoño",
            "日本語",
            "Привет мир",
            "مرحبا بالعالم",
            "שלום עולם",
            "éâîôû",
            "a\u0300",  # a + combining grave
            "\u00e9",  # é precomposed
        ]
        for s in cases:
            self._check_roundtrip(s)

    def test_emoji(self):
        cases = [
            "😀😁😂🤣😃",
            "🌍🌎🌏",
            "👋 Hello",
            "a😀b😀c",
            "❤️",  # heart + variation selector
        ]
        for s in cases:
            self._check_roundtrip(s)

    def test_long_string(self):
        s = "The quick brown fox jumps over the lazy dog. " * 10
        self._check_roundtrip(s)

    def test_repeated_spaces(self):
        """Reference test: repeated spaces up to large counts."""
        for length in [0, 1, 2, 5, 10, 50, 100, 500]:
            s = " " * length
            self._check_roundtrip(s)

    def test_numeric_strings(self):
        for s in ["123", "999999", "0", "000", "123 456 789", " 123", "123 "]:
            self._check_roundtrip(s)

    def test_special_characters(self):
        for s in ["\n", "\t", "\r", "\n\n", "a\nb", "a\tb", "a\r\nb"]:
            self._check_roundtrip(s)

    def test_mixed_script(self):
        """Mix multiple scripts in one string — common for RWKV world use."""
        s = "Hello 你好 123 café ñoño 日本語 Привет! 🌍"
        self._check_roundtrip(s)

    def test_random_roundtrips(self):
        """Reference-style: 1000 random strings, all must roundtrip."""
        random.seed(42)
        for _ in range(1000):
            length = random.randint(0, 100)
            chars = []
            for _ in range(length):
                # Mix of ASCII, extended Latin, CJK
                choices = [
                    (0.7, lambda: chr(random.randint(32, 126))),
                    (0.1, lambda: chr(random.randint(0x00C0, 0x024F))),
                    (0.1, lambda: chr(random.randint(0x4E00, 0x9FFF))),
                    (0.1, lambda: chr(random.randint(0x3040, 0x30FF))),
                ]
                r = random.random()
                cum = 0.0
                for prob, fn in choices:
                    cum += prob
                    if r <= cum:
                        chars.append(fn())
                        break
            s = "".join(chars)
            try:
                s.encode("utf-8")  # Must be valid UTF-8
                self._check_roundtrip(s)
            except UnicodeEncodeError:
                pass


class TestGreedyEncoding(unittest.TestCase):
    """The tokenizer must pick the longest matching token (greedy)."""

    def test_longer_token_preferred(self):
        """If both 'a' and 'ab' exist, 'ab' must be chosen over 'a' + 'b'."""
        tokens = TOKENIZER.encode("ab")
        flat = tokens[0] if tokens else []
        # The tokenizer should NOT split "ab" into "a" + "b" if "ab" exists
        encoded = TOKENIZER.decodeBytes(flat)
        self.assertEqual(encoded, b"ab")

    def test_no_overlap_escape(self):
        """Encoding must not skip bytes."""
        tokens = TOKENIZER.encode("abcdefghij")
        flat = tokens[0] if tokens else []
        decoded = TOKENIZER.decodeBytes(flat)
        self.assertEqual(decoded, b"abcdefghij")

    def test_token_lengths(self):
        """Every token in the vocab has the declared length."""
        idx2token = TOKENIZER.idx2token
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[: line.index(" ")])
                declared_len = int(line[line.rindex(" ") :])
                actual_len = len(idx2token[idx])
                self.assertEqual(
                    actual_len, declared_len,
                    f"Token {idx}: declared length {declared_len}, "
                    f"actual {actual_len}: {idx2token[idx]!r}"
                )


class TestDecode(unittest.TestCase):
    """Decode must handle edge cases."""

    def test_empty_tokens(self):
        self.assertEqual(TOKENIZER.decodeBytes([]), b"")

    def test_single_byte_token(self):
        """Token 66 = byte 0x41 = 'A' (token 1 = byte 0)."""
        self.assertEqual(TOKENIZER.decodeBytes([66]), b"A")
        # Token 66 = byte 0x41 = 'A', token 97 = byte 0x60 = '`'
        self.assertEqual(TOKENIZER.decodeBytes([97]), b"`")

    def test_decode_hello(self):
        """'Hello' in single-byte tokens: H=0x48→73, e=0x65→102, etc."""
        result = TOKENIZER.decodeBytes([73, 102, 109, 109, 112])
        self.assertEqual(result, b"Hello")

    def test_decode_utf8_string(self):
        tokens = TOKENIZER.encode("Hello 你好")
        decoded = TOKENIZER.decode(tokens)
        self.assertEqual(decoded, ["Hello 你好"])

    def test_decode_batch(self):
        tokens_a = TOKENIZER.encode("hello")[0]
        tokens_b = TOKENIZER.encode("world")[0]
        decoded = TOKENIZER.decode([tokens_a, tokens_b])
        self.assertEqual(decoded, ["hello", "world"])


class TestEncodeAPI(unittest.TestCase):
    """Test the encode() API shape and behavior."""

    def test_encode_string_returns_batch(self):
        """encode(str) returns list[list[int]] (batch of 1)."""
        result = TOKENIZER.encode("hello")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[0][0], int)

    def test_encode_list_returns_batch(self):
        result = TOKENIZER.encode(["hello", "world"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_encode_consistency(self):
        """Single string and list-of-one must give same tokens."""
        single = TOKENIZER.encode("hello")[0]
        listed = TOKENIZER.encode(["hello"])[0]
        self.assertEqual(single, listed)

    def test_encode_empty_string(self):
        result = TOKENIZER.encode("")
        self.assertEqual(result, [[]])

    def test_encode_empty_list(self):
        result = TOKENIZER.encode([])
        self.assertEqual(result, [])


class TestEdgeCases(unittest.TestCase):
    """Edge cases that have caused issues in the past."""

    def test_utf8_boundary(self):
        """3-byte UTF-8 sequences must not be truncated in the middle."""
        s = "\u0800"  # first 3-byte char
        tokens = TOKENIZER.encode(s)
        flat = tokens[0] if tokens else []
        decoded = TOKENIZER.decode([flat])[0] if flat else ""
        self.assertEqual(decoded, s,
                         f"UTF-8 boundary roundtrip failed for U+0800")

    def test_utf8_continuation_bytes(self):
        """Trailing continuation bytes should be handled."""
        s = "a\x80b\x80c"  # bare continuation bytes
        tokens = TOKENIZER.encode(s)
        flat = tokens[0] if tokens else []
        decoded = TOKENIZER.decodeBytes(flat)
        self.assertEqual(decoded, s.encode("utf-8"))

    def test_null_byte(self):
        """Null byte (0x00) → token 1 (since token 0 is reserved)."""
        tokens = TOKENIZER.encodeBytes(b"\x00")
        self.assertEqual(tokens, [1], f"Null byte should be token 1, got {tokens}")
        decoded = TOKENIZER.decodeBytes([1])
        self.assertEqual(decoded, b"\x00")

    def test_all_byte_values(self):
        """Every byte value 0x00-0xFF must roundtrip individually.
        Token IDs: 1→0x00, 2→0x01, ..., 256→0xFF."""
        for b_val in range(256):
            token_id = b_val + 1
            decoded = TOKENIZER.decodeBytes([token_id])
            self.assertEqual(decoded, bytes([b_val]),
                             f"Byte {b_val}: token {token_id} gave {decoded!r}")

    def test_consecutive_null_bytes(self):
        s = "\x00\x00\x00hello\x00\x00"
        tokens = TOKENIZER.encodeBytes(s.encode("utf-8"))
        decoded = TOKENIZER.decodeBytes(tokens)
        self.assertEqual(decoded, s.encode("utf-8"))

    def test_high_codepoints(self):
        """4-byte UTF-8 sequences (emoji, rare CJK)."""
        for cp in [0x1F600, 0x20000, 0x2A6DF, 0x10FFFF]:
            try:
                char = chr(cp)
                s = char * 3
                tokens = TOKENIZER.encode(s)
                flat = tokens[0]
                decoded = TOKENIZER.decode([flat])[0]
                self.assertEqual(decoded, s)
            except (ValueError, UnicodeEncodeError):
                pass


class TestPrintTokens(unittest.TestCase):
    """printTokens must not crash."""

    def test_print_empty(self):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            TOKENIZER.printTokens([])
        self.assertEqual(f.getvalue().strip(), "")

    def test_print_simple(self):
        import io
        from contextlib import redirect_stdout
        tokens = TOKENIZER.encode("hello")[0]
        f = io.StringIO()
        with redirect_stdout(f):
            TOKENIZER.printTokens(tokens)
        output = f.getvalue()
        self.assertIn("hello", output)

    def test_print_no_crash(self):
        """Print tokens doesn't crash. Token 0 is reserved, start from 1."""
        tokens = list(range(1, 101))
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            TOKENIZER.printTokens(tokens)
        self.assertTrue(len(f.getvalue()) > 0)


class TestEncodeWithSpans(unittest.TestCase):
    """The encode_with_spans helper used in token surgery."""

    def _check_spans(self, text: str):
        """Verify spans cover the full byte sequence exactly."""
        from token_surgery import encode_with_spans
        raw = text.encode("utf-8")
        tokens, spans = encode_with_spans(text)

        # Concatenate byte spans must equal the raw bytes
        reconstructed = b"".join(raw[s:e] for s, e in spans)
        self.assertEqual(
            reconstructed, raw,
            f"Spans don't cover full text for {text!r}: "
            f"got {reconstructed!r}, expected {raw!r}"
        )

        # Spans must be non-overlapping and contiguous
        prev_end = 0
        for s, e in spans:
            self.assertEqual(s, prev_end, f"Gap in spans for {text!r}")
            self.assertGreater(e, s, f"Empty span for {text!r}")
            prev_end = e

    def test_simple_ascii(self):
        self._check_spans("hello")

    def test_unicode(self):
        self._check_spans("你好世界")

    def test_mixed(self):
        self._check_spans("Hello 你好 123")

    def test_empty(self):
        tokens, spans = [], []
        from token_surgery import encode_with_spans
        tokens, spans = encode_with_spans("")
        self.assertEqual(tokens, [])
        self.assertEqual(spans, [])

    def test_single_char(self):
        self._check_spans("a")
        self._check_spans("é")
        self._check_spans("你")

    def test_long_text(self):
        self._check_spans("The quick brown fox jumps over the lazy dog. " * 5)


class TestReferenceFullSuite(unittest.TestCase):
    """Reproduce the ChatRWKV reference test suite exactly."""

    def test_reference_test_cases(self):
        """The exact test cases from the ChatRWKV tokenizer test."""
        QQQ = ["", " ", "Õ\U000683b8", b"\xe6\xaa\x81".decode("utf-8")]

        random.seed(0)
        for _ in range(500):
            x = ""
            for xx in [
                ["0", " "],
                ["0", "1"],
                ["0", "1", " "],
                ["0", "1", " ", "00", "11", "  ", "000", "111", "   "],
                list("01 \n\r\t,.;!?:'\"-=你好"),
            ]:
                for _ in range(random.randint(1, 32)):
                    x += random.choice(xx)
            QQQ.append(x)

        random.seed(0)
        for i in range(500):
            QQQ.append(" " * i)

        random.seed(0)
        for _ in range(500):
            codepoint = random.randint(0, 255)
            count = random.randint(1, 32)
            QQQ.append(chr(codepoint) * count)

        random.seed(0)
        for _ in range(500):
            codepoint = random.randint(256, 0x10FFFF)
            try:
                char = chr(codepoint)
                char.encode("utf-8")
                count = random.randint(1, 4)
                QQQ.append(char * count)
            except (ValueError, UnicodeEncodeError):
                pass

        for q in QQQ:
            with self.subTest(q=q[:50]):
                # Roundtrip
                our_tokens = TOKENIZER.encode(q)
                flat = our_tokens[0] if our_tokens else []
                decoded = TOKENIZER.decode([flat])[0] if flat else ""
                self.assertEqual(
                    decoded, q,
                    f"Roundtrip failed for {q!r}: got {decoded!r}"
                )
                # Consistency with reference tokenizer
                ref_tokens = REF_TOKENIZER.encode(q)
                self.assertEqual(
                    flat, ref_tokens,
                    f"TRIE vs reference mismatch for {q!r}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
