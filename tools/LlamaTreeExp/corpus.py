# -*- coding: utf-8 -*-
"""Shared encrypted corpus writer (AES-256-GCM, XOR fallback).

Used by the online engine (accept/backspace events) and the offline recorder
(commit segments). No torch dependency so the recorder stays lightweight.
"""
import base64
import hashlib
import json
import os
import sys

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover
    AESGCM = None


class CorpusWriter:
    @staticmethod
    def default_key_file(path):
        return os.path.join(os.path.dirname(os.path.abspath(path)), "corpus.key")

    def __init__(self, path, key=None, key_file=None):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        seed = key or os.environ.get("WEASEL_CORPUS_KEY")
        if not seed and key_file and os.path.exists(key_file):
            with open(key_file, "rb") as f:
                seed = f.read()
        # [CORPUS-002] Pin the key to a file next to the corpus.
        # Deriving it from the environment alone meant that a different
        # USERNAME, or a process launched from a different context, silently
        # produced a second key - and everything written under the first one
        # became unreadable. The oldest 14 records of diag/corpus.jsonl died
        # exactly that way (AES-GCM InvalidTag, base64 and length intact).
        # The legacy derivation is kept for the first run so already-written
        # records stay readable, then frozen for good.
        if not seed:
            pinned = key_file or self.default_key_file(path)
            if os.path.exists(pinned):
                try:
                    with open(pinned, "rb") as f:
                        seed = f.read().strip()
                except OSError:
                    seed = None
            if not seed:
                seed = ("weasel-corpus-" +
                        os.environ.get("USERNAME", "local")).encode("utf-8")
                try:
                    with open(pinned, "wb") as f:
                        f.write(seed)
                except OSError:
                    pass
        if isinstance(seed, str):
            seed = seed.encode("utf-8")
        self.key = hashlib.sha256(seed).digest()
        self.aead = AESGCM(self.key) if AESGCM else None

    def _xor(self, data):
        k = self.key
        return bytes(b ^ k[i % len(k)] for i, b in enumerate(data))

    def write(self, record):
        raw = json.dumps(record, ensure_ascii=False).encode("utf-8")
        if self.aead:
            nonce = os.urandom(12)
            blob = base64.b64encode(nonce + self.aead.encrypt(nonce, raw, None))
        else:
            blob = base64.b64encode(self._xor(raw))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(blob.decode("ascii") + "\n")

    def read_all(self):
        out = []
        if not os.path.exists(self.path):
            return out
        bad = 0
        total = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    data = base64.b64decode(line)
                    if self.aead:
                        raw = self.aead.decrypt(data[:12], data[12:], None)
                    else:
                        raw = self._xor(data)
                    out.append(json.loads(raw.decode("utf-8")))
                except Exception:
                    bad += 1
                    continue
        if bad:
            # [CORPUS-003] Silence here is a trap: before the key was pinned to
            # a file it could change with the environment, and then every record
            # of a healthy-looking file fails to decrypt and the caller just
            # sees an empty corpus with no explanation.
            print("[corpus] %d/%d 条记录无法解密，已跳过：%s"
                  % (bad, total, self.path), file=sys.stderr, flush=True)
        return out
