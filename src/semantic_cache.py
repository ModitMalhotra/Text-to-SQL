import hashlib
import json
import re
import time

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "once": "1",
    "twice": "2",
}

DIRECTION = {
    "highest": "max",
    "most": "max",
    "top": "max",
    "largest": "max",
    "biggest": "max",
    "best": "max",
    "maximum": "max",
    "max": "max",
    "longest": "max",
    "slowest": "max",
    "greatest": "max",
    "lowest": "min",
    "least": "min",
    "fewest": "min",
    "smallest": "min",
    "minimum": "min",
    "min": "min",
    "worst": "min",
    "shortest": "min",
    "fastest": "min",
    "bottom": "min",
}

AGGREGATE = {
    "average": "avg",
    "avg": "avg",
    "mean": "avg",
    "total": "sum",
    "sum": "sum",
    "median": "median",
}

COMPARATOR = {
    "more": "gt",
    "longer": "gt",
    "over": "gt",
    "above": "gt",
    "greater": "gt",
    "less": "lt",
    "fewer": "lt",
    "under": "lt",
    "below": "lt",
    "shorter": "lt",
}

VALUE = {
    "delivered": "delivered",
    "canceled": "canceled",
    "cancelled": "canceled",
    "shipped": "shipped",
    "invoiced": "invoiced",
    "processing": "processing",
    "unavailable": "unavailable",
    "created": "created",
    "boleto": "boleto",
    "voucher": "voucher",
    "credit": "credit_card",
    "debit": "debit_card",
}

TOKEN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")


class Embedder:
    def __init__(self, name=MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name)

    def encode(self, texts):
        vecs = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    def __init__(self, dim=256):
        self.name = f"hash-{dim}"
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            grams = words + [a + "_" + b for a, b in zip(words, words[1:])]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def slots(question):
    found = {
        "numbers": set(),
        "names": set(),
        "direction": set(),
        "aggregate": set(),
        "comparator": set(),
        "value": set(),
    }
    for sentence in re.split(r"(?<=[.?!])\s+", question):
        for i, tok in enumerate(TOKEN.findall(sentence)):
            low = tok.lower()
            if tok[0].isdigit():
                found["numbers"].add(tok)
            elif low in NUMBER_WORDS:
                found["numbers"].add(NUMBER_WORDS[low])
            elif i > 0 and tok[0].isupper():
                found["names"].add(low)
            if low in DIRECTION:
                found["direction"].add(DIRECTION[low])
            if low in AGGREGATE:
                found["aggregate"].add(AGGREGATE[low])
            if low in COMPARATOR:
                found["comparator"].add(COMPARATOR[low])
            if low in VALUE:
                found["value"].add(VALUE[low])
    return found


def conflicts(a, b):
    sa, sb = slots(a), slots(b)
    return [k for k in sa if sa[k] and sb[k] and sa[k] != sb[k]]


def compatible(a, b):
    return not conflicts(a, b)


class SemanticCache:
    def __init__(self, embedder, threshold=0.85, guard=False):
        self.embedder = embedder
        self.threshold = threshold
        self.guard = guard
        self.questions = []
        self.sqls = []
        self.results = []
        self.meta = {}
        self.matrix = None

    def __len__(self):
        return len(self.questions)

    def add(self, question, sql, result=None):
        self.add_many([(question, sql, result)])

    def add_many(self, items):
        if not items:
            return
        items = [tuple(item) + (None,) * (3 - len(item)) for item in items]
        vecs = self.embedder.encode([item[0] for item in items])
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])
        self.questions.extend(item[0] for item in items)
        self.sqls.extend(item[1] for item in items)
        self.results.extend(item[2] for item in items)

    def set_result(self, index, result):
        self.results[index] = result

    def search(self, question):
        start = time.perf_counter()
        if self.matrix is None:
            scores = np.zeros(0, dtype=np.float32)
            order = np.zeros(0, dtype=int)
        else:
            vec = self.embedder.encode([question])[0]
            scores = self.matrix @ vec
            order = np.argsort(-scores)
        ms = (time.perf_counter() - start) * 1000
        return {"question": question, "scores": scores, "order": order, "ms": ms}

    def pick(self, found, threshold=None, guard=None):
        threshold = self.threshold if threshold is None else threshold
        guard = self.guard if guard is None else guard
        for idx in found["order"]:
            score = float(found["scores"][idx])
            if score < threshold:
                return None
            if guard and not compatible(found["question"], self.questions[idx]):
                continue
            return {
                "index": int(idx),
                "score": score,
                "sql": self.sqls[idx],
                "result": self.results[idx],
                "matched_question": self.questions[idx],
            }
        return None

    def lookup(self, question, threshold=None, guard=None):
        found = self.search(question)
        return self.pick(found, threshold, guard), found["ms"]

    def save(self, path):
        np.save(
            path + ".npy",
            (
                self.matrix
                if self.matrix is not None
                else np.zeros((0, 0), dtype=np.float32)
            ),
        )
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.embedder.name,
                    "meta": self.meta,
                    "questions": self.questions,
                    "sqls": self.sqls,
                    "results": self.results,
                },
                f,
            )

    @classmethod
    def load(cls, path, embedder, threshold=0.85, guard=False):
        cache = cls(embedder, threshold, guard)
        with open(path + ".json", encoding="utf-8") as f:
            data = json.load(f)
        cache.questions = data["questions"]
        cache.sqls = data["sqls"]
        cache.results = data.get("results") or [None] * len(cache.questions)
        cache.meta = data.get("meta", {})
        matrix = np.load(path + ".npy")
        cache.matrix = matrix if len(cache.questions) else None
        return cache
