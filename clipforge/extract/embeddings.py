"""Per-segment embeddings via Ollama (spec §5.10). Phase 2 — not built.

Notes: `bge-small-en-v1.5` at 384 dims is the recommended default (~300 MB for
100 streams). Brute-force numpy cosine over 200k vectors is ~50 ms, so no vector
database is required — §16 rejects one explicitly.
"""
