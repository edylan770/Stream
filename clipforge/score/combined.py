"""§6.5's `score_combined`, and the metric that says whether it means anything.

§6.5 calls this "the most valuable output":

> "`score_combined` is **NOT** a sum or average. It must reward windows where
> *both* profiles fire, because that intersection is the operator's actual
> differentiator: a flashy Hawkeye play *plus* a genuinely funny reaction."

and gives a formula that does not run:

```python
def combined(e, g, alpha=0.5):
    e_n, g_n = normalize(e), normalize(g)
    return (e_n ** alpha) * (g_n ** (1 - alpha)) * (e_n + g_n)
```

TWO THINGS ARE WRONG WITH IT, AND ONE IS SILENT

`normalize` is never defined anywhere in the spec. And a profile composite is a
sum of *signed* weighted z-scores, so a fractional power of a negative one is
NaN — `(-0.4) ** 0.5` is `nan` in numpy and raises in plain Python. A NaN score
would sort unpredictably and, before commit 31's json fix, would have taken the
whole review payload down.

WHAT `normalize` IS HERE, AND WHY NOT MIN-MAX

Divide by the per-stream maximum, clamped at zero. The obvious alternative —
min-max — forces whichever candidate happens to be the *weakest* in a profile to
exactly 0.0, and the product form then returns 0 for it whatever the other
profile said. On a stream where `gameplay` is mostly a marker detector (which is
what it is before Phase 7), the funniest unmarked moment of the stream would
land at the bottom of the section §7.4 puts first. Dividing by the max keeps
every ratio and sends only a genuinely non-positive composite to zero.

Two consequences, written down rather than discovered later:

- **The combined score ranks WITHIN a stream and is not comparable across
  streams.** The normaliser is per-stream by construction, which is the same
  property §6.2's rolling z-score deliberately has and for the same reason.
- **It is in 0..2** where the per-profile scores are raw composites. The review
  UI shows it as "score".
"""

from __future__ import annotations

import numpy as np

#: Guards the division when every candidate in a profile scored zero or less.
#: Not a tunable: it is the difference between "no signal" and a ZeroDivision.
_EPS = 1e-12


def normalise(scores: np.ndarray) -> np.ndarray:
    """§6.5's undefined `normalize`: share of the stream's best, clamped at 0.

    Negative means "below this five-minute window's average", which is not a
    weaker version of notable — it is the absence of it. Clamping rather than
    shifting keeps zero meaning zero, so `combined` can use it as the
    annihilator the product form needs.
    """
    values = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    peak = float(values.max()) if values.size else 0.0
    if peak <= _EPS:
        return np.zeros_like(values)
    return values / peak


def combined(e_n: np.ndarray, g_n: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """§6.5's product form, over already-normalised scores.

    "Any formulation is acceptable provided it satisfies: **high+high >>
    high+low.**" This one does, and the test says so in those terms rather than
    against a number.

    Both inputs are in 0..1, so no fractional power ever meets a negative.
    """
    e = np.clip(np.asarray(e_n, dtype=np.float64), 0.0, None)
    g = np.clip(np.asarray(g_n, dtype=np.float64), 0.0, None)
    a = float(alpha)
    return (e ** a) * (g ** (1.0 - a)) * (e + g)


def rank_agreement(primary: np.ndarray, other: np.ndarray, top_n: int = 20) -> dict:
    """How much two rankings of the same candidates actually differ.

    THE number this phase cannot answer for itself. §6.5 justifies the combined
    score as the *intersection* of mechanics and personality, but before Phase 7
    only 20% of `gameplay`'s weight has a producer and 71% of that is
    `marker_definite` — so the combined ranking may be an expensive restatement
    of the entertainment one. §17's own tuning target for
    `combined_score_alpha` is "whether combined winners are actually the best
    clips", which needs footage; this needs nothing and is the leading
    indicator.

    Spearman's rho computed directly rather than through `scipy.stats`: it is
    Pearson over ranks, ties averaged, and importing scipy.stats for six lines
    would be the only use of it in the project.

    An INVENTED metric name — §14's table has no row for it.
    """
    a = np.asarray(primary, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if a.size != b.size:
        raise ValueError("rankings must cover the same candidates")

    out: dict[str, float | int | None] = {"n": int(a.size), "top_n": int(top_n)}
    if a.size < 2:
        # One candidate has no ordering to agree or disagree about.
        out["spearman"] = None
        out["top_n_overlap"] = None
        return out

    ra, rb = ranks(a), ranks(b)
    sa, sb = float(ra.std()), float(rb.std())
    if sa <= _EPS or sb <= _EPS:
        # Every candidate tied in one of the rankings: correlation is undefined
        # rather than zero, and reporting 0.0 would read as "they disagree".
        out["spearman"] = None
    else:
        out["spearman"] = round(float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb)), 4)

    cut = min(int(top_n), int(a.size))
    top_a = set(np.argsort(-a, kind="stable")[:cut].tolist())
    top_b = set(np.argsort(-b, kind="stable")[:cut].tolist())
    out["top_n_overlap"] = round(len(top_a & top_b) / cut, 4) if cut else None
    return out


def ranks(values: np.ndarray) -> np.ndarray:
    """Ascending ranks with ties averaged, which is what Spearman requires.

    Public because §14's `signal_firing_rate_by_rating` needs the identical
    tie handling for its Mann-Whitney separation, and reaching into another
    module's underscore name is how two copies of a helper come to exist.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)

    # Average the ranks inside each run of equal values.
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks
