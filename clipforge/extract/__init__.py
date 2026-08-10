"""Signal extraction — one module per signal family (spec §5).

Extraction is expensive and runs once per stream. It is separated from scoring
by a hard boundary (§5.1, C3): bad weights cost nothing, bad extraction costs
a re-run over the whole back catalogue.
"""