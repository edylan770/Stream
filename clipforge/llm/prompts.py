"""Prompt text lives in a YAML file, not in Python.

§17 wants every tunable in config, and a prompt is the most tunable thing in
the system: it is the one part of §9.4 and §10.3 whose quality cannot be
measured here at all, so it is the part most certain to be rewritten. Editing
one must not mean editing code, and must not mean a diff a reviewer has to read
around string escaping.

Same shape as `phrases.yaml` and `crop_templates.yaml`: a file named from
`clipforge.yaml`, resolved against the config directory, parsed by the module
that owns it.

**`llm:` is a top-level subtree, outside `VERSIONED_SUBTREES`.** `config.py`
versions only `extract` and `score`, and a prompt edit must never invalidate a
candidate or force a re-score -- the same reasoning that put `render:`,
`previews:`, `search:` and `digest:` outside it.

**Substitution is `$name`, not `{name}`.** Prompts contain JSON examples, and
JSON is made of braces; `str.format` on a template holding a schema example is
a landmine waiting for the first prompt that inlines one. `$` appears in no
prompt here, and `Template.substitute` raises on an unknown placeholder rather
than silently leaving it in the text a model then reads.

**`digest_of` is why the hash exists.** §9.1 says digests are kept forever and
are never regenerated. Two digests of one stream produced by two different
prompts are otherwise indistinguishable in origin, so the digest stage records
this hash beside its output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from string import Template

import yaml

from clipforge.config import CONFIG_DIR
from clipforge.llm.sources import LLMError


def prompts_path(cfg) -> Path:
    path = Path(str(cfg.get("llm.prompts.file", "prompts.yaml")))
    if not path.is_absolute():
        path = CONFIG_DIR / path
    return path


def load(cfg) -> dict[str, str]:
    """Every prompt in the configured file, by name."""
    path = prompts_path(cfg)
    if not path.is_file():
        raise LLMError(f"prompt file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts") or {}
    if not prompts:
        raise LLMError(f"{path} declares no prompts")
    return {str(name): str(body) for name, body in prompts.items()}


def get(cfg, name: str) -> str:
    """One prompt's raw template.

    A missing name lists what the file does hold. A prompt referenced by code
    that is not in the file is a typo on one side or the other, and guessing
    which would mean sending a model an empty instruction.
    """
    prompts = load(cfg)
    if name not in prompts:
        raise LLMError(
            f"no prompt {name!r} in {prompts_path(cfg)}. "
            f"It holds: {', '.join(sorted(prompts))}.")
    return prompts[name]


def render(cfg, name: str, /, **values) -> str:
    """One prompt with its placeholders filled in."""
    template = get(cfg, name)
    try:
        return Template(template).substitute(values)
    except KeyError as exc:
        raise LLMError(
            f"prompt {name!r} uses ${exc.args[0]}, which was not supplied. "
            f"Given: {', '.join(sorted(values)) or 'nothing'}.") from None
    except ValueError as exc:
        raise LLMError(f"prompt {name!r} is malformed: {exc}") from None


def digest_of(cfg, name: str) -> str:
    """A stable short hash of one prompt's text, for storing with output.

    Of the template, deliberately, not of the rendered prompt: the question a
    stored hash answers is "which version of the instructions produced this",
    and the rendered text also carries the stream's own content.
    """
    return hashlib.sha256(get(cfg, name).encode("utf-8")).hexdigest()[:12]
