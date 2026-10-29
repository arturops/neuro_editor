"""Settings, read from environment variables (see PLAN.md section 5)."""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VIEWER_MODES = ("per_user", "shared")


def _parse_triple(name: str, default: str, allow_zero: bool) -> tuple[int, int, int]:
    raw = os.environ.get(name, default)
    try:
        parts = tuple(int(p) for p in raw.replace(" ", "").split(","))
    except ValueError:
        raise ValueError(f"{name}={raw!r} must be three whole numbers like 30,300,300") from None
    if len(parts) != 3:
        raise ValueError(f"{name}={raw!r} must have exactly three numbers (z,y,x)")
    low = 0 if allow_zero else 1
    if any(p < low for p in parts):
        raise ValueError(f"{name}={raw!r}: every number must be at least {low}")
    return parts  # type: ignore[return-value]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    port: int
    public_url: str
    viewer_mode: str
    tokens: tuple[str, ...]
    tokens_were_generated: bool
    crop_start: tuple[int, int, int]  # (z, y, x)
    crop_size: tuple[int, int, int]  # (z, y, x)
    block_size: tuple[int, int, int]  # (z, y, x)
    dataset_url: str
    hdf_name: str

    @property
    def hdf_path(self) -> Path:
        return self.data_dir / self.hdf_name

    @property
    def db_path(self) -> Path:
        return self.data_dir / "edits.sqlite"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("DATA_DIR", "./data")).expanduser()
        port = int(os.environ.get("PORT", "8080"))
        public_url = os.environ.get("PUBLIC_URL", "").strip().rstrip("/") or f"http://localhost:{port}"

        mode = os.environ.get("VIEWER_MODE", "per_user").strip().lower()
        if mode not in VIEWER_MODES:
            raise ValueError(f"VIEWER_MODE={mode!r} must be one of {VIEWER_MODES}")

        raw_tokens = [t.strip() for t in os.environ.get("VIEWER_TOKENS", "").split(",") if t.strip()]
        generated = False
        if not raw_tokens:
            generated = True
            count = 1 if mode == "shared" else 3
            raw_tokens = [secrets.token_urlsafe(9) for _ in range(count)]
        for tok in raw_tokens:
            if not TOKEN_PATTERN.match(tok):
                raise ValueError(f"Token {tok!r} may only use letters, digits, '-' and '_'")
        if len(set(raw_tokens)) != len(raw_tokens):
            raise ValueError("VIEWER_TOKENS must not repeat a token")

        return cls(
            data_dir=data_dir,
            port=port,
            public_url=public_url,
            viewer_mode=mode,
            tokens=tuple(raw_tokens),
            tokens_were_generated=generated,
            crop_start=_parse_triple("CROP_START", "30,300,300", allow_zero=True),
            crop_size=_parse_triple("CROP_SIZE", "64,512,512", allow_zero=False),
            block_size=_parse_triple("BLOCK_SIZE", "8,64,64", allow_zero=False),
            dataset_url=os.environ.get("DATASET_URL", "https://cremi.org/static/data/sample_A_20160501.hdf"),
            hdf_name=os.environ.get("HDF_NAME", "sample_A_20160501.hdf"),
        )

    def short_token_warnings(self) -> list[str]:
        return [
            f"Token {t[:3]}... is short (<12 characters); anyone who guesses it can edit."
            for t in self.tokens
            if len(t) < 12
        ]
