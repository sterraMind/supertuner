"""Engine configuration and the per-command execution context.

Everything here is derived from `SUPERTUNER_*` environment variables; no
absolute path is ever hardcoded (see docs/ARCHITECTURE.md §5 "No path
guessing"). Relative values resolve against `SUPERTUNER_HOME`, which makes the
result independent of the shell's working directory.

| variable                     | default                |
| ---------------------------- | ---------------------- |
| SUPERTUNER_HOME              | ~/.supertuner          |
| SUPERTUNER_MODELS_DIR        | $HOME/models           |
| SUPERTUNER_DATASETS_DIR      | $HOME/datasets         |
| SUPERTUNER_CACHE_DIR         | $HOME/cache            |
| SUPERTUNER_MUSUBI_PATH       | unset (module probe)   |
| SUPERTUNER_ENGINE_LOG_LEVEL  | info                   |
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import BaseModel, Field

from .control import CancelToken

LOG_LEVELS = ("debug", "info", "warning", "error")
DEFAULT_LOG_LEVEL = "info"

#: Names used in the `--json` output of `env` / `plugins`.
ENV_PREFIX = "SUPERTUNER_"


def _expand_home(value: str) -> Path:
    """Expand `~` and normalise separators without touching the filesystem."""
    raw = value.strip().strip('"').strip("'")
    return Path(os.path.expanduser(raw))


def _resolve(value: str, base: Path) -> Path:
    """Absolute paths pass through; relative paths resolve against `base`."""
    path = _expand_home(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def repo_root() -> Path:
    """Repository root, derived from this package's location.

    `engine/src/supertuner_engine/config.py` -> parents[3]. Installed as a wheel
    the walk-up still yields the checkout when the package lives in `engine/`;
    otherwise the current working directory is the best available answer.
    """
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "engine" / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


class EngineConfig(BaseModel):
    """Resolved engine settings. Immutable; call `with_*` helpers for variants."""

    home: Path
    models_dir: Path
    datasets_dir: Path
    cache_dir: Path
    musubi_path: Path | None = None
    log_level: str = DEFAULT_LOG_LEVEL
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EngineConfig:
        environ = os.environ if env is None else env
        warnings: list[str] = []

        raw_home = environ.get("SUPERTUNER_HOME") or str(Path.home() / ".supertuner")
        home = _expand_home(raw_home)
        if not home.is_absolute():
            home = (Path.cwd() / home).resolve()

        models_dir = _resolve(environ.get("SUPERTUNER_MODELS_DIR") or "models", home)
        datasets_dir = _resolve(environ.get("SUPERTUNER_DATASETS_DIR") or "datasets", home)
        cache_dir = _resolve(environ.get("SUPERTUNER_CACHE_DIR") or "cache", home)

        musubi_raw = (environ.get("SUPERTUNER_MUSUBI_PATH") or "").strip()
        musubi_path = _resolve(musubi_raw, home) if musubi_raw else None

        log_level = (environ.get("SUPERTUNER_ENGINE_LOG_LEVEL") or DEFAULT_LOG_LEVEL).strip().lower()
        if log_level not in LOG_LEVELS:
            warnings.append(
                f"SUPERTUNER_ENGINE_LOG_LEVEL='{log_level}' 不是有效级别 {LOG_LEVELS}；已回退到 '{DEFAULT_LOG_LEVEL}'"
            )
            log_level = DEFAULT_LOG_LEVEL

        return cls(
            home=home,
            models_dir=models_dir,
            datasets_dir=datasets_dir,
            cache_dir=cache_dir,
            musubi_path=musubi_path,
            log_level=log_level,
            warnings=warnings,
        )

    def ensure_dirs(self) -> None:
        """Create the engine-owned directories. Safe to call repeatedly."""
        for path in (self.home, self.models_dir, self.datasets_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.home / "runs" / run_id

    def context(self, run_dir: Path | None = None) -> Context:
        normalised = Path(run_dir).expanduser().resolve() if run_dir is not None else None
        return Context(
            home=self.home,
            models_dir=self.models_dir,
            datasets_dir=self.datasets_dir,
            cache_dir=self.cache_dir,
            musubi_path=self.musubi_path,
            log_level=self.log_level,
            run_dir=normalised,
            cancel=CancelToken(run_dir=normalised),
        )


@dataclass(frozen=True)
class Context:
    """Paths and settings one compile/run sees. Frozen: derive variants with `replace`."""

    home: Path
    models_dir: Path
    datasets_dir: Path
    cache_dir: Path
    repo_root: Path = field(default_factory=repo_root)
    musubi_path: Path | None = None
    log_level: str = DEFAULT_LOG_LEVEL
    run_dir: Path | None = None
    cancel: CancelToken = field(default_factory=CancelToken)

    def with_run_dir(self, run_dir: Path) -> Context:
        resolved = Path(run_dir).expanduser().resolve()
        return replace(self, run_dir=resolved, cancel=CancelToken(run_dir=resolved))

    @property
    def output_dir(self) -> Path:
        """Where artifacts land: `<run_dir>/output`, or the models dir without a run."""
        base = self.run_dir if self.run_dir is not None else self.cache_dir
        return base / "output"

    @property
    def logs_dir(self) -> Path:
        base = self.run_dir if self.run_dir is not None else self.cache_dir
        return base / "logs"

    def ensure_run_dirs(self) -> None:
        """Create the run layout the runtime and backends write into."""
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, family: str, explicit: str | None = None) -> Path | None:
        """Resolve base weights: explicit override wins, else `<models_dir>/<family>`."""
        if explicit:
            return _resolve(explicit, self.home)
        return self.models_dir / family


def load_context(run_dir: Path | None = None, env: Mapping[str, str] | None = None) -> Context:
    return EngineConfig.from_env(env).context(run_dir)


_CACHED: EngineConfig | None = None


def cached_config() -> EngineConfig:
    """Process-wide configuration used by the CLI when no explicit env is given."""
    global _CACHED
    if _CACHED is None:
        _CACHED = EngineConfig.from_env(os.environ)
    return _CACHED