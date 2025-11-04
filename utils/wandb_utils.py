from __future__ import annotations
from typing import Any, Dict, Optional

_wandb = None


def _ensure_imported():
    global _wandb
    if _wandb is None:
        import importlib
        _wandb = importlib.import_module("wandb")


def init_run(project: str, config: Dict[str, Any], group: Optional[str] = None, name: Optional[str] = None, mode: Optional[str] = None) -> None:
    _ensure_imported()
    kwargs: Dict[str, Any] = {"project": project, "config": config}
    if group:
        kwargs["group"] = group
    if name:
        kwargs["name"] = name
    if mode:
        kwargs["mode"] = mode
    _wandb.init(**kwargs)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    _ensure_imported()
    _wandb.log(metrics, step=step)


def log_checkpoint(path: str) -> None:
    _ensure_imported()
    try:
        _wandb.save(path)
    except Exception:
        # Best-effort; continue even if artifact upload fails
        pass


def finish() -> None:
    _ensure_imported()
    _wandb.finish()


