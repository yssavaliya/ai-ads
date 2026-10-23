"""Small, deliberate concurrency primitives.

The pipelines are I/O bound (LLM, image providers, scrapers) and were strictly sequential. Work
that is independent runs through `parallel_map` with a hard worker cap; work that shares a rate
limit goes through `slots()`, a process-wide bounded semaphore, so nested parallelism (e.g. several
prompts each rendering several Reel beats) can never exceed the configured number of in-flight calls.

Every limit has a kill-switch: a value of 1 reproduces the old strictly sequential behaviour.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Outcome:
    """Result of one task: either a value or the exception it raised. Never both."""

    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


def _run[T, R](fn: Callable[[T], R], item: T) -> Outcome:
    try:
        return Outcome(value=fn(item))
    except BaseException as exc:  # noqa: BLE001 - captured and handed back to the caller
        return Outcome(error=exc)


def parallel_map[T, R](fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[Outcome]:
    """Apply `fn` to every item, at most `workers` at a time, preserving the input order.

    Exceptions are captured per item (one failure never cancels the others); the caller decides
    what to do with each. With workers <= 1 (or a single item) it runs inline on the calling
    thread: identical to a plain loop, no threads involved."""
    batch = list(items)
    workers = max(1, min(int(workers), len(batch)))
    if workers == 1:
        return [_run(fn, item) for item in batch]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pipeline") as pool:
        return [f.result() for f in [pool.submit(_run, fn, item) for item in batch]]


_slot_lock = threading.Lock()
_slot_registry: dict[tuple[str, int], threading.BoundedSemaphore] = {}


def slots(name: str, limit: int) -> threading.BoundedSemaphore:
    """Process-wide semaphore for a shared resource (`name`) with `limit` concurrent users."""
    limit = max(1, int(limit))
    with _slot_lock:
        return _slot_registry.setdefault((name, limit), threading.BoundedSemaphore(limit))
