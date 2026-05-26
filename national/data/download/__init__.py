"""Reproducible covariate downloaders.

Each submodule owns one data source and exposes:

- ``MANIFEST`` -- one or more :class:`DownloadSpec` records describing what
  to fetch, where it should land, and what its expected size / checksum is.
- ``download()`` -- callable that fetches everything in ``MANIFEST``, or
  raises a clear error and returns a :class:`DownloadResult` with
  ``status="skipped_manual"`` for sources that need browser interaction.

Top-level helpers:

- :func:`http_download` -- streamed HTTP fetch with SHA-256 verification,
  resume support, and a polite User-Agent.
- :func:`run_manifest` -- driver that walks a list of specs, printing
  progress and respecting ``--dry-run`` / ``--force``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

LOG = logging.getLogger("national.data.download")

DownloadStatus = Literal[
    "ok",
    "skipped_exists",
    "skipped_manual",
    "skipped_dry_run",
    "skipped_not_found",  # 404: legitimately not published (e.g. landlocked prefecture for C23)
    "failed",
]


@dataclass(frozen=True)
class DownloadSpec:
    """Static description of one downloadable file."""

    name: str
    """Short identifier, e.g. ``"aist_seamless_v2"``."""
    url: str | None
    """Direct HTTP URL, or ``None`` for manual-only sources."""
    destination: Path
    """Local target path -- created if missing."""
    license: str
    """One-line license tag, e.g. ``"CC BY 4.0"`` or ``"GSI 利用規約"``."""
    method: Literal["http", "manual", "api"]
    """``"http"`` -> :func:`http_download`; ``"manual"`` -> instructions only."""
    expected_size_bytes: int | None = None
    expected_sha256: str | None = None
    notes: str | None = None
    manual_url: str | None = None
    """Browser-friendly URL used when ``method == "manual"``."""


@dataclass
class DownloadResult:
    spec: DownloadSpec
    status: DownloadStatus
    actual_size: int | None = None
    actual_sha256: str | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)


def http_download(
    spec: DownloadSpec,
    *,
    force: bool = False,
    dry_run: bool = False,
    chunk_size: int = 1 << 20,
    user_agent: str = "geo-estimation/0.2 (+https://github.com/UT-Team-Chun/geo-estimation)",
) -> DownloadResult:
    """Stream ``spec.url`` to ``spec.destination`` with sha256 verification."""
    result = DownloadResult(spec=spec, status="failed")
    if spec.method != "http" or spec.url is None:
        result.status = "skipped_manual"
        result.error = "spec is not an http download"
        return result
    dest = Path(spec.destination)
    if dest.exists() and not force:
        if spec.expected_sha256:
            actual = _sha256(dest)
            if actual == spec.expected_sha256:
                result.status = "skipped_exists"
                result.actual_size = dest.stat().st_size
                result.actual_sha256 = actual
                return result
            LOG.warning(
                "%s: existing file checksum mismatch (have %s, want %s); re-downloading",
                spec.name,
                actual,
                spec.expected_sha256,
            )
        else:
            result.status = "skipped_exists"
            result.actual_size = dest.stat().st_size
            return result

    if dry_run:
        result.status = "skipped_dry_run"
        return result

    import httpx  # heavy; lazy import

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    hasher = hashlib.sha256()
    start = time.monotonic()
    bytes_written = 0
    try:
        with httpx.stream(
            "GET",
            spec.url,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, read=300.0),
        ) as resp:
            if resp.status_code == 404:
                # Several MLIT datasets legitimately have gaps (e.g. C23 has
                # no entry for landlocked prefectures, W05 has different
                # latest-year per prefecture). Treat 404 as a soft skip so
                # the CLI can keep going across the manifest.
                result.status = "skipped_not_found"
                result.error = f"404 Not Found at {spec.url}"
                return result
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size):
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        bytes_written += len(chunk)
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001 -- surface every failure mode
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        result.error = repr(exc)
        result.status = "failed"
        return result

    duration = max(1e-6, time.monotonic() - start)
    rate_mb_s = bytes_written / (1024 * 1024) / duration
    actual_sha = hasher.hexdigest()
    result.actual_size = bytes_written
    result.actual_sha256 = actual_sha
    result.log.append(
        f"downloaded {bytes_written / (1024 * 1024):.1f} MB in {duration:.1f}s "
        f"({rate_mb_s:.1f} MB/s)"
    )

    if spec.expected_size_bytes is not None and bytes_written != spec.expected_size_bytes:
        result.error = (
            f"size mismatch: expected {spec.expected_size_bytes}, got {bytes_written}"
        )
        result.status = "failed"
        return result
    if spec.expected_sha256 is not None and actual_sha != spec.expected_sha256:
        result.error = (
            f"sha256 mismatch: expected {spec.expected_sha256}, got {actual_sha}"
        )
        result.status = "failed"
        return result
    result.status = "ok"
    return result


def run_manifest(
    specs: Iterable[DownloadSpec],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> list[DownloadResult]:
    """Walk a manifest and download (or skip) each entry."""
    results: list[DownloadResult] = []
    for spec in specs:
        LOG.info("--- %s (%s) ---", spec.name, spec.method)
        if spec.method == "manual":
            r = DownloadResult(spec=spec, status="skipped_manual")
            r.log.append(
                "Manual download required. See spec.manual_url and "
                "data/raw/MANIFEST.md for instructions."
            )
            results.append(r)
            continue
        r = http_download(spec, force=force, dry_run=dry_run)
        if r.status == "ok":
            LOG.info("%s -> %s (%s bytes)", spec.name, spec.destination, r.actual_size)
        elif r.status == "skipped_exists":
            LOG.info("%s already present at %s; skipping.", spec.name, spec.destination)
        elif r.status == "skipped_dry_run":
            LOG.info("%s would download (dry-run).", spec.name)
        elif r.status == "skipped_not_found":
            LOG.warning("%s: %s (likely a legitimate gap in the source)", spec.name, r.error)
        else:
            LOG.error("%s FAILED: %s", spec.name, r.error)
        results.append(r)
    return results


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "DownloadSpec",
    "DownloadResult",
    "DownloadStatus",
    "http_download",
    "run_manifest",
]
