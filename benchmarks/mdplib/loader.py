"""Fetch and parse the MDPLIB MMDP (MaxMin diversity) instance sets.

The archive (Glover / Geo / Ran sets — 135 instance files, paired with multiple k
values in the published results — plus the best-known-value tables)
is downloaded from the maintainers' site at run time and cached locally — the
citation-based distribution terms mean instances are never committed to this repo. The
instance sets and the accompanying best-known values originate from Resende, Marti,
Gallego & Duarte (2010), "GRASP and path relinking for the max-min diversity problem",
Computers & Operations Research 37(3); the hosting library (MDPLIB) is surveyed in Marti,
Martinez-Gavara, Perez-Pelo & Sanchez-Oro (2022), EJOR 299(3).

The archive has not changed since 2008, so its size and SHA-256 are pinned here. A download
or cached copy is extracted only after it matches both; `_ensure_intact_archive` describes
what happens when it does not.

Formats (verified against the archive):
  - Glover / Geo: line 1 = n, line 2 = d, then n rows ``index coord_1 ... coord_d``.
  - Ran: line 1 = n, then an ``i j distance`` edge list (all unordered pairs).

The MMDP instances fix the ground set only; the selection size k is supplied by the
caller (published results pair each instance with specific k values).
"""

import hashlib
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem


class ArchiveDigest(NamedTuple):
    """A digest is an archive's byte count and hex SHA-256; two files match when their digests are equal."""

    size: int
    sha256: str

    @classmethod
    def from_file(cls, path: Path) -> "ArchiveDigest":
        """Compute the digest of the file at `path`."""
        data = path.read_bytes()
        return cls(len(data), hashlib.sha256(data).hexdigest())


_ARCHIVE_URL = "https://www.uv.es/rmarti/paper/results/mmdp%20instances.zip"
_ARCHIVE_DIGEST = ArchiveDigest(
    size=12_411_255, sha256="32c437ded98f9dfdc554d78db262c2d7cd1a516775efbd04487b3e24b4fde6fc"
)
_CACHE_DIR = Path("reports/benchmarks/mdplib")
_DOWNLOAD_ATTEMPTS = 3
_RETRY_DELAY_SEC = 5.0
FAMILIES = ("Glover", "Geo", "Ran")


class MdplibFetchError(RuntimeError):
    """The MMDP archive could not be downloaded intact within the allowed attempts."""


def fetch_mmdp_archive(
    cache_dir: Path = _CACHE_DIR,
    url: str = _ARCHIVE_URL,
    expected: ArchiveDigest = _ARCHIVE_DIGEST,
    retry_delay_sec: float = _RETRY_DELAY_SEC,
) -> Path:
    """Download (once) and extract the MMDP instance archive; return the instances dir.

    Args:
        cache_dir: Where the archive and its extracted instances live.
        url: Archive location; a parameter so tests can serve a stand-in archive.
        expected: Digest the downloaded archive must have.
        retry_delay_sec: Pause between download attempts.

    Raises:
        MdplibFetchError: If no attempt yields an archive with digest `expected`.
    """
    instances_dir = cache_dir / "instances"
    if instances_dir.is_dir():
        return instances_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "mmdp_instances.zip"
    _ensure_intact_archive(archive, url, expected, retry_delay_sec)
    with zipfile.ZipFile(archive) as z:
        z.extractall(cache_dir)
    return instances_dir


def _ensure_intact_archive(archive: Path, url: str, expected: ArchiveDigest, retry_delay_sec: float) -> None:
    """Leave an archive with digest `expected` at `archive`, downloading as needed.

    - A cached file whose digest differs is deleted and downloaded again.
    - A download whose digest differs, or that fails outright, is deleted; the archive is
      downloaded up to `_DOWNLOAD_ATTEMPTS` times.

    The error raised when the attempts run out describes every failed one, so a truncated
    transfer is reported with its byte count, not as an EOFError from the extraction.
    """
    if archive.exists():
        if ArchiveDigest.from_file(archive) == expected:
            return
        archive.unlink()
    failures: list[str] = []
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 -- defaults to a fixed https URL; tests override it
                archive.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as e:
            failures.append(f"attempt {attempt}: download failed ({e})")
        else:
            got = ArchiveDigest.from_file(archive)
            if got == expected:
                return
            failures.append(f"attempt {attempt}: got {got.size} bytes, sha256 {got.sha256}")
        archive.unlink(missing_ok=True)
        if attempt < _DOWNLOAD_ATTEMPTS:
            time.sleep(retry_delay_sec)
    raise MdplibFetchError(
        f"Could not download an intact MMDP archive from {url} "
        f"(expected {expected.size} bytes, sha256 {expected.sha256}): " + "; ".join(failures)
    )


def list_instances(family: str, cache_dir: Path = _CACHE_DIR) -> list[str]:
    """List instance file names of one family (``Glover`` / ``Geo`` / ``Ran``), sorted."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown MMDP family {family!r}; expected one of {FAMILIES}.")
    instances_dir = fetch_mmdp_archive(cache_dir)
    return sorted(p.name for p in (instances_dir / family).glob("*.txt"))


def load_instance(
    family: str,
    filename: str,
    k: int,
    diversity_metric: DiversityMetric = DiversityMetric.MIN_SEPARATION,
    cache_dir: Path = _CACHE_DIR,
) -> MaxDivProblem:
    """Load one MMDP instance as a MaxDivProblem.

    Glover/Geo instances carry coordinates and become vector problems; Ran instances
    carry a distance matrix and use the precomputed-distance input.

    Args:
        family: ``Glover``, ``Geo``, or ``Ran``.
        filename: Instance file name as returned by list_instances.
        k: Selection size (published results pair instances with specific k values).
        diversity_metric: Diversity metric of the constructed problem (published MMDP
            values correspond to MIN_SEPARATION).
        cache_dir: Archive cache location.
    """
    instances_dir = fetch_mmdp_archive(cache_dir)
    path = instances_dir / family / filename
    if family == "Ran":
        return MaxDivProblem.from_distances(_parse_edge_list(path), k=k, diversity_metric=diversity_metric)
    return MaxDivProblem.new(_parse_coordinates(path), k=k, diversity_metric=diversity_metric)


def _parse_coordinates(path: Path) -> np.ndarray:
    """Parse a Glover/Geo coordinate file into an (n, d) float32 array."""
    lines = path.read_text().split("\n")
    n, d = int(lines[0]), int(lines[1])
    vectors = np.empty((n, d), dtype=np.float32)
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        idx = int(parts[0])
        vectors[idx] = [float(v) for v in parts[1 : d + 1]]
    return vectors


def _parse_edge_list(path: Path) -> np.ndarray:
    """Parse a Ran edge-list file into a symmetric (n, n) float32 distance matrix."""
    lines = path.read_text().split("\n")
    n = int(lines[0])
    distances = np.zeros((n, n), dtype=np.float32)
    for line in lines[1:]:
        if not line.strip():
            continue
        i_str, j_str, dist_str = line.split()
        i, j = int(i_str), int(j_str)
        distances[i, j] = distances[j, i] = float(dist_str)
    return distances
