"""Fetch and parse the MDPLIB MMDP (MaxMin diversity) instance sets.

The archive (Glover / Geo / Ran sets — 135 instance files, paired with multiple k
values in the published results — plus the best-known-value tables)
is downloaded from the maintainers' site at run time and cached locally — the
citation-based distribution terms mean instances are never committed to this repo. The
instance sets and the accompanying best-known values originate from Resende, Marti,
Gallego & Duarte (2010), "GRASP and path relinking for the max-min diversity problem",
Computers & Operations Research 37(3); the hosting library (MDPLIB) is surveyed in Marti,
Martinez-Gavara, Perez-Pelo & Sanchez-Oro (2022), EJOR 299(3).

The archive has not changed since 2008, so its size and SHA-256 are pinned here and every
download is checked against them before it is extracted: a short or altered download is
discarded and retried, and a cached archive that fails the check is replaced. That keeps a
truncated transfer from failing the extraction with an unhelpful EOFError, and means a
cached copy (locally, or in CI's cache) is always an intact one.

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

import numpy as np

from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem

_ARCHIVE_URL = "https://www.uv.es/rmarti/paper/results/mmdp%20instances.zip"
_ARCHIVE_SIZE = 12_411_255
_ARCHIVE_SHA256 = "32c437ded98f9dfdc554d78db262c2d7cd1a516775efbd04487b3e24b4fde6fc"
_CACHE_DIR = Path("reports/benchmarks/mdplib")
_DOWNLOAD_ATTEMPTS = 3
_RETRY_DELAY_SEC = 5.0
FAMILIES = ("Glover", "Geo", "Ran")


class MdplibFetchError(RuntimeError):
    """The MMDP archive could not be downloaded intact within the allowed attempts."""


def fetch_mmdp_archive(
    cache_dir: Path = _CACHE_DIR,
    url: str = _ARCHIVE_URL,
    expected_size: int = _ARCHIVE_SIZE,
    expected_sha256: str = _ARCHIVE_SHA256,
    retry_delay_sec: float = _RETRY_DELAY_SEC,
) -> Path:
    """Download (once) and extract the MMDP instance archive; return the instances dir.

    The archive is extracted only after its size and SHA-256 match the expected values; see
    `_ensure_intact_archive` for how a failing download or cached copy is handled.

    Args:
        cache_dir: Where the archive and its extracted instances live.
        url: Archive location; a parameter so tests can serve a stand-in archive.
        expected_size: Byte count the downloaded archive must have.
        expected_sha256: Hex digest the downloaded archive must have.
        retry_delay_sec: Pause between download attempts.

    Raises:
        MdplibFetchError: If no attempt yields an archive matching `expected_size` and
            `expected_sha256`.
    """
    instances_dir = cache_dir / "instances"
    if instances_dir.is_dir():
        return instances_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "mmdp_instances.zip"
    _ensure_intact_archive(archive, url, expected_size, expected_sha256, retry_delay_sec)
    with zipfile.ZipFile(archive) as z:
        z.extractall(cache_dir)
    return instances_dir


def _ensure_intact_archive(
    archive: Path, url: str, expected_size: int, expected_sha256: str, retry_delay_sec: float
) -> None:
    """Leave an archive matching the expected size and digest at `archive`, downloading as needed.

    A cached file that fails the check is deleted and downloaded again; a download that fails
    the check, or fails outright, is deleted and retried up to `_DOWNLOAD_ATTEMPTS` times. Every
    failed attempt is described in the error raised when the attempts run out, so a truncated
    transfer is reported with its byte count instead of surfacing later as an EOFError.
    """
    if archive.exists():
        if _matches(archive, expected_size, expected_sha256):
            return
        archive.unlink()
    failures: list[str] = []
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 -- fixed https URL
                archive.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as e:
            failures.append(f"attempt {attempt}: download failed ({e})")
        else:
            if _matches(archive, expected_size, expected_sha256):
                return
            size, sha256 = _size_and_sha256(archive)
            failures.append(f"attempt {attempt}: got {size} bytes, sha256 {sha256}")
        archive.unlink(missing_ok=True)
        if attempt < _DOWNLOAD_ATTEMPTS:
            time.sleep(retry_delay_sec)
    raise MdplibFetchError(
        f"Could not download an intact MMDP archive from {url} "
        f"(expected {expected_size} bytes, sha256 {expected_sha256}): " + "; ".join(failures)
    )


def _matches(archive: Path, expected_size: int, expected_sha256: str) -> bool:
    """Report whether `archive` has the expected byte count and SHA-256 digest."""
    return _size_and_sha256(archive) == (expected_size, expected_sha256)


def _size_and_sha256(archive: Path) -> tuple[int, str]:
    """Return the byte count and hex SHA-256 digest of `archive`."""
    data = archive.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


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
