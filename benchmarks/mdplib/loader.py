"""Fetch and parse the MDPLIB MMDP (MaxMin diversity) instance sets.

The archive (Glover / Geo / Ran sets, 195 instances, plus published best-known values)
is downloaded from the maintainers' site at run time and cached locally — MDPLIB's
citation-based terms mean instances are never committed to this repo. When using these
instances, cite: Marti, Martinez-Gavara, Perez-Pelo & Sanchez-Oro (2022), "A review on
discrete diversity and dispersion maximization from an OR perspective", EJOR 299(3)
(MDPLIB 2.0).

Formats (verified against the archive):
  - Glover / Geo: line 1 = n, line 2 = d, then n rows ``index coord_1 ... coord_d``.
  - Ran: line 1 = n, then an ``i j distance`` edge list (all unordered pairs).

The MMDP instances fix the ground set only; the selection size k is supplied by the
caller (published results pair each instance with specific k values).
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem

_ARCHIVE_URL = "https://www.uv.es/rmarti/paper/results/mmdp%20instances.zip"
_CACHE_DIR = Path("reports/benchmarks/mdplib")
FAMILIES = ("Glover", "Geo", "Ran")


def fetch_mmdp_archive(cache_dir: Path = _CACHE_DIR) -> Path:
    """Download (once) and extract the MMDP instance archive; return the instances dir."""
    instances_dir = cache_dir / "instances"
    if instances_dir.is_dir():
        return instances_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "mmdp_instances.zip"
    if not archive.exists():
        with urllib.request.urlopen(_ARCHIVE_URL, timeout=60) as response:  # noqa: S310 -- fixed https URL
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as z:
        z.extractall(cache_dir)
    return instances_dir


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
