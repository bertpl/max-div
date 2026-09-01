import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.mdplib import list_instances, load_instance
from benchmarks.mdplib.loader import MdplibFetchError, fetch_mmdp_archive
from max_div.problem import DistanceMaxDivProblem, VectorMaxDivProblem


# =================================================================================================
#  Helpers
# =================================================================================================
def _stand_in_archive() -> bytes:
    """Build a minimal archive with the layout the loader extracts (an `instances/<family>/` tree)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("instances/Glover/Glover 10 1.txt", "10\n2\n")
    return buffer.getvalue()


def _serve(monkeypatch: pytest.MonkeyPatch, payloads: list[bytes | Exception]) -> list[str]:
    """Make `urlopen` answer successive calls with `payloads`; return the list of URLs requested."""
    requested: list[str] = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    def fake_urlopen(url: str, timeout: float) -> _Response:
        requested.append(url)
        payload = payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return requested


def _fetch(tmp_path: Path, archive: bytes) -> Path:
    """Fetch into `tmp_path`, expecting the size and digest of `archive`, with no retry pause."""
    return fetch_mmdp_archive(
        cache_dir=tmp_path,
        url="https://example.test/mmdp.zip",
        expected_size=len(archive),
        expected_sha256=hashlib.sha256(archive).hexdigest(),
        retry_delay_sec=0.0,
    )


# =================================================================================================
#  Archive download & verification (no network)
# =================================================================================================
def test_short_download_is_retried_until_intact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A truncated transfer is discarded and the next attempt's intact archive is extracted."""
    # --- arrange ----------------------
    archive = _stand_in_archive()
    requested = _serve(monkeypatch, [archive[:-20], archive])

    # --- act --------------------------
    instances_dir = _fetch(tmp_path, archive)

    # --- assert -----------------------
    assert len(requested) == 2
    assert (instances_dir / "Glover" / "Glover 10 1.txt").is_file()
    assert (tmp_path / "mmdp_instances.zip").read_bytes() == archive


def test_failed_download_is_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A download that errors out counts as a failed attempt and is retried."""
    # --- arrange ----------------------
    archive = _stand_in_archive()
    requested = _serve(monkeypatch, [urllib.error.URLError("connection reset"), archive])

    # --- act --------------------------
    instances_dir = _fetch(tmp_path, archive)

    # --- assert -----------------------
    assert len(requested) == 2
    assert instances_dir.is_dir()


def test_corrupt_cached_archive_is_replaced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A cached archive failing the check is downloaded again instead of being extracted."""
    # --- arrange ----------------------
    archive = _stand_in_archive()
    (tmp_path / "mmdp_instances.zip").write_bytes(archive[:-20])
    requested = _serve(monkeypatch, [archive])

    # --- act --------------------------
    instances_dir = _fetch(tmp_path, archive)

    # --- assert -----------------------
    assert len(requested) == 1
    assert instances_dir.is_dir()
    assert (tmp_path / "mmdp_instances.zip").read_bytes() == archive


def test_intact_cached_archive_is_not_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """An archive passing the check is extracted without touching the network."""
    # --- arrange ----------------------
    archive = _stand_in_archive()
    (tmp_path / "mmdp_instances.zip").write_bytes(archive)
    requested = _serve(monkeypatch, [])

    # --- act --------------------------
    instances_dir = _fetch(tmp_path, archive)

    # --- assert -----------------------
    assert requested == []
    assert instances_dir.is_dir()


def test_persistently_short_download_raises_naming_the_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When every attempt is short, the error names the URL and the byte counts, and no archive is kept."""
    # --- arrange ----------------------
    archive = _stand_in_archive()
    requested = _serve(monkeypatch, [archive[:-20]] * 3)

    # --- act / assert -----------------
    with pytest.raises(MdplibFetchError, match=r"example\.test/mmdp\.zip.*expected \d+ bytes.*got \d+ bytes"):
        _fetch(tmp_path, archive)
    assert len(requested) == 3
    assert not (tmp_path / "mmdp_instances.zip").exists()


# =================================================================================================
#  Instance loading (network-dependent)
# =================================================================================================
@pytest.fixture(scope="module")
def mdplib_available() -> None:
    """Skip the module when the MDPLIB archive cannot be fetched (network robustness)."""
    try:
        list_instances("Glover")
    except (urllib.error.URLError, OSError, MdplibFetchError) as e:  # pragma: no cover -- network-dependent
        pytest.skip(f"MDPLIB archive not fetchable: {e}")


def test_families_have_expected_instance_counts(mdplib_available):
    # --- act / assert ------------------------------------
    assert len(list_instances("Glover")) == 15
    assert len(list_instances("Geo")) == 60
    assert len(list_instances("Ran")) == 60


def test_coordinate_instance_loads_as_vector_problem(mdplib_available):
    # --- act ---------------------------------------------
    problem = load_instance("Geo", "Geo 100 1.txt", k=10)

    # --- assert ------------------------------------------
    assert isinstance(problem, VectorMaxDivProblem)
    assert problem.n == 100
    assert problem.k == 10


def test_edge_list_instance_loads_as_distance_problem(mdplib_available):
    # --- act ---------------------------------------------
    problem = load_instance("Ran", "Ran 100 1.txt", k=10)

    # --- assert ------------------------------------------
    assert isinstance(problem, DistanceMaxDivProblem)
    assert problem.n == 100
    condensed = problem.condensed_distances()
    assert condensed.shape == (100 * 99 // 2,)
    assert np.all(condensed > 0)


def test_unknown_family_is_rejected():
    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="Unknown MMDP family"):
        list_instances("Nope")
