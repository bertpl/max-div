"""MDPLIB MMDP instance loading (external benchmark library; fetched, never redistributed)."""

from .loader import fetch_mmdp_archive, list_instances, load_instance

__all__ = [
    "fetch_mmdp_archive",
    "list_instances",
    "load_instance",
]
