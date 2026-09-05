"""Run the three head-to-head comparison tiers back to back, as one campaign.

Run detached, so the Bash tool's or terminal's session limit cannot end it:

    nohup uv run --group benchmarks --python 3.14 python -u -m benchmarks.campaign > campaign.log 2>&1 &

Each tier's driver skips cells already on file, so a campaign that stops resumes by rerunning.
A tier that raises is reported and the next tier still runs; the log's final lines say which
tiers completed.
"""

import time
import traceback

from benchmarks.tier1 import full as tier1
from benchmarks.tier2 import full as tier2
from benchmarks.tier3 import full as tier3

TIERS = (("tier 1", tier1.main), ("tier 2", tier2.main), ("tier 3", tier3.main))


def main() -> None:
    """Run every tier in order and print a per-tier summary at the end."""
    outcomes = []
    for name, run in TIERS:
        t0 = time.monotonic()
        print(f"=== {name} start {time.strftime('%H:%M:%S')}", flush=True)
        try:
            run()
            outcome = "completed"
        except Exception:  # noqa: BLE001 -- one tier's failure must not cancel the night
            traceback.print_exc()
            outcome = "FAILED"
        elapsed_min = (time.monotonic() - t0) / 60
        print(f"=== {name} {outcome} after {elapsed_min:.0f} min", flush=True)
        outcomes.append((name, outcome, elapsed_min))
    print("=== campaign summary", flush=True)
    for name, outcome, elapsed_min in outcomes:
        print(f"  {name}: {outcome} ({elapsed_min:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
