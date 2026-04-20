"""Cross-backend machine-type selector with a cost+availability tiebreaker.

The dispatch: given a list of candidate machine types (each already
verified to meet the function's resource constraints — `brev search`
does this server-side; a hand-curated GCP/AWS catalog does it locally
once #25 lands), pick the cheapest option, with a small-cost-difference
tiebreaker that falls through to an availability hint when one is
available.

Why a tolerance rather than strict cheapest-wins? The $0.01/hr spread
between the top two Brev matches is not worth a job that sits 5 minutes
in a queue because the cheaper region has no supply. The tolerance
defaults to 5% — wide enough to prefer a readily-available box, tight
enough that we never jump several price tiers.

Callers construct `Candidate` objects from whatever their backend
exposes. `availability_hint` is an opaque "lower is better" float
(seconds to start, queue depth, capacity bucket — backend's choice).
When no candidate in the tolerance band has a hint, we fall back to
the original (cheapest-first) order so behavior doesn't silently
diverge from the pre-selector implementation.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

DEFAULT_COST_TOLERANCE = 0.05


@dataclass(frozen=True)
class Candidate:
    name: str
    hourly_usd: float
    availability_hint: Optional[float] = None
    region: Optional[str] = None
    # Backend-specific passthrough. Consumers may stash the whole
    # `brev search` row here so the caller can pull out fields the
    # selector doesn't care about.
    raw: Optional[dict] = None


@dataclass(frozen=True)
class MachineChoice:
    name: str
    hourly_usd: float
    region: Optional[str]
    # Human-readable reason the selector landed on this candidate.
    # Useful in logs ("cheapest", "within 5% of cheapest but lower ETA").
    reason: str


def pick_machine(
    candidates: Iterable[Candidate],
    *,
    cost_tolerance: float = DEFAULT_COST_TOLERANCE,
) -> Optional[MachineChoice]:
    """Choose the best-fit candidate with a cost-tolerant tiebreaker.

    Algorithm:
      1. Drop candidates without a positive hourly_usd (can't compare).
      2. Sort by hourly_usd ascending.
      3. Collect the tolerance band: everything at or below
         cheapest * (1 + cost_tolerance).
      4. If at least one candidate in the band has a numeric
         `availability_hint`, pick the one with the lowest hint.
         Ties within the hint are broken by cost.
      5. Otherwise pick the cheapest (= band[0]).

    Returns None if no candidate has a usable price.
    """
    pool = [c for c in candidates if c.hourly_usd is not None and c.hourly_usd > 0]
    if not pool:
        return None
    pool.sort(key=lambda c: c.hourly_usd)

    cheapest = pool[0].hourly_usd
    band_ceiling = cheapest * (1.0 + cost_tolerance)
    band = [c for c in pool if c.hourly_usd <= band_ceiling]

    # Availability tiebreaker inside the band.
    hinted = [c for c in band if c.availability_hint is not None]
    if hinted:
        hinted.sort(key=lambda c: (c.availability_hint, c.hourly_usd))
        winner = hinted[0]
        if winner is band[0]:
            reason = "cheapest (also lowest availability hint in tolerance band)"
        else:
            reason = (
                f"within {int(cost_tolerance * 100)}% of cheapest "
                f"${cheapest:.3f}/hr but lower availability hint "
                f"({winner.availability_hint} vs {band[0].availability_hint})"
            )
        return MachineChoice(
            name=winner.name,
            hourly_usd=winner.hourly_usd,
            region=winner.region,
            reason=reason,
        )

    # No availability signal anywhere in the band → cost wins.
    winner = band[0]
    if len(band) == 1:
        reason = "cheapest"
    else:
        reason = (
            f"cheapest; {len(band) - 1} other candidate(s) within "
            f"{int(cost_tolerance * 100)}% but none had an availability hint"
        )
    return MachineChoice(
        name=winner.name,
        hourly_usd=winner.hourly_usd,
        region=winner.region,
        reason=reason,
    )
