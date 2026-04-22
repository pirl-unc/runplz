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


def pick_machines(
    candidates: Iterable[Candidate],
    *,
    n: int,
    cost_tolerance: float = DEFAULT_COST_TOLERANCE,
) -> list:
    """Return up to `n` ranked MachineChoices for fallback dispatch.

    Same preference order as `pick_machine` applied sequentially:
      1. Entries in the tolerance band (cheapest * (1 + cost_tolerance))
         come first. Within the band, availability-hinted entries beat
         unhinted (lower hint first); ties broken by cost.
      2. Entries outside the band come next, cheapest-first. This is
         the "if A fails try B" fallback lane — the tolerance band is
         about optimal first-pick, fallback is about resilience so
         we're okay reaching up the price ladder.

    Empty / all-no-price candidate sets return an empty list. `n` must
    be a positive int.
    """
    if n < 1:
        raise ValueError(f"pick_machines(n={n!r}): n must be a positive int.")

    pool = [c for c in candidates if c.hourly_usd is not None and c.hourly_usd > 0]
    if not pool:
        return []
    pool.sort(key=lambda c: c.hourly_usd)

    cheapest = pool[0].hourly_usd
    band_ceiling = cheapest * (1.0 + cost_tolerance)
    band = [c for c in pool if c.hourly_usd <= band_ceiling]
    outside = [c for c in pool if c.hourly_usd > band_ceiling]

    # Within-band ordering: if ANY band member has a hint, sort the
    # whole band by (hint presence, hint value, cost). Unhinted entries
    # come after hinted ones so a known-fast box always beats an
    # unknown-fast box at the same price tier.
    hinted_present = any(c.availability_hint is not None for c in band)
    if hinted_present:

        def _band_key(c: Candidate):
            # (hint_available, hint_value, cost) — tuples sort
            # left-to-right. Using `float('inf')` for missing hints puts
            # unhinted entries last within the band.
            hint = c.availability_hint if c.availability_hint is not None else float("inf")
            return (hint, c.hourly_usd)

        band.sort(key=_band_key)

    ordered = band + outside  # band wins over outside; outside stays cost-ascending

    choices = []
    for i, c in enumerate(ordered[:n]):
        if i == 0:
            reason = _describe_pick(c, band, cheapest, cost_tolerance, hinted_present)
        else:
            reason = f"fallback #{i} (ordered by tolerance-band first, then cheapest-outside-band)"
        choices.append(
            MachineChoice(name=c.name, hourly_usd=c.hourly_usd, region=c.region, reason=reason)
        )
    return choices


def _describe_pick(
    winner: Candidate,
    band: list,
    cheapest_price: float,
    cost_tolerance: float,
    hinted_present: bool,
) -> str:
    """Human-readable reason string for the top pick from pick_machines.
    Mirrors pick_machine's prose so logs stay consistent."""
    if hinted_present and winner is not band[0]:
        # This shouldn't happen — we put the hinted-best at band[0] above.
        # Defensive fallback so we never crash on a reason string.
        return f"within {int(cost_tolerance * 100)}% of cheapest ${cheapest_price:.3f}/hr"
    if hinted_present:
        if len(band) == 1:
            return "cheapest (only candidate in tolerance band)"
        if winner.availability_hint is None:
            return "cheapest; tolerance band had availability hints but we picked cost"
        return (
            f"lowest availability hint in the {int(cost_tolerance * 100)}% "
            f"tolerance band (hint={winner.availability_hint}, "
            f"cost=${winner.hourly_usd:.3f}/hr)"
        )
    if len(band) == 1:
        return "cheapest"
    return (
        f"cheapest; {len(band) - 1} other candidate(s) within "
        f"{int(cost_tolerance * 100)}% but none had an availability hint"
    )
