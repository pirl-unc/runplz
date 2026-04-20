"""Coverage for the cross-backend machine-type selector.

See runplz/_selector.py for the algorithm. Exercises the 5% cost
tolerance + availability-hint tiebreaker plus the fallback paths.
"""

from runplz._selector import Candidate, pick_machine


def _c(name, price, hint=None, region=None):
    return Candidate(name=name, hourly_usd=price, availability_hint=hint, region=region)


def test_empty_candidates_returns_none():
    assert pick_machine([]) is None


def test_no_priced_candidates_returns_none():
    # A candidate with price=None / 0 isn't usable.
    assert pick_machine([_c("x", None), _c("y", 0)]) is None


def test_single_candidate_is_chosen():
    choice = pick_machine([_c("only-one", 0.5)])
    assert choice is not None
    assert choice.name == "only-one"
    assert choice.reason == "cheapest"


def test_cheapest_wins_when_no_hints():
    cands = [_c("mid", 1.0), _c("cheap", 0.5), _c("expensive", 5.0)]
    choice = pick_machine(cands)
    assert choice.name == "cheap"
    assert "cheapest" in choice.reason


def test_within_5pct_plus_availability_hint_picks_faster():
    """$1.00 vs $1.04 is within 5%. The $1.04 one has eta=30s vs eta=300s.
    Selector should prefer the faster-to-start one."""
    cands = [
        _c("slow-cheap", 1.00, hint=300.0),
        _c("fast-slightly-more", 1.04, hint=30.0),
    ]
    choice = pick_machine(cands)
    assert choice.name == "fast-slightly-more"
    assert "lower availability hint" in choice.reason


def test_outside_5pct_cost_always_wins_even_with_hint():
    """$1.00 vs $1.10 is 10%, past the 5% tolerance. Cheapest must win
    regardless of availability."""
    cands = [
        _c("slow-cheap", 1.00, hint=500.0),
        _c("fast-expensive", 1.10, hint=1.0),
    ]
    choice = pick_machine(cands)
    assert choice.name == "slow-cheap"


def test_tie_on_hint_breaks_by_cost():
    """Within band, equal hints → prefer cheaper."""
    cands = [
        _c("a", 1.02, hint=100.0),
        _c("b", 1.00, hint=100.0),
    ]
    choice = pick_machine(cands)
    assert choice.name == "b"


def test_mixed_some_hinted_some_not_still_prefers_hinted_in_band():
    """The band contains a hinted and a non-hinted candidate at the same
    price — the hinted one should win because the selector can compare
    it, and silently dropping it because others lack hints would
    defeat the point of having them."""
    cands = [
        _c("nohint-cheap", 1.00),
        _c("hinted-tiny-bit-more", 1.02, hint=10.0),
    ]
    choice = pick_machine(cands)
    assert choice.name == "hinted-tiny-bit-more"


def test_tolerance_boundary_is_inclusive():
    """Exactly +5% counts as in-band."""
    cands = [
        _c("cheapest", 1.00, hint=500.0),
        _c("boundary", 1.05, hint=10.0),
    ]
    choice = pick_machine(cands)
    assert choice.name == "boundary"


def test_custom_tolerance_respected():
    # Shrink tolerance so the $1.04 candidate is now outside the band.
    cands = [
        _c("cheapest", 1.00, hint=500.0),
        _c("slight", 1.04, hint=10.0),
    ]
    choice = pick_machine(cands, cost_tolerance=0.01)
    assert choice.name == "cheapest"


def test_reason_mentions_tolerance_in_band_with_no_hints():
    cands = [_c("a", 1.00), _c("b", 1.03), _c("c", 1.04)]
    choice = pick_machine(cands)
    # All three are within 5%, none has a hint, cheapest wins.
    assert choice.name == "a"
    assert "2 other" in choice.reason
    assert "availability hint" in choice.reason
