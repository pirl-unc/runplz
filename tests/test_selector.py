"""Coverage for the cross-backend machine-type selector.

See runplz/_selector.py for the algorithm. Exercises the 5% cost
tolerance + availability-hint tiebreaker plus the fallback paths.
"""

import pytest

from runplz._selector import Candidate, pick_machine, pick_machines


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


# --- pick_machines: multi-type fallback (#44) ------------------------


def test_pick_machines_empty_returns_empty_list():
    assert pick_machines([], n=3) == []


def test_pick_machines_zero_n_raises():
    with pytest.raises(ValueError, match="positive int"):
        pick_machines([_c("x", 1.0)], n=0)


def test_pick_machines_returns_up_to_n():
    cands = [_c("a", 1.00), _c("b", 1.20), _c("c", 1.50), _c("d", 2.00)]
    got = [c.name for c in pick_machines(cands, n=2)]
    assert got == ["a", "b"]


def test_pick_machines_returns_fewer_than_n_when_pool_is_small():
    got = pick_machines([_c("only", 0.5)], n=5)
    assert [c.name for c in got] == ["only"]


def test_pick_machines_within_band_orders_by_hint_then_cost():
    """Within the 5% band, availability-hinted candidates beat unhinted
    ones; ties broken by cost. Outside the band, cheapest-first."""
    cands = [
        _c("cheap-slow", 1.00, hint=500),
        _c("mid-fast", 1.04, hint=10),
        _c("nohint", 1.02),
        _c("expensive", 1.50, hint=1),  # outside 5% band → ranks AFTER band
    ]
    names = [c.name for c in pick_machines(cands, n=4)]
    # In-band: fast (hint=10) > slow (hint=500) > nohint (inf).
    # Out-of-band: expensive comes last even though its hint is best.
    assert names == ["mid-fast", "cheap-slow", "nohint", "expensive"]


def test_pick_machines_fallback_goes_outside_band_cheapest_first():
    """The whole point for #44: top pick is within-band; fallbacks 2..N
    come from OUTSIDE the band, cheapest-first, because we want
    resilience (try a 10%-pricier option if the cheap one fails) over
    strict cost-optimality."""
    cands = [
        _c("A-cheap-in-band", 1.00),
        _c("B-also-in-band", 1.03),
        _c("C-10pct-over", 1.10),
        _c("D-20pct-over", 1.20),
    ]
    got = [c.name for c in pick_machines(cands, n=3)]
    # Band: A, B. Outside: C, D. Top 3: A, B, C.
    assert got == ["A-cheap-in-band", "B-also-in-band", "C-10pct-over"]


def test_pick_machines_skips_candidates_without_price():
    cands = [_c("no-price", None), _c("cheap", 0.5), _c("zero", 0)]
    assert [c.name for c in pick_machines(cands, n=3)] == ["cheap"]


def test_pick_machines_top_pick_reason_matches_pick_machine():
    """Behavioral regression guard: the #1 result from pick_machines
    should reflect the same logic as pick_machine (so logs stay
    consistent across the single- vs multi-pick code paths)."""
    cands = [_c("a", 1.0, hint=100), _c("b", 1.03, hint=5)]
    single = pick_machine(cands)
    top = pick_machines(cands, n=2)[0]
    assert top.name == single.name
    assert "lower availability hint" in top.reason or "lowest availability" in top.reason


def test_pick_machines_fallback_entries_have_a_reason_each():
    cands = [_c("a", 1.0), _c("b", 1.2), _c("c", 1.5)]
    got = pick_machines(cands, n=3)
    assert len(got) == 3
    assert got[0].reason  # top-pick reason
    assert "fallback" in got[1].reason
    assert "fallback" in got[2].reason
