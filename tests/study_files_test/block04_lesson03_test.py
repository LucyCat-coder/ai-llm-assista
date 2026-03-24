# =========================
# BLOCK 04 — LESSON 03 TESTS (НЕ МЕНЯТЬ)
# =========================

def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps

def run_all_tests():
    recs = [
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 0, "bought": 0},
    {"clicked": 0, "bought": 0},
    {"clicked": 1, "bought": 1},
    {"clicked": 1, "bought": 0},
    {"clicked": 1, "bought": 0},
]
    # counts expected
    n = len(recs)
    c_click = sum(1 for r in recs if r["clicked"] == 1)
    c_buy = sum(1 for r in recs if r["bought"] == 1)
    c_both = sum(1 for r in recs if r["clicked"] == 1 and r["bought"] == 1)

    assert n == 12
    assert c_click == 9
    assert c_buy == 4
    assert c_both == 4

    assert approx(prob_event(c_click, n), 9/12)
    assert approx(prob_event(c_buy, n), 4/12)
    assert approx(prob_conditional(c_both, c_click), 4/9)

    assert is_independent_by_counts(4/12, 4/9, tol=0.05) is False  # далеко

    t = contingency_2x2(recs, "clicked", "bought")
    # clicked=0: bought=0 -> 3, bought=1 -> 0
    # clicked=1: bought=0 -> 5, bought=1 -> 4
    assert t == [[3, 0], [5, 4]]

    # simulation sanity
    clicked_sim, bought_sim = simulate_click_buy(50_000, 0.6, 0.05, 0.25, seed=123)
    # estimate P(buy|click=1) should be near 0.25
    count_click1 = int(clicked_sim.sum())
    est = float((bought_sim & clicked_sim).sum()) / count_click1
    assert abs(est - 0.25) < 0.02

    print("✅ BLOCK04 LESSON03: all tests passed")

run_all_tests()
