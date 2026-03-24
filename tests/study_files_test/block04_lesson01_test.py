# =========================
# BLOCK 04 — LESSON 01 TESTS (НЕ МЕНЯТЬ)
# =========================

def approx(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps

def run_all_tests():
    s = [10, 11, 9, 10, 12, 9, 10, 11, 10, 9]

    assert len(s) == 10
    assert min(s) == 9
    assert max(s) == 12

    assert approx(mean(s), 10.1)
    assert median(s) == 10.0

    # mean=10.1, sum((x-m)^2)=8.9 => /9
    assert approx(round(variance_sample(s), 6), round(8.9/9, 6))
    assert approx(round(std_sample(s), 6), round((8.9/9) ** 0.5, 6))

    s2 = with_outlier(s, 100)
    assert len(s2) == 11
    assert approx(round(mean(s2), 6), round((sum(s)+100)/11, 6))
    assert median(s2) == 10.0  # медиана устойчива

    assert approx(trimmed_mean([1, 2, 3, 100], k=1), 2.5)
    assert trimmed_mean([1, 2, 3, 4, 100, 101], k=1) == mean([2, 3, 4, 100])

    d = describe(s)
    assert d["n"] == 10
    assert d["min"] == 9 and d["max"] == 12
    assert approx(d["mean"], 10.1)
    assert d["median"] == 10.0
    assert d["std"] is not None

    print("✅ BLOCK04 LESSON01: all tests passed")

run_all_tests()
