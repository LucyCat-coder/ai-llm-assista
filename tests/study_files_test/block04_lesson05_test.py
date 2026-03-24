def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps

def run_all_tests():
    import numpy as np
    rng = np.random.default_rng(42)
    d = rng.normal(10.0, 2.0, 50)

    m = mean(d)
    s = std_sample(d)
    assert 8.0 < m < 12.0
    assert s > 0.0

    low, high = ci_mean_normal_approx(d)
    assert low < high
    assert low < m < high  # обычно так (синтетика)

    ci_b = bootstrap_ci_mean(d, n_boot=1000, seed=1)
    assert ci_b[0] < ci_b[1]
    assert ci_b[0] < m < ci_b[1]

    rng2 = np.random.default_rng(7)
    d20 = rng2.normal(10.0, 2.0, 20)
    d200 = rng2.normal(10.0, 2.0, 200)
    ci20 = bootstrap_ci_mean(d20, n_boot=800, seed=2)
    ci200 = bootstrap_ci_mean(d200, n_boot=800, seed=2)
    w20 = ci20[1] - ci20[0]
    w200 = ci200[1] - ci200[0]
    assert w20 > 0 and w200 > 0
    assert w200 < w20  # чаще всего так

    print("✅ BLOCK04 LESSON05: all tests passed")

run_all_tests()
