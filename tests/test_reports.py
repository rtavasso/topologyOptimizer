from tad.evaluation.reports import _metrics_table, _skill_vs_strongest


def test_skill_vs_strongest_uses_hardest_baseline():
    metrics = {
        "skill_vs_tuned_ema": 0.5,
        "skill_vs_persistence": 0.2,
        "skill_vs_vector_autoregression": -0.1,
    }

    assert _skill_vs_strongest(metrics) == -0.1


def test_metrics_table_includes_strongest_baseline_column():
    table = _metrics_table({
        "layer_gru": {
            "future_gradient": {
                "1": {
                    "nmse": 1.0,
                    "cosine": 0.0,
                    "r2": -1.0,
                    "skill_vs_tuned_ema": 0.5,
                    "skill_vs_persistence": -0.25,
                }
            }
        }
    })

    assert "skill vs strongest" in table
    assert "-0.250" in table
