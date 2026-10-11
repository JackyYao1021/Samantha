"""Deterministic statistics with episodes as independent sampling units."""

from collections import defaultdict
import math
import random
import statistics as std_statistics


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Metrics must be finite numbers")
    return float(value)


def percentile(values, quantile):
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be in [0, 1]")
    if not values:
        return None
    ordered = sorted(_finite(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_mean_ci(values, *, seed=17, samples=2000, confidence=.95):
    values = [_finite(value) for value in values]
    if type(samples) is not int or samples < 1 or type(seed) is not int or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0, "seed": seed, "samples": samples,
                "confidence": confidence, "method": "percentile_bootstrap"}
    generator = random.Random(seed)
    means = [sum(generator.choice(values) for _ in values) / len(values) for _ in range(samples)]
    tail = (1 - confidence) / 2
    return {"mean": std_statistics.mean(values), "low": percentile(means, tail), "high": percentile(means, 1 - tail),
            "n": len(values), "seed": seed, "samples": samples, "confidence": confidence, "method": "percentile_bootstrap"}


def wilson_interval(successes, total, *, confidence=.95):
    if (type(successes) is not int or type(total) is not int or not 0 <= successes <= total or not 0 < confidence < 1):
        raise ValueError("Invalid binomial counts")
    if not total:
        return {"low": None, "high": None, "confidence": confidence, "method": "wilson", "n": 0}
    z = std_statistics.NormalDist().inv_cdf((1 + confidence) / 2)
    rate = successes / total
    divisor = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / divisor
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / divisor
    return {"low": max(0., center - radius), "high": min(1., center + radius), "confidence": confidence, "method": "wilson", "n": total}


def metric_value(result, name):
    matches = [score.get("value") for score in result.get("scores", []) if score.get("metric") == name]
    if len(matches) != 1:
        raise ValueError("Each saved result needs exactly one requested metric: " + name)
    return _finite(matches[0])


def episode_rates(results, metric="task_success"):
    grouped, attempts = defaultdict(list), set()
    for index, result in enumerate(results):
        key = result.get("episode_id", result.get("case_id"))
        if not isinstance(key, str) or not key:
            raise ValueError("Saved results need an episode identity")
        attempt = (key, result.get("attempt_id", index))
        if attempt in attempts:
            raise ValueError("Repeated results contain duplicate attempts")
        attempts.add(attempt)
        grouped[key].append(metric_value(result, metric))
    return {key: {"p_i": std_statistics.mean(values), "started_repeats": len(values),
                  "repeat_variance": std_statistics.variance(values) if len(values) > 1 else None,
                  "binomial_variance": std_statistics.mean(values) * (1 - std_statistics.mean(values))
                    if all(value in {0, 1} for value in values) else None}
            for key, values in sorted(grouped.items())}


def paired_episode_comparison(left_results, right_results, *, metric="task_success", seed=17, bootstrap_samples=2000, confidence=.95):
    left, right = episode_rates(left_results, metric), episode_rates(right_results, metric)
    paired = sorted(left.keys() & right.keys())
    differences = [right[key]["p_i"] - left[key]["p_i"] for key in paired]
    interval = bootstrap_mean_ci(differences, seed=seed, samples=bootstrap_samples, confidence=confidence)
    interval["method"] = "paired_episode_percentile_bootstrap"
    return {"schema_version": "1.0", "metric": metric, "direction": "right_minus_left", "paired_episodes": len(paired),
            "left_only_episodes": sorted(left.keys() - right.keys()), "right_only_episodes": sorted(right.keys() - left.keys()),
            "left_episode_mean": std_statistics.mean(left[key]["p_i"] for key in paired) if paired else None,
            "right_episode_mean": std_statistics.mean(right[key]["p_i"] for key in paired) if paired else None,
            "difference": interval, "wins": sum(value > 0 for value in differences), "losses": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences), "episode_variance": std_statistics.variance(differences) if len(differences) > 1 else None,
            "episodes": [{"episode_id": key, "left": left[key], "right": right[key], "difference": difference}
                         for key, difference in zip(paired, differences)],
            "exploratory": len(paired) < 50, "sampling_unit": "episode; repeats averaged within episode",
            "limitations": ["Paired descriptive bootstrap; repeated attempts are not independent samples.",
                            "Scripted offline results cannot establish model improvement.",
                            "The episode population and eligibility rules determine what this interval describes."]}
