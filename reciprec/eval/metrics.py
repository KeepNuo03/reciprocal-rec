"""Metrics for ranked lists and candidate exposure."""


def exposure_gini(histogram):
    population = sum(histogram.values())
    impressions = sum(exposure * count for exposure, count in histogram.items())
    if not impressions:
        return 0.0
    weighted_ranks = 0
    preceding = 0
    for exposure, count in sorted(histogram.items()):
        weighted_ranks += exposure * count * (2 * preceding + count + 1)
        preceding += count
    return (weighted_ranks - (population + 1) * impressions) / (population * impressions)
