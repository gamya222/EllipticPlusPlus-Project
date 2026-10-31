"""Plot calculated RMSE values from evaluation_metrics/results."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("rmse", "regression")


if __name__ == "__main__":
    main()
