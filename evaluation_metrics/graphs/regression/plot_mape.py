"""Plot calculated MAPE values; no graph is created when values are undefined."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("mape", "regression")


if __name__ == "__main__":
    main()
