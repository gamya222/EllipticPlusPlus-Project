"""Plot calculated R-squared values, retaining negative values."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("r2", "regression")


if __name__ == "__main__":
    main()
