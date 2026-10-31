"""Plot calculated recall values."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("recall", "classification")


if __name__ == "__main__":
    main()
