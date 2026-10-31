"""Plot calculated F1-score values."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("f1_score", "classification")


if __name__ == "__main__":
    main()
