"""Plot signed mean error with a zero reference line."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("bias", "regression")


if __name__ == "__main__":
    main()
