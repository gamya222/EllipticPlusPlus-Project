"""Plot calculated binary log-loss values."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("log_loss", "classification")


if __name__ == "__main__":
    main()
