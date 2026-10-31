"""Plot calculated Cohen's kappa values."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("cohens_kappa", "classification")


if __name__ == "__main__":
    main()
