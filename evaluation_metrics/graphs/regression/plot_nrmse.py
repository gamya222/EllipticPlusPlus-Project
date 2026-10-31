"""Plot calculated range-normalized RMSE values."""

from evaluation_metrics.graphs._plotting import generate_metric_plot


def main():
    return generate_metric_plot("nrmse", "regression")


if __name__ == "__main__":
    main()
