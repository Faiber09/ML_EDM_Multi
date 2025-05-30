import argparse


def parse_args():
    """Parse command line arguments.

    This function uses argparse to handle command line arguments for train.py script
    It includes options for dataset name, classifier type, random seed, number of splits,
    sampling ratio, stratification, correlated instances, hyperparameters, and fold partitioning.
    The arguments are parsed and returned as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Train time series classifiers")
    parser.add_argument(
        "--dataset", type=str, default="SelfRegulationSCP1", help="Name of the dataset"
    )
    parser.add_argument(
        "--classifier", type=str, default="MiniRocketClassifier", help="Name of the classifier"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=int, default=3, help="Number of cross-validation splits")
    parser.add_argument(
        "--sampling-ratio", type=float, default=0.05, help="Sampling ratio for timestamps"
    )
    parser.add_argument(
        "--no-stratify",
        dest="stratify",
        action="store_false",
        help="Disable stratification in folds",
    )
    parser.add_argument(
        "--corr-instances",
        action="store_true",
        default=False,
        help="Whether to use correlated instances",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="JSON string with classifier hyperparameters (e.g. '{\"n_kernels\": 1024}')",
    )
    # Add fold partitioning parameters
    parser.add_argument(
        "--window-length-pct",
        type=float,
        default=0.5,
        help="Window length percentage for time series partitioning",
    )
    parser.add_argument(
        "--step-length-pct",
        type=float,
        default=0.2,
        help="Step length percentage for time series partitioning",
    )
    parser.add_argument(
        "--fh-pct",
        type=float,
        default=0.2,
        help="Forecasting horizon percentage for time series partitioning",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable shuffling when creating folds",
    )
    args = parser.parse_args()

    # Parse hyperparameters if provided
    if args.hyperparams:
        import json

        try:
            args.hyperparams = json.loads(args.hyperparams)
        except json.JSONDecodeError:
            print("Error: Invalid JSON for hyperparameters")
            args.hyperparams = None

    return args
