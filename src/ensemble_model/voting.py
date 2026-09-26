import numpy as np


def hard_voting(predictions):
    """
    Majority voting.

    predictions:
        shape = [num_models, num_samples]
    """

    num_models, num_samples = (
        predictions.shape
    )

    ensemble_predictions = []

    for i in range(num_samples):

        votes = predictions[:, i]

        counts = np.bincount(
            votes,
            minlength=5,
        )

        ensemble_predictions.append(
            np.argmax(counts)
        )

    return np.array(
        ensemble_predictions
    )


def soft_voting(probabilities):
    """
    Equal-weight probability averaging.

    probabilities:
        shape = [num_models, num_samples, 5]
    """

    mean_probabilities = (
        np.mean(
            probabilities,
            axis=0,
        )
    )

    predictions = np.argmax(
        mean_probabilities,
        axis=1,
    )

    return (
        mean_probabilities,
        predictions,
    )


def weighted_soft_voting(
    probabilities,
    weights,
):
    """
    Weighted probability averaging.

    probabilities:
        [num_models, num_samples, 5]

    weights:
        [num_models]
    """

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    weights = (
        weights
        / weights.sum()
    )

    weighted_probabilities = np.sum(
        probabilities
        * weights[:, None, None],
        axis=0,
    )

    predictions = np.argmax(
        weighted_probabilities,
        axis=1,
    )

    return (
        weighted_probabilities,
        predictions,
    )