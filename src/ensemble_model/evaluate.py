from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)


def evaluate_predictions(
    labels,
    predictions,
):
    """
    Evaluate multi-class DR predictions.
    """

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    qwk = cohen_kappa_score(
        labels,
        predictions,
        weights="quadratic",
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "qwk": qwk,
    }