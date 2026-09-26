import torch


@torch.no_grad()
def predict_model(
    model,
    dataloader,
    device,
):
    """
    Generate probabilities and predictions
    for an entire dataloader.

    Returns:
        probabilities: [N, 5]
        predictions:   [N]
        labels:        [N]
        image_ids:     list
    """

    model.eval()

    all_probabilities = []
    all_predictions = []
    all_labels = []
    all_image_ids = []

    for batch in dataloader:

        images = batch["image"].to(
            device
        )

        labels = batch["label"]

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_probabilities.append(
            probabilities.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        all_image_ids.extend(
            batch["image_id"]
        )

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    predictions = torch.cat(
        all_predictions
    ).numpy()

    labels = torch.cat(
        all_labels
    ).numpy()

    return (
        probabilities,
        predictions,
        labels,
        all_image_ids,
    )