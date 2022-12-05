from argparse import ArgumentParser
from itertools import cycle
from pathlib import Path

from joblib import dump
import numpy as np
import pandas as pd

from data_set import BiasBountyDataset
from train import TransferLearning

PRED_DIR = "/home/justin/projects/bias_bounty/predictions"


def predict(model_id, dataset):
    model = TransferLearning.load(model_id)
    return model.predict(dataset)


def get_random_labels(n):
    s, a, g = (
        cycle(BiasBountyDataset.SKIN_TONES),
        cycle(BiasBountyDataset.AGES),
        cycle(BiasBountyDataset.GENDERS),
    )
    labels = []
    for _ in range(n):
        labels.append((next(s), next(a), next(g)))
    return labels


def probs_to_text_labels(probs, dataset):
    skin_tone_preds = (
        probs["skin_tone"].max(axis=1).reshape(-1, 1).repeat(10, axis=1)
        == probs["skin_tone"]
    ).astype("int")
    age_preds = (
        probs["age"].max(axis=1).reshape(-1, 1).repeat(4, axis=1) == probs["age"]
    ).astype("int")
    gender_preds = (
        probs["gender"].max(axis=1).reshape(-1, 1).repeat(2, axis=1) == probs["gender"]
    ).astype("int")
    labels = dataset.label_encoder.inverse_transform(
        np.concatenate((skin_tone_preds, age_preds, gender_preds), axis=1)
    ).tolist()
    # The classes separate a ton, so this is fine and doesn't really require tuning.
    object_prob_threshold = 0.5
    n_objects = (probs["object"] > object_prob_threshold).sum()
    print(f"Predicted {n_objects} objects")
    random_labels = iter(get_random_labels(n_objects))
    random_label_indices = []
    for i, prob in enumerate(probs["object"].squeeze()):
        if prob > object_prob_threshold:
            labels[i] = next(random_labels)
            random_label_indices.append(i)
    return labels, random_label_indices


def parse_args():
    parser = ArgumentParser(description="Make predictions")
    parser.add_argument("model_id", type=str, help="ID of the model")
    parser.add_argument("data_path", type=Path, help="Path to data directory")
    parser.add_argument("--suffix", default="", help="Suffix for predictions file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = BiasBountyDataset(args.data_path, is_train=False)
    predictions = predict(args.model_id, dataset)
    pred_path = Path(PRED_DIR) / f"{args.model_id}{args.suffix}_probs.pkl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    dump(predictions, pred_path)
    print(f"Saved predictions to {pred_path}")
    labels = probs_to_text_labels(predictions, dataset)
    output = pd.DataFrame(
        labels[0],
        index=dataset.labels_df["name"],
        columns=["skin_tone", "age", "gender"],
    )
    output.to_csv(Path(PRED_DIR) / f"{args.model_id}{args.suffix}_labels.csv")
    pred_path = Path(PRED_DIR) / f"{args.model_id}{args.suffix}_labels.pkl"
    dump(labels, pred_path)
    print(f"Saved labels to {pred_path}")
