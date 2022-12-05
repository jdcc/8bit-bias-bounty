from argparse import ArgumentParser
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision import transforms
from tqdm.auto import tqdm


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class BiasBountyDataset(torch.utils.data.Dataset):
    SKIN_TONES = [f"monk_{i}" for i in range(1, 11)]
    AGES = ["0_17", "18_30", "31_60", "61_100"]
    GENDERS = ["female", "male"]

    def __init__(
        self, images_path, is_train=True, transform=DEFAULT_TRANSFORM, indices=None
    ):
        self.images_path = Path(images_path)
        self.is_train = is_train
        self.transform = transform
        self.labels_df = pd.read_csv(self.images_path / "labels.csv", header=0)
        self.indices = indices
        self.n_images = len(indices) if indices is not None else self.labels_df.shape[0]
        self.label_encoder = OneHotEncoder(
            categories=[self.SKIN_TONES, self.AGES, self.GENDERS]
        )

    def __len__(self):
        "Denotes the total number of samples"
        return self.n_images

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        return self.get_image(index), self.get_label(index)

    def get_label_words(self, index, columns):
        return self.labels_df.loc[index, columns].values

    def get_label(self, index):
        if self.is_train:
            return self.get_train_label(index)
        return self.get_test_label(index)

    def get_test_label(self, index):
        labels = self.get_label_words(index, ["skin_tone", "age", "gender"])
        return self.label_encoder.fit_transform([labels]).toarray()[0]

    def get_train_label(self, index):
        labels = self.get_label_words(
            index, ["skin_tone", "age", "gender", "has_person"]
        )
        if labels[-1]:  # Image has a person in it
            output = np.append(
                self.label_encoder.fit_transform([labels[:-1]]).toarray()[0], 0
            )
        else:  # Image does not have a person in it
            output = np.zeros(17)
            output[-1] = 1.0
        return output

    def get_image(self, index, do_transform=True):
        filename = self.images_path / self.labels_df.loc[index, "name"]

        image = Image.open(str(filename))
        if self.transform and do_transform:
            image = self.transform(image)
        return image


class InMemoryBiasBountyDataset(BiasBountyDataset):
    def __init__(self, *args, always_transform=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.always_transform = always_transform
        self.dataset = [
            (self.get_image(i, not always_transform), self.get_label(i))
            for i in tqdm(
                range(len(self)), desc="Loading images", leave=True, smoothing=0
            )
        ]

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.always_transform:
            image = self.transform(image)
        return image, label


def preprocess_dataset(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    csv = pd.read_csv(input_path / "labels.csv", header=0, index_col=None)
    # Remove rows with missing labels.
    csv = csv.loc[~(csv.skin_tone.isna() | csv.age.isna() | csv.gender.isna())]
    csv["n_faces"] = 0
    csv["yolo_result"] = ""

    model = torch.hub.load("ultralytics/yolov5", "yolov5x")

    to_drop = []
    for i, row in tqdm(csv.iterrows(), smoothing=0, desc="Images", total=csv.shape[0]):
        image = Image.open(str(input_path / row["name"]))
        if image.mode != "RGB":
            to_drop.append(row["name"])
            continue

        # face_locations = face_recognition.face_locations(np.array(image))
        # csv.loc[csv["name"] == row["name"], "n_faces"] = len(face_locations)
        csv.loc[csv["name"] == row["name"], "yolo_result"] = preds_to_str(model(image))

        shutil.copy(str(input_path / row["name"]), str(output_path / row["name"]))
    csv.loc[~csv["name"].isin(set(to_drop))].to_csv(
        output_path / "labels.csv", index=False
    )


def preds_to_str(preds):
    outputs = []
    for pred in preds.pred:
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                outputs.append(f"{n} {preds.names[int(c)]}{'s' * (n > 1)}")
    return ", ".join(outputs)


def parse_args():
    parser = ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("input_path", type=Path, help="Path to data directory.")
    parser.add_argument(
        "output_path", type=Path, help="Path to directory for processed data."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(args.input_path, args.output_path)
