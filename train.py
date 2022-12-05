from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import time
import uuid

from joblib import dump, load
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm.auto import tqdm

from data_set import BiasBountyDataset, DEFAULT_TRANSFORM

N_OUTPUTS = 17
PROJECT_DIR = "/home/justin/projects/bias_bounty"
MODEL_DIR = f"{PROJECT_DIR}/models"
PARAMS = {
    "seed": 0,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "n_epochs": 100,
    "dropout": 0.6,
    "unfreeze": 2,
}


class TransferLearning:
    def __init__(
        self,
        params=PARAMS,
        load_model=None,
        algorithm="AdamW",
        pretrained="EfficientNet_V2_L",
        optimizer=None,
    ):
        super().__init__()
        self.batch_size = params["batch_size"]
        self.learning_rate = params["learning_rate"]
        self.n_epochs = params["n_epochs"]
        self.load_model = load_model
        self.pretrained = pretrained
        self.algorithm = algorithm
        self.best_state = None
        self.optimizer = optimizer
        self.unfreeze = params["unfreeze"]
        self.dropout = params["dropout"]
        self.seed = params["seed"]
        self.params = params
        self.model_id = str(uuid.uuid4())

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.model_name = f"{now}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.load_model:
            self.model = torch.load(self.load_model)
            self.model.to(self.device)
        else:
            model_class_name = self.pretrained.lower()
            model_class = getattr(models, model_class_name)
            weights_attr = getattr(models, f"{self.pretrained}_Weights")
            self.model = get_model(
                self.device,
                model_class(weights=weights_attr.DEFAULT),
                self.unfreeze,
                self.dropout,
            )

        # From the demo.ipynb
        self.skin_tone_weights = torch.Tensor(
            [
                3.30739,
                1.32605,
                1.00000,
                1.09607,
                1.32502,
                2.68987,
                2.44957,
                3.50515,
                5.88235,
                14.40678,
            ]
        ).to(self.device)
        self.age_weights = torch.Tensor([2.23526, 1.00000, 1.99859, 15.38768]).to(
            self.device
        )
        self.gender_weights = torch.Tensor([1.00000, 1.48128]).to(self.device)

    def checkpoint(self):
        return {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }

    def to_save(self):
        return {
            "best_state": self.best_state,
            "current_state": self.checkpoint(),
            "pretrained": self.pretrained,
            "algorithm": self.algorithm,
            "optimizer": self.optimizer,
            "model_name": self.model_name,
            "params": self.params,
        }

    @classmethod
    def load(cls, model_id):
        state = load(Path(MODEL_DIR) / f"{model_id}.pkl")["model"]
        instance = cls(
            params=state["params"],
            pretrained=state["pretrained"],
            algorithm=state["algorithm"],
            optimizer=state["optimizer"],
        )
        if state["best_state"] is None:
            instance.model.load_state_dict(state["current_state"]["model"])
            instance.optimizer.load_state_dict(state["current_state"]["optimizer"])
        else:
            instance.model.load_state_dict(state["best_state"]["model"])
            instance.optimizer.load_state_dict(state["best_state"]["optimizer"])
        return instance

    def log_to_tensorboard(self, phase, epoch, epoch_loss):
        logdir = str(Path(MODEL_DIR) / "tensorboard" / self.model_id)
        hparam_dict = {
            "lr": self.learning_rate,
            "bsize": self.batch_size,
            "optimizer": self.optimizer.__class__.__name__,
            "pre-trained": self.pretrained,
            "unfreeze": self.unfreeze,
            "dropout": self.dropout,
        }

        metric_dict = {
            "hparam/loss": self.best_loss,
        }

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            w_hp.add_hparams(hparam_dict, metric_dict, run_name=f"/{logdir}")

    def loss(self, preds, labels):
        skin_tone_loss = nn.functional.binary_cross_entropy_with_logits(
            preds[:, :10], labels[:, :10], pos_weight=self.skin_tone_weights
        )
        age_loss = nn.functional.binary_cross_entropy_with_logits(
            preds[:, 10:14], labels[:, 10:14], pos_weight=self.age_weights
        )
        gender_loss = nn.functional.binary_cross_entropy_with_logits(
            preds[:, 14:16], labels[:, 14:16], pos_weight=self.gender_weights
        )
        non_person_loss = nn.functional.binary_cross_entropy_with_logits(
            preds[:, 16:17], labels[:, 16:17]
        )
        return 10 * skin_tone_loss + 4 * age_loss + 2 * gender_loss + non_person_loss

    def fit(self, train, validate=None, trial=None):
        assert self.learning_rate is not None
        assert self.n_epochs is not None

        self.best_acc = 0.0
        self.best_loss = 99999999999
        self.last_loss = 0

        if self.optimizer is None:
            optim_class = getattr(optim, self.algorithm)
            self.optimizer = optim_class(self.model.parameters(), self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(train),
            epochs=self.n_epochs,
        )

        dataloaders = {"train": train}
        if validate is not None:
            dataloaders["val"]: validate

        for epoch in tqdm(range(self.n_epochs), desc="Epoch", leave=False):
            for phase in tqdm(dataloaders.keys(), desc="Phase", leave=False):
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                batch_losses = []

                for inputs, labels in tqdm(
                    dataloaders[phase], desc="Batch", leave=False
                ):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        loss = self.loss(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                    batch_loss = loss.item()
                    batch_losses.append(batch_loss)

                epoch_loss = np.array(batch_losses).mean()

                if phase == "val":
                    self.last_loss = epoch_loss
                    if trial:
                        trial.report(epoch_loss, epoch)
                        if trial.should_prune():
                            self.log_to_tensorboard(phase, epoch, epoch_loss)
                            raise optuna.TrialPruned()
                    if epoch_loss < self.best_loss:
                        self.best_loss = epoch_loss
                        self.best_state = self.checkpoint()
                        self.write_model_to_disk(epoch)

                self.log_to_tensorboard(phase, epoch, epoch_loss)
                if epoch % 10 == 0:
                    self.write_model_to_disk(epoch, suffix=epoch)

        return self

    def write_model_to_disk(self, epoch=None, suffix=None):
        if epoch:
            tqdm.write(f"Saving model {self.model_id} at epoch {epoch}")
        else:
            tqdm.write(f"Saving model {self.model_id}")

        model_path = Path(MODEL_DIR) / f"{self.model_id}.pkl"
        if suffix:
            model_path = Path(MODEL_DIR) / f"{self.model_id}_{suffix}.pkl"

        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            dump(
                {
                    "model": self.to_save() if hasattr(self, "to_save") else self,
                    "params": self.params,
                },
                f,
            )
        return model_path

    def predict(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        all_outputs = []

        for inputs, _ in tqdm(dataloader, desc="Batch"):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                all_outputs.append(self.model(inputs))

        results = torch.cat(all_outputs, dim=0)
        probs_for_skin_tone = (
            torch.nn.functional.softmax(results[:, :10], dim=1).cpu().numpy()
        )
        probs_for_age = (
            torch.nn.functional.softmax(results[:, 10:14], dim=1).cpu().numpy()
        )
        probs_for_gender = (
            torch.nn.functional.softmax(results[:, 14:16], dim=1).cpu().numpy()
        )
        probs_for_object = torch.sigmoid(results[:, 16:17]).cpu().numpy()
        if len(dataset) != len(probs_for_skin_tone):
            raise ValueError(
                f"Number of results ({len(results)}) "
                f"does not match number of inputs ({len(dataset)})"
            )

        probs = {
            "skin_tone": probs_for_skin_tone,
            "age": probs_for_age,
            "gender": probs_for_gender,
            "object": probs_for_object,
        }
        return probs


def get_model(device, model, unfreeze=0, dropout=0.0):
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last Conv2dNormActivation
    if unfreeze > 0:
        for param in model.features[-1].parameters():
            param.requires_grad = True

    for i in range(unfreeze - 1):
        for param in model.features[-2][-i].parameters():
            param.requires_grad = True

    n_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout, inplace=True), nn.Linear(n_features, N_OUTPUTS)
    )
    return model.to(device)


def train(train, validate=None, params=PARAMS, trial=None):
    torch.manual_seed(params["seed"])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )
    if validate:
        valid_loader = torch.utils.data.DataLoader(
            validate,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=10,
            pin_memory=True,
        )
    else:
        valid_loader = None

    model = TransferLearning(params)
    return model.fit(train_loader, valid_loader, trial=trial)


def parse_args():
    parser = ArgumentParser(description="Train a model")
    parser.add_argument("input_path", type=Path, help="Path to data directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    seed = 0

    jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5)
    hflipper = transforms.RandomHorizontalFlip(p=0.5)
    persp = transforms.RandomPerspective(distortion_scale=0.3, p=0.4)
    rotate = transforms.RandomRotation(degrees=45)
    transform = transforms.Compose([jitter, hflipper, persp, rotate, DEFAULT_TRANSFORM])

    # data = InMemoryBiasBountyDataset(
    #    args.input_path, transform=transform, always_transform=True
    # )
    all_data = BiasBountyDataset(args.input_path)
    train_split, validate_split = random_split(
        all_data, [0.8, 0.2], torch.Generator().manual_seed(seed)
    )
    train_data = BiasBountyDataset(
        args.input_path, transform=transform, indices=train_split.indices
    )
    validate_data = BiasBountyDataset(
        args.input_path, transform=DEFAULT_TRANSFORM, indices=validate_split.indices
    )

    def objective(trial):
        params = {
            "seed": seed,
            "batch_size": 100,
            "learning_rate": 2.33e-05,  # trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            "n_epochs": 100,
            "dropout": 0.66,  # trial.suggest_float("dropout", 0.5, 0.8),
            "unfreeze": 5,
        }
        # model = train(train_data, validate_data, params, trial=trial)
        model = train(all_data, params=params, trial=trial)
        trial.set_user_attr("model_id", model.model_id)
        return model.last_loss

    study = optuna.create_study(
        study_name="bias_bounty_study",
        storage=f"sqlite:///{PROJECT_DIR}/studies_7.db",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=75),
    )
    study.optimize(objective, n_trials=100, catch=(torch.cuda.OutOfMemoryError,))  # type: ignore
