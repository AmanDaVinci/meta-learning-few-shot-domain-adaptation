import argparse
import os
import time
import torch
from torch import optim
from typing import Iterable

from meta_infomax.models.classifier import SentimentClassifier
from project1.SNLI_data import get_data_iterators
from project1.encoder import *

MODELS_PATH = "models"
LOGS_PATH = "logs"


class Trainer:
    def __init__(self,
                 classifier: SentimentClassifier,
                 experiment_name: str = "",
                 device: torch.device = "cpu"):
        self.classifier = classifier.to(device)
        self.experiment_name = experiment_name
        self.device = device
        self.optimizer_encoder = optim.AdamW(self.classifier.encoder.parameters())
        self.optimizer_head = optim.Adam(self.classifier.head.parameters())
        self.freeze_encoder_layers((10, 11))

    def freeze_encoder_layers(self, train_layers: Tuple = (10, 11)):
        for name, param in self.classifier.encoder.named_parameters():
            if not name.startswith(f"pooler"):
                for layer_index in train_layers:
                    if not name.startswith(f"encoder.layer.{layer_index}"):
                        param.requires_grad = False

    def train(self, data_train: Iterable, data_valid: Iterable, max_epochs: int):
        print(f"Training begins...")
        for epoch in range(max_epochs):
            self.classifier.train()

            epoch_time = time.time()
            running_acc, running_loss, num_samples = 0, 0, 0

            for i, batch in enumerate(data_train):
                (text, text_lengths), label = batch.text, batch.label

                # Ignore the bigger chunks...  for  now
                if (text_lengths > 512).any():
                    continue

                self.optimizer_encoder.zero_grad()
                self.optimizer_head.zero_grad()

                output = self.classifier(text, label)
                loss = output['loss']
                acc = output['acc']
                loss.backward()

                self.optimizer_encoder.step()
                self.optimizer_head.step()

                bs = len(batch)
                running_acc += acc * bs
                running_loss += loss.item() * bs
                num_samples += bs

            loss_train, acc_train = running_loss / num_samples, running_acc / num_samples
            loss_val, acc_val = self.evaluate(data_valid)

            print(f"Epoch[{epoch + 1}/{max_epochs}]({(time.time() - epoch_time) / 60:.1f} mins)\t"
                  f"Train Loss:{loss_train:.3f}\t"
                  f"Valid Loss:{loss_val:.3f}\t"
                  f"Train Acc:{acc_train * 100:.2f}\t"
                  f'Valid Acc:{acc_val * 100:.2f}{" (New Best!)" if acc_val > self.scheduler_1.best else ""}\t', flush=True)

            self.save_checkpoint(self.experiment_name)

    def evaluate(self, dataloader: Iterable) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            running_acc, running_loss, num_samples = 0, 0, 0
            for batch in dataloader:
                (text, text_lengths), label = batch.text, batch.label
                output = self.classifier(text, label)
                loss = output['loss']
                acc = output['acc']

                bs = len(batch)
                running_acc += acc * bs
                running_loss += loss.item() * bs
                num_samples += bs

        return running_loss / num_samples, running_acc / num_samples

    # TODO: rewrite according to the current model
    def save_checkpoint(self, experiment_name: str):
        state = {
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_1': self.scheduler_1.state_dict(),
            'scheduler_2': self.scheduler_2.state_dict()
        }
        torch.save(state, self.get_checkpoint_path(experiment_name))

    # TODO: rewrite according to the current model
    def load_checkpoint(self, experiment_name: str = "", file_path: str = None):
        try:
            if not file_path:
                file_path = self.get_checkpoint_path(experiment_name)
            checkpoint = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler_1.load_state_dict(checkpoint['scheduler_1'])
            self.scheduler_2.load_state_dict(checkpoint['scheduler_2'])
            print(f"Checkpoint loaded")
        except OSError:
            print("No checkpoint found")

    def get_checkpoint_path(self, experiment_name: str):
        os.makedirs(MODELS_PATH, exist_ok=True)
        return f"{MODELS_PATH}/{experiment_name}.pt"


# TODO: rewrite according to the current model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', required=True, type=str, help='the encoder class to be used')
    parser.add_argument('--experiment', required=True, type=str, help='the experiment name')
    parser.add_argument('--epochs', default=50, type=int, help='the number of epochs to train the model')
    args = parser.parse_args()

    try:
        encoder_cls = eval(args.encoder)
    except:
        print("Couldn`t find the specified encoder")
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using '{device}' for training")

    batch_size = 64
    train_data, valid_data, test_data, text_field, label_field = get_data_iterators(batch_size, device=device)

    vocab_size, emb_dim = text_field.vocab.vectors.shape
    encoder = encoder_cls(vocab_size=vocab_size, emb_dim=emb_dim, emb_weights=text_field.vocab.vectors)
    trainer = Trainer(encoder, experiment_name=args.experiment, device=device)

    trainer.train(train_data, valid_data, max_epochs=args.epochs)
    trainer.load_checkpoint(experiment_name=args.experiment)
    loss, accuracy = trainer.evaluate(test_data)
    print(f"Test accuracy for {args.encoder} is {accuracy * 100:.2f}%")
