import logging
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from counterfactualms.datasets.calabresi import CalabresiDataset
from counterfactualms.arch.medical import Encoder

logger = logging.getLogger(__name__)


class ClassifierExperiment(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size
        self.input_size = ((3,) if hparams.pseudo3d else (1,)) + tuple(hparams.resize)
        self.classifier = Encoder(num_convolutions=hparams.num_convolutions,
                                  filters=hparams.filters,
                                  input_size=self.input_size,
                                  latent_dim=1,
                                  use_weight_norm=hparams.use_weight_norm,
                                  dropout_rate=hparams.dropout_rate)
        self.classifier.fc = nn.Linear(np.prod(self.classifier.intermediate_shape), 1, bias=True)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        if hparams.validate:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)

        resize = None if self.hparams.resize == (0,0) else self.hparams.resize
        train_crop_type = self.hparams.train_crop_type if hasattr(self.hparams, 'train_crop_type') else 'random'
        crop_size = self.hparams.crop_size if hasattr(self.hparams, 'crop_size') else (224, 224)
        self.calabresi_train = CalabresiDataset(self.hparams.train_csv, crop_size=crop_size, crop_type=train_crop_type, resize=resize)  # noqa: E501
        self.calabresi_val = CalabresiDataset(self.hparams.valid_csv, crop_size=crop_size, crop_type='center', resize=resize)
        self.calabresi_test = CalabresiDataset(self.hparams.test_csv, crop_size=crop_size, crop_type='center', resize=resize)

    @property
    def required_data(self):
        return {'x', 'type'}

    def configure_optimizers(self):
        optimizer = AdamW(self.classifier.parameters(), lr=self.hparams.lr,
                          betas=self.hparams.betas, weight_decay=self.hparams.weight_decay)
        return optimizer

    def _dataloader_params(self):
        num_workers = len(os.sched_getaffinity(0)) // 2  # use half of the available cpus
        return {'num_workers': num_workers, 'pin_memory': self.trainer.on_gpu}

    def train_dataloader(self):
        return DataLoader(self.calabresi_train, batch_size=self.train_batch_size,
                          shuffle=True, drop_last=True, **self._dataloader_params())

    def val_dataloader(self):
        return DataLoader(self.calabresi_val, batch_size=self.test_batch_size,
                          shuffle=False, **self._dataloader_params())

    def test_dataloader(self):
        return DataLoader(self.calabresi_test, batch_size=self.test_batch_size,
                          shuffle=False, **self._dataloader_params())

    def _theis_noise(self, obs):
        """ add noise to discrete variables per Theis 2016 """
        if self.training:
            obs['x'] += (torch.rand_like(obs['x']) - 0.5)
        return obs

    def prep_batch(self, batch):
        x = batch['image'].float()
        out = dict(x=x)
        for k in self.required_data:
            if k in batch:
                out[k] = batch[k].unsqueeze(1).float()
        out = self._theis_noise(out)
        return out

    def _step(self, batch, label=''):
        batch = self.prep_batch(batch)
        preds = self.classifier(batch['x'])
        loss = F.binary_cross_entropy_with_logits(preds, batch['type'])
        self.log(f'{label}_loss', loss)
        acc = getattr(self, f'{label}_acc')
        acc(preds.sigmoid(), batch['type'])
        self.log(f'{label}_acc', acc, on_step=False, on_epoch=True)
        return loss

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, 'test')
        return loss

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--train-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/train_png.csv", type=str, help="csv for training data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--valid-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/valid_png.csv", type=str, help="csv for validation data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--test-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/test_png.csv", type=str, help="csv for testing data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--crop-size', default=(224,224), type=int, nargs=2, help="size of patch to take from image (default: %(default)s)")
        parser.add_argument('--resize', default=(128,128), type=int, nargs=2, help="resize cropped image to this size (use 0,0 for no resize) (default: %(default)s)")
        parser.add_argument('--train-crop-type', default='random', choices=['random', 'center'], help="how to crop training images (default: %(default)s)")
        parser.add_argument('--train-batch-size', default=128, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test-batch-size', default=256, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--lr', default=1e-3, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--weight-decay', default=0., type=float, help="weight decay for adam (default: %(default)s)")
        parser.add_argument('--dropout-rate', default=0., type=float, help="dropout rate for classifier (default: %(default)s)")
        parser.add_argument('--betas', default=(0.9,0.999), type=float, nargs=2, help="betas for adam (default: %(default)s)")
        parser.add_argument('--filters', default=[8,16,32,64,128], nargs='+', type=int, help="number of filters in each layer of classifier (default: %(default)s)")
        parser.add_argument('--num-convolutions', default=3, type=int, help="number of convolutions in each layer (default: %(default)s)")
        parser.add_argument('--use-weight-norm', default=False, action='store_true', help="use weight norm in conv layers (default: %(default)s)")
        parser.add_argument('--pseudo3d', default=False, action='store_true', help="use pseudo-3d images (3 channels) (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="more verbose validation (default: %(default)s)")
        return parser
