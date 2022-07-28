from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pandas as pd

from tqdm import tqdm
import numpy as np
import cv2


class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.img_labels = df
        self.images = []
        for fn in tqdm(self.img_labels['filename'].tolist(), total=len(df)):
            fn2 = fn.replace('.json', '.jpg')
            #_img = cv2.imread(f'/data/Image_processed/{fn2}')
            _img = f'/data/Image_processed/{fn2}'
            self.images.append(_img)
        self.images = np.array(self.images)
        self.ages = np.array(self.img_labels['age'].tolist()).astype(np.float32)
        self.transform = transform
        # transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.RandomAffine(
        #             degrees=5,
        #             translate=(0,0.1),
        #             scale=(0.8, 1.2),
        #         ),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         )
        #     ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #img_name = self.img_labels['filename'][idx]
        image = self.images[idx]
        image = cv2.imread(image)
        image = self.transform(image)
        age = self.ages[idx]
        return image, age

class AgeDataModule(LightningDataModule):
    """Example of LightningDataModule for AGE dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "/data/Image_processed/",
        train_val_test_split: Tuple[int, int, int] = (35000, 10793, 10793),
        batch_size: int = 10,
        num_workers: int = 2,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # @property
    # def num_classes(self):
    #     return 10

    # def prepare_data(self):
    #     """Download data if needed.

    #     Do not use it to assign state (self.x = y).
    #     """
    #     MNIST(self.hparams.data_dir, train=True, download=True)
    #     MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        df = pd.read_csv('/data/Label/trimmed_patient_metadata.csv')
        df = df[df.position == 'Lateral']
        df = df.reset_index()[25000:]
        # train_df, val_df, test_df = df[:35000], df[35000:45000], df[45000:]

        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ImageDataset(df, transform=self.transforms)
            # trainset = ImageDataset( train_df, train=True, transform=self.transforms)
            # testset = imagedataset( test_df, train=false, transform=self.transforms)
            # validset = imagedataset( valid_df, train=false, transform=self.transforms)
            # dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            # self.data_train = self.data_train[:300]
            # self.data_test = self.data_test[:300]
            # self.data_val = self.data_val[:300]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "age.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
