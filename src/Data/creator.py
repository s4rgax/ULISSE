from torch import Tensor

from src.Data.dataset import Sentinel2Dataset, Sentinel2TimeseriesDataset
from torch.utils.data import random_split
from src.Utils.costants import s2_resnet_channels
from configilm.extra.BENv2_utils import means, stds
from torchvision import transforms
import numpy as np
import torch


class DatasetCreator:
    def __init__(self, data_config, seed=42):
        self.seed = seed
        self.channel_mean = np.array([means['no_interpolation'][band] for band in s2_resnet_channels])
        self.channel_std = np.array([stds['no_interpolation'][band] for band in s2_resnet_channels])
        self.data_config = data_config

        self.transform = transforms.Compose([])
        if self.data_config['norm']:
            self.transform = transforms.Compose([
                transforms.Normalize(self.channel_mean, self.channel_std)
            ])
        elif self.data_config['div_10k']:
            self.transform = transforms.Compose([
                DivideByConstant()
            ])

    def create_train_val_datasets(self, tile_size=224, enable_augmentation=False, val_split=0.2, **kwargs):
        if not kwargs.get('train_images_dir') and not kwargs.get('additional_images_dirs_train') and (
                'timeseries1_images_dirs' not in kwargs or 'timeseries2_images_dirs' not in kwargs):
            raise ValueError(
                "Devi fornire 'train_images_dir' o 'additional_images_dirs_train' oppure 'timeseries1_images_dirs' e 'timeseries2_images_dirs' nei kwargs.")

        if ('timeseries1_images_dirs' in kwargs and kwargs['timeseries1_images_dirs']['train'] and
                'timeseries2_images_dirs' in kwargs and kwargs['timeseries2_images_dirs']['train']):
            full_train_dataset = Sentinel2TimeseriesDataset(
                tile_size=tile_size,
                phase='train',
                img_transform=self.transform,
                enable_augmentation=enable_augmentation,
                **kwargs
            )
        elif kwargs.get('train_images_dir') and kwargs.get('additional_images_dirs_train'):
            full_train_dataset = Sentinel2Dataset(
                tile_size=tile_size,
                phase='train',
                img_transform=self.transform,
                enable_augmentation=enable_augmentation,
                **kwargs
            )
        else:
            full_train_dataset = Sentinel2Dataset(
                tile_size=tile_size,
                phase='train',
                img_transform=self.transform,
                enable_augmentation=enable_augmentation,
                **kwargs
            )

        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        return train_dataset, val_dataset

    def create_test_dataset(self, tile_size=224, enable_forest_type=False, **kwargs):
        if not kwargs.get('test_images_dir') and not kwargs.get('additional_images_dirs_test') and (
                'timeseries1_images_dirs' not in kwargs or 'timeseries2_images_dirs' not in kwargs):
            raise ValueError(
                "Devi fornire 'test_images_dir' o 'additional_images_dirs_test' oppure 'timeseries1_images_dirs' e 'timeseries2_images_dirs' nei kwargs.")

        if ('timeseries1_images_dirs' in kwargs and kwargs['timeseries1_images_dirs']['test'] and
                'timeseries2_images_dirs' in kwargs and kwargs['timeseries2_images_dirs']['test']):
            return Sentinel2TimeseriesDataset(
                tile_size=tile_size,
                phase='test',
                img_transform=self.transform,
                enable_augmentation=False,
                add_forest_type=enable_forest_type,
                **kwargs
            )
        elif kwargs.get('test_images_dir') and kwargs.get('additional_images_dirs_test'):
            return Sentinel2Dataset(
                images_dir=kwargs['test_images_dir'],
                additional_images_dirs=kwargs['additional_images_dirs_test'],
                tile_size=tile_size,
                phase='test',
                img_transform=self.transform,
                enable_augmentation=False,
                add_forest_type=enable_forest_type,
                **kwargs
            )
        else:
            return Sentinel2Dataset(
                images_dir=kwargs['test_images_dir'],
                tile_size=tile_size,
                phase='test',
                img_transform=self.transform,
                enable_augmentation=False,
                add_forest_type=enable_forest_type,
                **kwargs
            )


class DivideByConstant(torch.nn.Module):
    """Divide un tensore per una costante.
    Data una costante, questa trasformazione dividerà ogni elemento del tensore
    di input per tale costante, i.e.,
    ``output = input / constant``

    .. note::
        Questa trasformazione agisce out of place, ovvero non modifica il tensore di input.

    Args:
        constant (float): La costante per cui dividere il tensore.
        inplace (bool, optional): Bool per eseguire l'operazione in-place.
    """

    def __init__(self, constant=10000.0, inplace=False):
        super().__init__()
        self.constant = constant
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensore da dividere.

        Returns:
            Tensor: Tensore diviso per la costante.
        """
        if self.inplace:
            tensor.div_(self.constant)
            return tensor
        else:
            return tensor.div(self.constant)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(constant={self.constant})"