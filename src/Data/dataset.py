import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio
import torch.nn.functional as F
import albumentations as A
import os


class Sentinel2Dataset:
    def __init__(self, tile_size=224, phase='train', img_transform=None, mask_transform=None, enable_augmentation=False, add_forest_type=False, fill_gap=False, **kwargs):
        self.tile_size = tile_size
        self.phase = phase
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.enable_augmentation = enable_augmentation
        self.add_forest_type = add_forest_type

        self.images_dir = kwargs.get(f'{phase}_images_dir')
        self.additional_images_dirs = kwargs.get(f'additional_images_dirs_{phase}')
        self.mask_values_map = kwargs.get('mask_values_map')
        self.masks_dir = Path(kwargs.get(f'{phase}_masks_dir'))
        self.forest_type_path = kwargs.get('forest_type_path')

        self.fill_gap = fill_gap

        self.augmentations = [('original', None)]
        if self.enable_augmentation:
            self.augmentations.extend([
                ('horizontal_flip', A.HorizontalFlip(p=1)),
                ('vertical_flip', A.VerticalFlip(p=1)),
                ('rotate90', A.RandomRotate90(p=1)),
                ('transpose', A.Transpose(p=1)),
                ('grid_distortion', A.GridDistortion(p=1)),
                # ('random_grid_shuffle', A.RandomGridShuffle(p=1))
            ])

        # self.mask_files = list(self.masks_dir.glob('mask_*.tif'))
        # self.mask_files = list(self.masks_dir.glob('geojson_*.tif'))

        mask_globs = ['mask_*.tif', 'geojson_*.tif']
        self.mask_files = []
        for glob_pattern in mask_globs:
            self.mask_files.extend(list(self.masks_dir.glob(glob_pattern)))

        self.tile_indices = []

        self.current_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.new_channels = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]

        for mask_path in self.mask_files:
            mask_id = mask_path.stem.split('_')[1]
            image_path = os.path.join(self.images_dir, f"geojson_{mask_id}.tif")
            print(image_path)

            additional_image_paths = []
            all_additional_images_exist = True

            if self.additional_images_dirs:
                for additional_dir in self.additional_images_dirs:
                    additional_path = os.path.join(additional_dir, f"geojson_{mask_id}.tif")
                    if not os.path.exists(additional_path):
                        print(f"Warning: Additional image not found at {additional_path}")
                        all_additional_images_exist = False
                    additional_image_paths.append(additional_path)

            forest_type_path = os.path.join(self.forest_type_path,
                                            f"geojson_{mask_id}.tif") if self.forest_type_path else None

            if os.path.exists(image_path):
                with rasterio.open(image_path) as src:
                    image = src.read()
                    _, h, w = image.shape

                if self.add_forest_type and (not self.forest_type_path or not os.path.exists(forest_type_path)):
                    print(f"Warning: Forest type image not found for {mask_id}")
                    continue

                for i in range(0, h, self.tile_size):
                    for j in range(0, w, self.tile_size):
                        for aug_name, _ in self.augmentations:
                            elem_dict = {
                                'image_path': image_path,
                                'additional_image_paths': additional_image_paths,
                                'mask_path': mask_path,
                                'forest_type_path': forest_type_path,
                                'i': i,
                                'j': j,
                                'h': h,
                                'w': w,
                                'aug_type': aug_name
                            }
                            self.tile_indices.append(elem_dict)
            else:
                print(f"Warning: No primary image found for mask {mask_path}")

        print(f"Generated metadata for {len(self.tile_indices)} tiles for {phase} "
              f"(including {len(self.augmentations)} augmentation variants)")

    def apply_augmentation(self, image, mask, forest_type=None, aug_type='original'):
        """
        Apply specified augmentation to image, mask, and optional forest type.

        Args:
            image (np.ndarray): Image to augment (C, H, W)
            mask (np.ndarray): Mask to augment (H, W)
            forest_type (np.ndarray, optional): Forest type image to augment (H, W)
            aug_type (str): Augmentation type to apply

        Returns:
            tuple: Augmented (image, mask, forest_type)
        """
        if aug_type == 'original':
            return (image, mask, forest_type) if forest_type is not None else (image, mask)

        aug = next((transform for name, transform in self.augmentations if name == aug_type), None)
        if aug is None:
            return (image, mask, forest_type) if forest_type is not None else (image, mask)

        image = np.transpose(image, (1, 2, 0))

        aug_inputs = {'image': image, 'mask': mask}
        if forest_type is not None:
            aug_inputs['forest_type'] = forest_type

        transformed = aug(**aug_inputs)

        aug_image = np.transpose(transformed['image'], (2, 0, 1))
        aug_mask = transformed['mask']
        aug_forest_type = transformed.get('forest_type')

        return (aug_image, aug_mask, aug_forest_type) if aug_forest_type is not None else (aug_image, aug_mask)

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, idx):
        """
        Retrieve a specific tile from the dataset.

        Returns:
            Dict containing primary image, list of additional images, mask, valid mask,
            position, image ID, and optionally forest type image
        """
        tile_info = self.tile_indices[idx]
        image_path = tile_info['image_path']
        additional_image_paths = tile_info['additional_image_paths']
        mask_path = tile_info['mask_path']
        forest_type_path = tile_info['forest_type_path'] if self.add_forest_type else None
        i, j = tile_info['i'], tile_info['j']
        h, w = tile_info['h'], tile_info['w']
        aug_type = tile_info['aug_type']

        with rasterio.open(image_path) as src:
            image = src.read()
            image = self.reorder_select_channels(image, self.current_channels, self.new_channels)
            if self.fill_gap:
                image = self.fill_gaps_nearest_neighbor(image)

        additional_images = []
        for add_path in additional_image_paths:
            try:
                with rasterio.open(add_path) as src:
                    add_image = src.read()
                    add_image = self.reorder_select_channels(add_image, self.current_channels, self.new_channels)
                    if self.fill_gap:
                        add_image = self.fill_gaps_nearest_neighbor(add_image)
                    additional_images.append(add_image)
            except:
                additional_images.append(np.zeros_like(image))
                print(f"Warning: Could not read additional image at {add_path}")

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            if self.mask_values_map:
                mask = np.isin(mask, self.mask_values_map).astype(int)

        forest_type = None
        if self.add_forest_type and forest_type_path:
            with rasterio.open(forest_type_path) as src:
                forest_type = src.read(1)

        i_end = min(i + self.tile_size, h)
        j_end = min(j + self.tile_size, w)

        img_tile = image[:, i:i_end, j:j_end]
        additional_img_tiles = [img[:, i:i_end, j:j_end] for img in additional_images]
        mask_tile = mask[i:i_end, j:j_end]
        forest_type_tile = forest_type[i:i_end, j:j_end] if forest_type is not None else None

        if forest_type_tile is not None:
            img_tile, mask_tile, forest_type_tile = self.apply_augmentation(
                image=img_tile, mask=mask_tile, forest_type=forest_type_tile, aug_type=aug_type
            )
            additional_img_tiles = [
                self.apply_augmentation(
                    image=add_tile, mask=mask_tile.copy(), forest_type=forest_type_tile.copy(), aug_type=aug_type
                )[0] for add_tile in additional_img_tiles
            ]
        else:
            img_tile, mask_tile = self.apply_augmentation(image=img_tile, mask=mask_tile, aug_type=aug_type)
            additional_img_tiles = [
                self.apply_augmentation(image=add_tile, mask=mask_tile.copy(), aug_type=aug_type)[0]
                for add_tile in additional_img_tiles
            ]

        valid_mask = torch.ones((1, img_tile.shape[1], img_tile.shape[2]))

        if mask_tile.ndim == 2:
            mask_tile = np.expand_dims(mask_tile, axis=0)
        if forest_type_tile is not None and forest_type_tile.ndim == 2:
            forest_type_tile = np.expand_dims(forest_type_tile, axis=0)

        # Handle padding if needed
        if img_tile.shape[1] < self.tile_size or img_tile.shape[2] < self.tile_size:
            padding = ((0, 0), (0, self.tile_size - img_tile.shape[1]), (0, self.tile_size - img_tile.shape[2]))
            img_tile = np.pad(img_tile, padding, mode='constant', constant_values=0)
            additional_img_tiles = [np.pad(add_tile, padding, mode='constant', constant_values=0)
                                    for add_tile in additional_img_tiles]
            mask_tile = np.pad(mask_tile, padding, mode='constant', constant_values=0)
            if forest_type_tile is not None:
                forest_type_tile = np.pad(forest_type_tile, padding, mode='constant', constant_values=0)
            valid_mask = F.pad(valid_mask,
                               (0, self.tile_size - valid_mask.shape[2],
                                0, self.tile_size - valid_mask.shape[1]),
                               value=0)

        img_tensor = torch.from_numpy(img_tile).float()
        additional_img_tensors = [torch.from_numpy(add_tile).float() for add_tile in additional_img_tiles]
        mask_tensor = torch.from_numpy(mask_tile).float()
        forest_type_tensor = torch.from_numpy(forest_type_tile).float() if forest_type_tile is not None else None

        if self.img_transform:
            img_tensor = self.img_transform(img_tensor)
            additional_img_tensors = [self.img_transform(add_tensor) for add_tensor in additional_img_tensors]
        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_tensor)
        if forest_type_tensor is not None and self.forest_type_transform:
            forest_type_tensor = self.forest_type_transform(forest_type_tensor)

        #TODO aggiungere qui il masking eventuale mettendo in and mask ==1 or forest_type == 2 (questi sono i pixel da lasciare, allora 1)

        image_id = Path(image_path).stem
        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'valid_mask': valid_mask,
            'position': (i, j),
            'image_id': image_id,
            'patch_mask': int(mask_tile.sum() > 0)
        }

        if len(additional_img_tensors) > 0:
            additional_img_tensors = torch.stack(additional_img_tensors)
            result['additional_images'] = additional_img_tensors

        if forest_type_tensor is not None:
            result['forest_type'] = forest_type_tensor

        return result

    def reorder_select_channels(self, image, current_channels, new_channels):
        """
        Reorder channels of a NumPy array based on a new channel list.

        Args:
            image: NumPy array of shape (C, H, W)
            current_channels: List of current channels
            new_channels: List of new channels in desired order
        Returns:
            NumPy array with reordered channels
        """
        channel_to_index = {channel: idx for idx, channel in enumerate(current_channels)}
        new_indices = [channel_to_index[channel] for channel in new_channels]
        reordered_image = image[new_indices, :, :]
        return reordered_image


    def fill_gaps_nearest_neighbor(self, image):
        """
        Fill NaN/nodata values in image using nearest neighbor interpolation.

        Args:
            image: NumPy array of shape (C, H, W)

        Returns:
            NumPy array with filled gaps
        """
        filled_image = image.copy()

        for channel_idx in range(image.shape[0]):
            channel = filled_image[channel_idx]

            mask = np.isnan(channel) | (channel == 0)

            if np.any(mask):
                valid_coords = np.argwhere(~mask)
                invalid_coords = np.argwhere(mask)

                if len(valid_coords) > 0 and len(invalid_coords) > 0:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(valid_coords)
                    _, nearest_indices = tree.query(invalid_coords)

                    for i, invalid_coord in enumerate(invalid_coords):
                        nearest_coord = valid_coords[nearest_indices[i]]
                        filled_image[channel_idx, invalid_coord[0], invalid_coord[1]] = \
                            channel[nearest_coord[0], nearest_coord[1]]

        return filled_image


class Sentinel2TimeseriesDataset(Dataset):
    def __init__(self, tile_size=224, phase='train', img_transform=None, enable_augmentation=False, mask_transform=None, **kwargs):
        self.tile_size = tile_size
        self.phase = phase
        self.img_transform = img_transform
        self.enable_augmentation = enable_augmentation
        self.mask_transform = mask_transform

        self.timeseries1_images_dirs = [Path(p) for p in kwargs.get('timeseries1_images_dirs', {}).get(phase, []) or []]
        self.timeseries2_images_dirs = [Path(p) for p in kwargs.get('timeseries2_images_dirs',{}).get(phase,[]) or []]
        self.masks_dir = Path(kwargs.get(f'{phase}_masks_dir', ''))
        self.forest_type_dir = Path(kwargs.get('forest_type_path', '')) if kwargs.get('forest_type_path') else None
        self.mask_values_map = kwargs.get('mask_values_map', [])
        self.add_forest_type = kwargs.get('add_forest_type', False)
        # TODO da metterci i valori da filtrare per le forest type da trasformare in uno
        self.forest_type_transform = kwargs.get('forest_type_transform', None)

        self.augmentations = [('original', None)]
        if self.enable_augmentation:
            self.augmentations.extend([
                ('horizontal_flip', A.HorizontalFlip(p=1)),
                ('vertical_flip', A.VerticalFlip(p=1)),
                ('rotate90', A.RandomRotate90(p=1)),
                ('transpose', A.Transpose(p=1)),
                ('grid_distortion', A.GridDistortion(p=1))
            ])

        self.current_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.new_channels = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]

        self.tile_indices = []
        self.mask_files = list(self.masks_dir.glob('mask_*.tif'))

        for mask_path in self.mask_files:
            mask_id = mask_path.stem.split('_')[1]

            timeseries1_paths = []
            for ts1_dir in self.timeseries1_images_dirs:
                ts1_path = os.path.join(ts1_dir, f"geojson_{mask_id}.tif")
                if not os.path.exists(ts1_path):
                    print(f"Warning: Time series 1 image not found at {ts1_path}")
                timeseries1_paths.append(ts1_path)

            timeseries2_paths = []
            for ts2_dir in self.timeseries2_images_dirs:
                ts2_path = os.path.join(ts2_dir, f"geojson_{mask_id}.tif")
                if not os.path.exists(ts2_path):
                    print(f"Warning: Time series 2 image not found at {ts2_path}")
                timeseries2_paths.append(ts2_path)

            forest_type_path = os.path.join(self.forest_type_dir,
                                            f"geojson_{mask_id}.tif") if self.forest_type_dir else None

            if len(timeseries1_paths) > 0 and os.path.exists(timeseries1_paths[0]) and \
                    len(timeseries2_paths) > 0 and os.path.exists(timeseries2_paths[0]):

                with rasterio.open(timeseries1_paths[0]) as src:
                    image = src.read()
                    _, h, w = image.shape

                if self.add_forest_type and (not self.forest_type_dir or not os.path.exists(forest_type_path)):
                    print(f"Warning: Forest type image not found for {mask_id}")
                    continue

                for i in range(0, h, self.tile_size):
                    for j in range(0, w, self.tile_size):
                        for aug_name, _ in self.augmentations:
                            elem_dict = {
                                'timeseries1_paths': timeseries1_paths,
                                'timeseries2_paths': timeseries2_paths,
                                'mask_path': mask_path,
                                'forest_type_path': forest_type_path,
                                'i': i,
                                'j': j,
                                'h': h,
                                'w': w,
                                'aug_type': aug_name
                            }
                            self.tile_indices.append(elem_dict)
            else:
                print(f"Warning: Time series images not found for mask {mask_path}")

        print(f"Generated metadata for {len(self.tile_indices)} tiles for {phase} "
              f"(including {len(self.augmentations)} augmentation variants)")

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, idx):
        """
        Retrieve a specific tile from the dataset with two time series.

        Returns:
            Dict containing time series 1 and 2 images, mask, valid mask,
            position, image ID, and optionally forest type image
        """
        tile_info = self.tile_indices[idx]
        timeseries1_paths = tile_info['timeseries1_paths']
        timeseries2_paths = tile_info['timeseries2_paths']
        mask_path = tile_info['mask_path']
        forest_type_path = tile_info['forest_type_path'] if self.add_forest_type else None
        i, j = tile_info['i'], tile_info['j']
        h, w = tile_info['h'], tile_info['w']
        aug_type = tile_info['aug_type']

        timeseries1_images = []
        for ts1_path in timeseries1_paths:
            try:
                with rasterio.open(ts1_path) as src:
                    image = src.read()
                    image = self.reorder_select_channels(image, self.current_channels, self.new_channels)
                    timeseries1_images.append(image)
            except Exception as e:
                print(f"Error loading time series 1 image {ts1_path}: {e}")
                timeseries1_images.append(np.zeros_like(timeseries1_images[0]) if timeseries1_images else None)

        timeseries2_images = []
        for ts2_path in timeseries2_paths:
            try:
                with rasterio.open(ts2_path) as src:
                    image = src.read()
                    image = self.reorder_select_channels(image, self.current_channels, self.new_channels)
                    timeseries2_images.append(image)
            except Exception as e:
                print(f"Error loading time series 2 image {ts2_path}: {e}")
                timeseries2_images.append(np.zeros_like(timeseries2_images[0]) if timeseries2_images else None)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            if self.mask_values_map:
                mask = np.isin(mask, self.mask_values_map).astype(int)

        forest_type = None
        if self.add_forest_type and forest_type_path:
            with rasterio.open(forest_type_path) as src:
                forest_type = src.read(1)

        i_end = min(i + self.tile_size, h)
        j_end = min(j + self.tile_size, w)

        ts1_img_tiles = [img[:, i:i_end, j:j_end] for img in timeseries1_images]
        ts2_img_tiles = [img[:, i:i_end, j:j_end] for img in timeseries2_images]
        mask_tile = mask[i:i_end, j:j_end]
        forest_type_tile = forest_type[i:i_end, j:j_end] if forest_type is not None else None

        if forest_type_tile is not None:
            ts1_img_tiles, ts2_img_tiles, mask_tile, forest_type_tile = self.apply_augmentation(
                images_t1=ts1_img_tiles,
                images_t2=ts2_img_tiles,
                mask=mask_tile.copy(),
                forest_type=forest_type_tile.copy(),
                aug_type=aug_type
            )
        else:
            ts1_img_tiles, ts2_img_tiles, mask_tile, forest_type_tile = self.apply_augmentation(
                images_t1=ts1_img_tiles,
                images_t2=ts2_img_tiles,
                mask=mask_tile.copy(),
                aug_type=aug_type
            )

        valid_mask = torch.ones((1, ts1_img_tiles[0].shape[1], ts1_img_tiles[0].shape[2]))

        if mask_tile.ndim == 2:
            mask_tile = np.expand_dims(mask_tile, axis=0)
        if forest_type_tile is not None and forest_type_tile.ndim == 2:
            forest_type_tile = np.expand_dims(forest_type_tile, axis=0)

        if ts1_img_tiles[0].shape[1] < self.tile_size or ts1_img_tiles[0].shape[2] < self.tile_size:
            padding = (
            (0, 0), (0, self.tile_size - ts1_img_tiles[0].shape[1]), (0, self.tile_size - ts1_img_tiles[0].shape[2]))

            ts1_img_tiles = [np.pad(tile, padding, mode='constant', constant_values=0) for tile in ts1_img_tiles]
            ts2_img_tiles = [np.pad(tile, padding, mode='constant', constant_values=0) for tile in ts2_img_tiles]
            mask_tile = np.pad(mask_tile, padding, mode='constant', constant_values=0)
            if forest_type_tile is not None:
                forest_type_tile = np.pad(forest_type_tile, padding, mode='constant', constant_values=0)
            valid_mask = F.pad(valid_mask,
                               (0, self.tile_size - valid_mask.shape[2],
                                0, self.tile_size - valid_mask.shape[1]),
                               value=0)

        ts1_img_tensors = [torch.from_numpy(img_tile).float() for img_tile in ts1_img_tiles]
        ts2_img_tensors = [torch.from_numpy(img_tile).float() for img_tile in ts2_img_tiles]
        mask_tensor = torch.from_numpy(mask_tile).float()
        forest_type_tensor = torch.from_numpy(forest_type_tile).float() if forest_type_tile is not None else None

        if self.img_transform:
            ts1_img_tensors = [self.img_transform(img_tensor) for img_tensor in ts1_img_tensors]
            ts2_img_tensors = [self.img_transform(img_tensor) for img_tensor in ts2_img_tensors]

        if self.mask_transform:
            mask_tensor = self.mask_transform(mask_tensor)

        if forest_type_tensor is not None and self.forest_type_transform:
            forest_type_tensor = self.forest_type_transform(forest_type_tensor)

        image_id = Path(timeseries1_paths[0]).stem
        result = {
            'timeseries1_images': torch.stack(ts1_img_tensors),
            'timeseries2_images': torch.stack(ts2_img_tensors),
            'mask': mask_tensor,
            'valid_mask': valid_mask,
            'position': (i, j),
            'image_id': image_id
        }

        if forest_type_tensor is not None:
            result['forest_type'] = forest_type_tensor

        return result

    def reorder_select_channels(self, image, current_channels, new_channels):
        """
        Reorder channels of a NumPy array based on a new channel list.

        Args:
            image: NumPy array of shape (C, H, W)
            current_channels: List of current channels
            new_channels: List of new channels in desired order
        Returns:
            NumPy array with reordered channels
        """
        channel_to_index = {channel: idx for idx, channel in enumerate(current_channels)}
        new_indices = [channel_to_index[channel] for channel in new_channels]
        reordered_image = image[new_indices, :, :]
        return reordered_image

    def apply_augmentation(self, images_t1, images_t2, mask, forest_type=None, aug_type='original'):
        """
        Apply specified augmentation to two lists of images, mask, and optional forest type.

        Args:
            images_t1 (list): Prima lista di immagini da aumentare, ciascuna di forma (C, H, W)
            images_t2 (list): Seconda lista di immagini da aumentare, ciascuna di forma (C, H, W)
            mask (np.ndarray): Mask da aumentare (H, W)
            forest_type (np.ndarray, optional): Forest type image da aumentare (H, W)
            aug_type (str): Tipo di augmentation da applicare

        Returns:
            tuple: (images_t1_aumentate, images_t2_aumentate, mask_aumentata, forest_type_aumentato) o
                   (images_t1_aumentate, images_t2_aumentate, mask_aumentata) se forest_type è None
        """
        if aug_type == 'original':
            return (images_t1, images_t2, mask, forest_type) if forest_type is not None else (
            images_t1, images_t2, mask, None)

        aug = next((transform for name, transform in self.augmentations if name == aug_type), None)
        if aug is None:
            return (images_t1, images_t2, mask, forest_type) if forest_type is not None else (
            images_t1, images_t2, mask, None)

        aug_inputs = {'image': np.transpose(images_t1[0], (1, 2, 0)), 'mask': mask}
        if forest_type is not None:
            aug_inputs['forest_type'] = forest_type

        additional_targets = {}
        for i in range(1, len(images_t1)):
            additional_targets[f'image_t1_{i}'] = 'image'
        for i in range(len(images_t2)):
            additional_targets[f'image_t2_{i}'] = 'image'

        augmentation = A.Compose([aug], additional_targets=additional_targets)

        for i in range(1, len(images_t1)):
            img_key = f'image_t1_{i}'
            aug_inputs[img_key] = np.transpose(images_t1[i], (1, 2, 0))

        for i in range(len(images_t2)):
            img_key = f'image_t2_{i}'
            aug_inputs[img_key] = np.transpose(images_t2[i], (1, 2, 0))

        transformed = augmentation(**aug_inputs)

        aug_images_t1 = []
        aug_images_t1.append(np.transpose(transformed['image'], (2, 0, 1)))  # Prima immagine (chiave 'image')
        for i in range(1, len(images_t1)):
            img_key = f'image_t1_{i}'
            aug_img = np.transpose(transformed[img_key], (2, 0, 1))
            aug_images_t1.append(aug_img)

        aug_images_t2 = []
        for i in range(len(images_t2)):
            img_key = f'image_t2_{i}'
            aug_img = np.transpose(transformed[img_key], (2, 0, 1))
            aug_images_t2.append(aug_img)

        aug_mask = transformed['mask']
        aug_forest_type = transformed.get('forest_type')

        return aug_images_t1, aug_images_t2, aug_mask, aug_forest_type