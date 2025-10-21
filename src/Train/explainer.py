import os
import random
import re
from pathlib import Path
import rasterio
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from collections import defaultdict
from more_itertools import chunked
from PIL import Image
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from matplotlib.table import Table
import matplotlib.pyplot as plt
import seaborn as sns

from src.Model.models import ResNetUNet, MultiLoraResNetUNet, TimeseriesMultiLoraResNetUNet
from src.Model.utils import TverskyLoss
from src.Utils.enums import TemporalMode
from src.Utils.functions import compute_metrics_from_conf_matrix, compute_f1, transform_batch_positions



class UNetExplainer:
    def __init__(
            self,
            model_name,
            train_dataset,
            val_dataset,
            test_dataset=None,
            data_tile_size=224,
            model_tile_size=224,
            optimization_metric='loss',
            sfreeze_encoder_after=10,
            freeze_encoder=False,
            num_workers_dl=4,
            output_dir='results',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            temporal_mode=False,
            peft_encoder = None,
            num_additional_images = 0,
            num_images_per_series=3,
            fusion_mode=None,
            fusion_technique=None,
            batch_size = [4, 8, 16, 32, 64],
            rank = [2, 4, 8, 16, 32, 64]
    ):
        self.device = device
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.data_input_size = (data_tile_size, data_tile_size)
        self.model_input_size = (model_tile_size, model_tile_size)
        self.optimization_metric = optimization_metric
        self.sfreeze_encoder_after = sfreeze_encoder_after
        self.freeze_encoder = freeze_encoder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers_dl = num_workers_dl
        self.temporal_mode = temporal_mode
        self.num_additional_images = num_additional_images
        self.fusion_mode = fusion_mode
        self.fusion_technique = fusion_technique
        self.len_image_series = num_images_per_series
        self.rank = rank
        self.batch_size = batch_size

        self.models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.peft_encoder = peft_encoder

        self.set_reproducibility()

        print(f"Batch size choices: ", self.batch_size)
        self.space = {
            'batch_size': hp.choice('batch_size', self.batch_size),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-3)),
            'tversky_alpha': hp.choice('tversky_alpha', [.1, .2, .3, .4, .5]),
        }
        if self.peft_encoder:
            print(f"Rank choices: ", self.rank)
            self.space['lora_rank'] = hp.choice('lora_rank', self.rank)

        self.trials_history = []
        self.best_global_metric = float('inf') if optimization_metric == 'loss' else float('-inf')
        self.best_test_global_metric = float('inf') if optimization_metric == 'loss' else float('-inf')
        self.best_trial_id = None

    def load_best_model(self, disable_peft_indexes=None):
        '''checkpoint = torch.load(os.path.join(self.output_dir, 'best_model.pth'), map_location=torch.device(self.device))'''
        checkpoint = torch.load(os.path.join(self.output_dir, 'best_model.pth'), weights_only=False, map_location=torch.device(self.device))
        print(checkpoint['hyperparameters'])
        lora_rank = checkpoint['hyperparameters']['lora_rank'] if self.peft_encoder else None
        if self.temporal_mode == TemporalMode.TIMESERIES.value:
            self.model = MultiLoraResNetUNet.from_pretrained(
                self.model_name,
                data_tile_size=self.data_input_size,
                model_input_size=self.model_input_size,
                num_classes=1,
                peft=self.peft_encoder,
                peft_attr={'lora_rank': lora_rank},
                # disable_peft_indexes=disable_peft_indexes,
                num_additional_images=self.num_additional_images,
                fusion_mode=self.fusion_mode,
                fusion_technique=self.fusion_technique
            )
        elif self.temporal_mode == TemporalMode.SINGLE.value:
            self.model = ResNetUNet.from_pretrained(
                self.model_name,
                peft=self.peft_encoder,
                peft_attr={'lora_rank': lora_rank},
                data_tile_size=self.data_input_size,
                model_input_size=self.model_input_size,
                num_classes=1
            )

        incompatible_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if incompatible_keys.missing_keys:
            print(f"Ignored {len(incompatible_keys.missing_keys)} missing keys")
        if incompatible_keys.unexpected_keys:
            print(f"Ignored {len(incompatible_keys.unexpected_keys)} unexpected keys")

        self.model = self.model.to(self.device)
        return checkpoint['hyperparameters']

    def calculate_metrics(self, predictions, targets, valid_mask):
        """Calculate confusion matrix and F1 scores for both classes"""
        valid_mask = valid_mask.to(self.device)
        pred_binary = (predictions > 0.5).float() * valid_mask
        targets = targets * valid_mask

        pred_np = pred_binary.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        valid_np = valid_mask.cpu().numpy().flatten()

        valid_indices = valid_np > 0
        pred_np = pred_np[valid_indices]
        target_np = target_np[valid_indices]

        conf_matrix = confusion_matrix(target_np, pred_np, labels=[0, 1])  # Forza le etichette a [0,1]

        if conf_matrix.shape != (2, 2):
            metrics = {
                'conf_matrix': np.zeros((2, 2)),  # Matrice 2x2 di zeri
                'f1_class0': 0.0,
                'f1_class1': 0.0,
                'accuracy': 1.0 if len(np.unique(target_np)) == 1 else 0.0
            }
        else:
            try:
                f1_class0 = f1_score(target_np, pred_np, pos_label=0, zero_division=0)
                f1_class1 = f1_score(target_np, pred_np, pos_label=1, zero_division=0)
                accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()

                metrics = {
                    'conf_matrix': conf_matrix,
                    'f1_class0': f1_class0,
                    'f1_class1': f1_class1,
                    'accuracy': accuracy
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                print(f"Unique values in predictions: {np.unique(pred_np)}")
                print(f"Unique values in targets: {np.unique(target_np)}")
                metrics = {
                    'conf_matrix': np.zeros((2, 2)),
                    'f1_class0': 0.0,
                    'f1_class1': 0.0,
                    'accuracy': 0.0
                }

        return metrics

    def calculate_per_image_metrics(self, predictions, targets, valid_mask, image_ids=None):
        """Calculate metrics for each image individually"""
        batch_size = predictions.size(0)
        metrics_list = []

        for i in range(batch_size):
            metrics = self.calculate_metrics(
                predictions[i:i + 1],
                targets[i:i + 1],
                valid_mask[i:i + 1]
            )

            metrics_dict = {
                'image_id': image_ids[i] if image_ids is not None else i,
                'accuracy': metrics['accuracy'],
                'f1_class0': metrics['f1_class0'],
                'f1_class1': metrics['f1_class1']
            }
            metrics_list.append(metrics_dict)

        return metrics_list

    def evaluate_epoch(self, data_loader=None, phase="eval", calculate_forest_type_metrics=False,
                       precomputed_metrics=None):
        """
        Evaluate the model either using precomputed metrics or data loader

        Args:
            data_loader: Optional DataLoader for evaluation
            phase: Phase name (train/val/test)
            calculate_forest_type_metrics: Whether to calculate forest type specific metrics
            precomputed_metrics: Dict containing precomputed 'loss' and 'conf_matrix'
        """
        if precomputed_metrics is not None:
            total_loss = precomputed_metrics['loss']
            cumulative_conf_matrix = precomputed_metrics['conf_matrix']
            num_samples = precomputed_metrics.get('num_samples', 1)
            cumulative_forest_type_conf_matrix = precomputed_metrics.get(
                'forest_type_conf_matrix') if calculate_forest_type_metrics else None

        elif data_loader is not None:
            self.model.eval()
            total_loss = 0
            cumulative_conf_matrix = np.zeros((2, 2))
            cumulative_forest_type_conf_matrix = np.zeros((2, 2)) if calculate_forest_type_metrics else None
            num_samples = 0

            with torch.no_grad():
                with tqdm(data_loader, desc=f'Evaluating {phase} phase...') as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        masks = batch['mask']
                        valid_masks = batch['valid_mask']
                        forest_type = batch.get('forest_type', None)
                        masks = masks.to(self.device)
                        if self.temporal_mode == TemporalMode.TIMESERIES.value:
                            images = batch['image'].to(self.device)
                            additional_images = batch['additional_images'].to(self.device)
                            outputs = self.model(images, additional_images)
                            num_samples += images.size(0)
                        elif self.temporal_mode == TemporalMode.SINGLE.value:
                            images = batch['image'].to(self.device)
                            outputs = self.model(images)
                            num_samples += images.size(0)
                        loss = self.criterion(outputs, masks, valid_masks)
                        batch_metrics = self.calculate_metrics(
                            torch.sigmoid(outputs),
                            masks,
                            valid_masks
                        )
                        if batch_metrics['conf_matrix'].shape == (2, 2):
                            cumulative_conf_matrix += batch_metrics['conf_matrix']
                        total_loss += loss.item()
                        if calculate_forest_type_metrics and forest_type is not None:
                            forest_type_masks = (forest_type != 0).float()
                            forest_type_outputs = outputs[forest_type_masks == 1]
                            forest_type_true_masks = masks[forest_type_masks == 1]
                            forest_type_valid_masks = valid_masks[forest_type_masks == 1]

                            if forest_type_outputs.numel() > 0:
                                forest_type_metrics = self.calculate_metrics(
                                    torch.sigmoid(forest_type_outputs),
                                    forest_type_true_masks,
                                    forest_type_valid_masks
                                )
                                if forest_type_metrics['conf_matrix'].shape == (2, 2):
                                    cumulative_forest_type_conf_matrix += forest_type_metrics['conf_matrix']
        else:
            raise ValueError("Either data_loader or precomputed_metrics must be provided")

        # Compute metrics for overall dataset
        TP, FP, FN, TN = compute_metrics_from_conf_matrix(cumulative_conf_matrix)

        epoch_metrics = {
            'phase': phase,
            'loss': total_loss / num_samples,
            'accuracy': (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
            'precision_class1': TP / (TP + FP) if (TP + FP) > 0 else 0,
            'recall_class1': TP / (TP + FN) if (TP + FN) > 0 else 0,
            'f1_class1': compute_f1(TP, FP, FN),
            'precision_class0': TN / (TN + FN) if (TN + FN) > 0 else 0,
            'recall_class0': TN / (TN + FP) if (TN + FP) > 0 else 0,
            'f1_class0': compute_f1(TN, FN, FP),
            'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP
        }

        # Add forest type metrics if calculated
        if calculate_forest_type_metrics and cumulative_forest_type_conf_matrix is not None:
            forest_type_TP, forest_type_FP, forest_type_FN, forest_type_TN = compute_metrics_from_conf_matrix(
                cumulative_forest_type_conf_matrix)

            forest_type_metrics = {
                'forest_type_accuracy': (forest_type_TP + forest_type_TN) / (
                        forest_type_TP + forest_type_TN + forest_type_FP + forest_type_FN) if (
                                                                                                      forest_type_TP + forest_type_TN + forest_type_FP + forest_type_FN) > 0 else 0,
                'forest_type_precision_class1': forest_type_TP / (forest_type_TP + forest_type_FP) if (
                                                                                                              forest_type_TP + forest_type_FP) > 0 else 0,
                'forest_type_recall_class1': forest_type_TP / (forest_type_TP + forest_type_FN) if (
                                                                                                           forest_type_TP + forest_type_FN) > 0 else 0,
                'forest_type_f1_class1': compute_f1(forest_type_TP, forest_type_FP, forest_type_FN),
                'forest_type_TN': forest_type_TN,
                'forest_type_FP': forest_type_FP,
                'forest_type_FN': forest_type_FN,
                'forest_type_TP': forest_type_TP
            }

            epoch_metrics.update(forest_type_metrics)

        return epoch_metrics

    def test_with_disabled_peft_months(
            self,
            calculate_forest_type_metrics: bool = False
    ):
        """
        Test the model by replacing PEFT-enabled encoders with base encoders one at a time
        and evaluate the performance impact. Saves results to CSV files.

        Args:
            calculate_forest_type_metrics: Whether to calculate forest type specific metrics
        """
        if self.test_dataset is None:
            print("No test dataset provided")
            return

        if not self.peft_encoder:
            print("PEFT (LoRA) is not enabled for this model. Cannot disable adapters.")
            return

        months = ["September", "August", "July", "June", "May", "April"]

        print("Running baseline test with all LoRA adapters enabled...")
        best_hyperparams = self.load_best_model()

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=int(best_hyperparams['batch_size']),
            shuffle=False,
            num_workers=self.num_workers_dl,
            pin_memory=True
        )

        self.criterion = TverskyLoss(
            alpha=float(best_hyperparams['tversky_alpha']),
            beta=1 - float(best_hyperparams['tversky_alpha'])
        )

        base_metrics = self.evaluate_epoch(
            self.test_loader,
            'test_all_enabled',
            calculate_forest_type_metrics=calculate_forest_type_metrics
        )

        base_metrics.update(best_hyperparams)
        base_metrics['disabled_month'] = "None"
        base_metrics['disabled_adapter_index'] = -1

        all_test_results = [base_metrics]

        month_impacts = []
        base_f1 = base_metrics['f1_class1']
        base_accuracy = base_metrics['accuracy']

        for i, month in enumerate(months):
            print(f"Testing with LoRA adapter for {month} disabled...")

            best_hyperparams = self.load_best_model()

            if hasattr(self.model, 'replace_encoder_with_base'):
                print(f"Replacing PEFT-enabled encoder at index {i} with base encoder")
                self.model.replace_encoder_with_base(i)
            else:
                print("Model does not have replace_encoder_with_base method, skipping")
                continue

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=int(best_hyperparams['batch_size']),
                shuffle=False,
                num_workers=self.num_workers_dl,
                pin_memory=True
            )

            disabled_metrics = self.evaluate_epoch(
                self.test_loader,
                f'test_{month}_disabled',
                calculate_forest_type_metrics=calculate_forest_type_metrics
            )

            disabled_metrics.update(best_hyperparams)
            disabled_metrics['disabled_month'] = month
            disabled_metrics['disabled_adapter_index'] = i

            all_test_results.append(disabled_metrics)

            f1_diff = disabled_metrics['f1_class1'] - base_f1
            accuracy_diff = disabled_metrics['accuracy'] - base_accuracy

            month_impact = {
                'month': month,
                'adapter_index': i,
                'f1_class1': disabled_metrics['f1_class1'],
                'f1_class0': disabled_metrics['f1_class0'],
                'accuracy': disabled_metrics['accuracy'],
                'f1_diff': f1_diff,
                'f1_diff_pct': f1_diff * 100,
                'accuracy_diff': accuracy_diff,
                'accuracy_diff_pct': accuracy_diff * 100,
                'importance_rank': 0  # Will be assigned later
            }
            month_impacts.append(month_impact)

            print(f"Results with {month} disabled:")
            print(f"  Accuracy: {disabled_metrics['accuracy']:.4f}")
            print(f"  F1 (class 1): {disabled_metrics['f1_class1']:.4f}")
            print(f"  F1 (class 0): {disabled_metrics['f1_class0']:.4f}")
            print(f"  Loss: {disabled_metrics['loss']:.4f}")
            print(f"  F1 Change: {f1_diff:.4f} ({'+' if f1_diff >= 0 else ''}{f1_diff * 100:.2f}%)")
            print("-" * 60)

        results_df = pd.DataFrame(all_test_results)
        results_df.to_csv(self.output_dir / 'test_metrics_peft_ablation.csv', index=False)

        ranked_months = sorted(month_impacts, key=lambda x: abs(x['f1_diff']), reverse=True)
        for rank, month_data in enumerate(ranked_months):
            month_data['importance_rank'] = rank + 1

        month_impacts_df = pd.DataFrame(month_impacts)
        month_impacts_df.to_csv(self.output_dir / 'month_impact_analysis.csv', index=False)

        print("\nMonth importance based on F1 score impact when disabled:")
        for month_data in ranked_months:
            print(f"  {month_data['month']}: {abs(month_data['f1_diff']):.4f} impact " +
                  f"({'positive' if month_data['f1_diff'] < 0 else 'negative'})")

        return results_df, month_impacts_df

    def xai_band_occlusion_test(self, occlusion_mode='zero'):
        """
        Test the model by occluding each band individually and analyzing the impact on predictions.
        Creates CSV files with XAI values for each pixel and each band.

        Args:
            occlusion_mode: How to occlude bands - 'zero' (set to 0) or 'avg' (set to dataset average)

        XAI value calculation:
        - If ground truth is 0: XAI = f- - f+
        - If ground truth is 1: XAI = f+ - f-

        The positive XAI means important band

        Where:
        - f+ is the sigmoid output with all bands
        - f- is the sigmoid output with the specific band occluded (set to 0 or avg)
        """
        if not self.test_dataset:
            print("No test dataset provided")
            return

        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"

        best_hyperparams = self.load_best_model()
        batch_size = int(best_hyperparams['batch_size'])

        xai_output_dir = os.path.join(self.output_dir, f'xai_band_occlusion{mode_suffix}')
        os.makedirs(xai_output_dir, exist_ok=True)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers_dl,
            pin_memory=True
        )

        self.model.eval()

        band_names = self.test_dataset.new_channels if hasattr(self.test_dataset, 'new_channels') else [f"B{i}" for i in
                                                                                                        range(10)]

        if self.temporal_mode == TemporalMode.TIMESERIES.value:
            num_bands = self.test_dataset[0]['image'].shape[0]
        elif self.temporal_mode == TemporalMode.SINGLE.value:
            num_bands = self.test_dataset[0]['image'].shape[0]
        else:
            raise ValueError(f"Unsupported temporal mode: {self.temporal_mode}")

        print(f"Processing band occlusion analysis for {num_bands} bands: {band_names}")
        print(f"Occlusion mode: {occlusion_mode}")

        band_averages = None
        if occlusion_mode == 'avg':
            print("Calculating band averages across the dataset...")
            band_averages = self._calculate_band_averages()
            print(f"Band averages: {band_averages}")

        image_tile_registry = defaultdict(list)

        for batch in tqdm(self.test_loader, desc="Registering image tiles"):
            batch['position'] = transform_batch_positions(batch['position'])
            for i, image_id in enumerate(batch['image_id']):
                tile_info = {
                    'position': batch['position'][i],
                    'valid_mask': batch['valid_mask'][i],
                    'true_mask': batch['mask'][i]
                }

                if self.temporal_mode == TemporalMode.TIMESERIES.value:
                    tile_info['image'] = batch['image'][i]
                    tile_info['additional_images'] = batch['additional_images'][i]
                elif self.temporal_mode == TemporalMode.SINGLE.value:
                    tile_info['image'] = batch['image'][i]

                image_tile_registry[image_id].append(tile_info)

        for image_id, tiles in tqdm(image_tile_registry.items(), desc="Processing images"):
            sorted_tiles = sorted(tiles, key=lambda x: x['position'])

            max_height = max(tile['position'][0] + tile['valid_mask'].shape[1] for tile in sorted_tiles)
            max_width = max(tile['position'][1] + tile['valid_mask'].shape[2] for tile in sorted_tiles)

            if self.temporal_mode == TemporalMode.TIMESERIES.value:
                def process_batch(batch_tiles, occlude_band=None, occlude_idx=None):
                    batch_images = torch.stack([t['image'] for t in batch_tiles]).to(self.device)
                    batch_additional_images = torch.stack([t['additional_images'] for t in batch_tiles]).to(self.device)

                    if occlude_band is not None and occlude_idx is not None:
                        occlusion_value = 0
                        if occlusion_mode == 'avg' and band_averages is not None:
                            occlusion_value = band_averages['main'][occlude_idx]

                        batch_images[:, occlude_idx] = occlusion_value

                        for i in range(batch_additional_images.shape[1]):
                            if occlusion_mode == 'avg' and band_averages is not None:
                                occlusion_value = band_averages['additional'][i][occlude_idx]
                            batch_additional_images[:, i, occlude_idx] = occlusion_value

                    with torch.no_grad():
                        preds = self.model(batch_images, batch_additional_images)
                        return torch.sigmoid(preds).cpu()

            elif self.temporal_mode == TemporalMode.SINGLE.value:
                def process_batch(batch_tiles, occlude_band=None, occlude_idx=None):
                    batch_images = torch.stack([t['image'] for t in batch_tiles]).to(self.device)

                    if occlude_band is not None and occlude_idx is not None:
                        occlusion_value = 0
                        if occlusion_mode == 'avg' and band_averages is not None:
                            occlusion_value = band_averages['main'][occlude_idx]

                        batch_images[:, occlude_idx] = occlusion_value

                    with torch.no_grad():
                        preds = self.model(batch_images)
                        return torch.sigmoid(preds).cpu()


            all_bands_pred_tiles = []
            for tile_batch in chunked(sorted_tiles, batch_size):
                all_bands_pred_tiles.extend(process_batch(tile_batch))

            reconstructed_all_bands_pred = self._reconstruct_image(
                all_bands_pred_tiles,
                [t['position'] for t in sorted_tiles],
                [t['valid_mask'] for t in sorted_tiles],
                max_height,
                max_width
            )

            reconstructed_true_mask = self._reconstruct_image(
                [t['true_mask'] for t in sorted_tiles],
                [t['position'] for t in sorted_tiles],
                [t['valid_mask'] for t in sorted_tiles],
                max_height,
                max_width
            )

            pixel_data = []

            for band_idx in range(num_bands):
                band_name = band_names[band_idx]
                print(f"Processing image {image_id}, occluding band {band_idx} ({band_name})")

                occluded_pred_tiles = []
                for tile_batch in chunked(sorted_tiles, batch_size):
                    occluded_pred_tiles.extend(process_batch(tile_batch, occlude_band=True, occlude_idx=band_idx))

                reconstructed_occluded_pred = self._reconstruct_image(
                    occluded_pred_tiles,
                    [t['position'] for t in sorted_tiles],
                    [t['valid_mask'] for t in sorted_tiles],
                    max_height,
                    max_width
                )

                valid_mask = reconstructed_true_mask > -1

                valid_coords = torch.nonzero(valid_mask.squeeze())

                for y, x in valid_coords:
                    f_plus = float(reconstructed_all_bands_pred[0, y, x])
                    f_minus = float(reconstructed_occluded_pred[0, y, x])
                    ground_truth = int(reconstructed_true_mask[0, y, x] > 0.5)

                    if ground_truth == 0:
                        xai_value = f_minus - f_plus
                    else:
                        xai_value = f_plus - f_minus

                    pixel_data.append({
                        'row_idx': int(y),
                        'col_idx': int(x),
                        'band_idx': band_idx,
                        'band_name': band_name,
                        'f_plus': f_plus,
                        'f_minus': f_minus,
                        'xai_value': xai_value,
                        'ground_truth': ground_truth,
                        'predicted_value': int(f_plus > 0.5),
                        'predicted_value_occluded': int(f_minus > 0.5)
                    })

                del reconstructed_occluded_pred
                torch.cuda.empty_cache()

            df = pd.DataFrame(pixel_data)

            csv_path = os.path.join(xai_output_dir, f'{image_id}_xai_band_occlusion.csv')
            df.to_csv(csv_path, index=False)

            pivot_df = pd.DataFrame()

            pivot_df['row_idx'] = df['row_idx'].drop_duplicates().sort_values()
            pivot_df['col_idx'] = df['col_idx'].drop_duplicates().sort_values()

            unique_pixels = df[['row_idx', 'col_idx', 'ground_truth', 'predicted_value']].drop_duplicates()

            pivot_df = unique_pixels.copy()

            for band_idx, band_name in enumerate(band_names):
                band_data = df[df['band_idx'] == band_idx]

                band_columns = {
                    'f_plus': f'f_plus_{band_name}',
                    'f_minus': f'f_minus_{band_name}',
                    'xai_value': f'xai_value_{band_name}',
                    'predicted_value_occluded': f'predicted_value_occluded_{band_name}'
                }

                band_pivot = band_data[
                    ['row_idx', 'col_idx', 'f_plus', 'f_minus', 'xai_value', 'predicted_value_occluded']].rename(
                    columns=band_columns
                )

                pivot_df = pd.merge(pivot_df, band_pivot, on=['row_idx', 'col_idx'])

            pivot_csv_path = os.path.join(xai_output_dir, f'{image_id}_xai_band_occlusion_pivot.csv')
            pivot_df.to_csv(pivot_csv_path, index=False)

            del reconstructed_all_bands_pred, reconstructed_true_mask
            torch.cuda.empty_cache()

        print(f"XAI band occlusion analysis ({occlusion_mode} mode) completed. Results saved to {xai_output_dir}")
        return xai_output_dir

    def analyze_xai_data(self, occlusion_mode='zero'):
        """
        Main function to analyze XAI data and create visualizations.
        Creates all requested visualizations in the xai_explanation directory.
        """
        # Create output directory
        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"

        xai_explanation_dir = os.path.join(self.output_dir, f'xai_explanation{mode_suffix}')
        os.makedirs(xai_explanation_dir, exist_ok=True)

        xai_source_dir = os.path.join(self.output_dir, f'xai_band_occlusion{mode_suffix}')
        if not os.path.exists(xai_source_dir):
            print(f"XAI data directory not found: {xai_source_dir}")
            return

        csv_files = [f for f in os.listdir(xai_source_dir) if f.endswith('_pivot.csv')]
        if not csv_files:
            print(f"No XAI data files found in {xai_source_dir}")
            return

        all_data = []
        for csv_file in tqdm(csv_files, desc="Loading XAI data"):
            file_path = os.path.join(xai_source_dir, csv_file)
            df = pd.read_csv(file_path)
            image_id = csv_file.split('_xai_band_occlusion_pivot.csv')[0]
            df['image_id'] = image_id
            all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded data for {len(csv_files)} scenes with {len(combined_data)} total pixels")

        band_names = []
        for col in combined_data.columns:
            if col.startswith('xai_value_'):
                band_name = col.replace('xai_value_', '')
                band_names.append(band_name)

        print(f"Detected {len(band_names)} bands: {band_names}")

        # TODO UNCOMMENT THE VISUALIZATIONS YOU WANT TO CREATE
        # ROMANIA
        ts_dir = os.path.join(xai_explanation_dir, '..', '..', 'timeseries_september_try_12')
        # CZ
        # ts_dir = os.path.join(xai_explanation_dir, '..', '..', 'timeseries_sept-apr_no_lora')

        self._create_band_statistics_table(combined_data, band_names, xai_explanation_dir, occlusion_mode, ts_dir)
        self._create_xai_boxplots(combined_data, band_names, xai_explanation_dir)
        self._create_f_plus_minus_boxplots(combined_data, band_names, xai_explanation_dir)
        self._create_model_comparison_scatterplots(xai_explanation_dir, occlusion_mode)
        self._create_band_average_comparison(xai_explanation_dir, occlusion_mode)

        # WITHOUT COUNTOURS
        self._create_xai_heatmaps(xai_source_dir, band_names, xai_explanation_dir)

        self._create_prediction_scatterplots(combined_data, band_names, xai_explanation_dir)
        self._create_consolidated_scatter_plot(combined_data, band_names, xai_explanation_dir)
        self._create_individual_consolidated_scatter_plots(combined_data, band_names, xai_explanation_dir)

        self.create_enhanced_xai_visualization(xai_explanation_dir, occlusion_mode)
        self.create_polarized_xai_visualization(xai_explanation_dir, occlusion_mode)


        '''band_names = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]
        enhanced_dir = os.path.join(xai_explanation_dir, 'enhanced_heatmaps')
        scene_dirs = os.listdir(os.path.join(xai_explanation_dir, 'enhanced_heatmaps'))
        self._create_html_viewer_with_contours(xai_explanation_dir, enhanced_dir, band_names, scene_dirs)'''

        print(f"XAI analysis complete. Results saved to {xai_explanation_dir}")

    def _create_band_statistics_table(self, combined_data, band_names, output_dir, occlusion_mode, comparison_model_dir=None):
        """
        Create a table with mean and standard deviation for each band's XAI values
        across all pixels and split by ground truth class (GT=0 vs GT=1).
        Also creates bar charts showing mean XAI values for each configuration.

        Args:
            combined_data: DataFrame with XAI data for current model
            band_names: List of band names
            output_dir: Directory to save outputs
            comparison_model_dir: Optional path to another model's output directory for comparison
        """
        print("Generating band statistics table...")

        min_xai = float('inf')
        max_xai = float('-inf')

        for band in band_names:
            xai_col = f'xai_value_{band}'
            min_val = combined_data[xai_col].min()
            max_val = combined_data[xai_col].max()
            min_xai = min(min_xai, min_val)
            max_xai = max(max_xai, max_val)

        print(f"Global XAI range: [{min_xai:.4f}, {max_xai:.4f}]")

        normalized_data = combined_data.copy()
        xai_range = max_xai - min_xai

        if xai_range <= 1e-10:
            print("Warning: XAI range is too small, skipping normalization")
        else:
            for band in band_names:
                xai_col = f'xai_value_{band}'
                normalized_data[xai_col] = (combined_data[xai_col] - min_xai) / xai_range

        stats_all = []
        stats_gt0 = []
        stats_gt1 = []

        for band in band_names:
            xai_col = f'xai_value_{band}'
            f_plus_col = f'f_plus_{band}'
            f_minus_col = f'f_minus_{band}'

            xai_mean = normalized_data[xai_col].mean()
            xai_std = normalized_data[xai_col].std()
            f_plus_mean = combined_data[f_plus_col].mean()
            f_plus_std = combined_data[f_plus_col].std()
            f_minus_mean = combined_data[f_minus_col].mean()
            f_minus_std = combined_data[f_minus_col].std()

            stats_all.append({
                'Band': band,
                'XAI_Mean': xai_mean,
                'XAI_Std': xai_std,
                'f_plus_Mean': f_plus_mean,
                'f_plus_Std': f_plus_std,
                'f_minus_Mean': f_minus_mean,
                'f_minus_Std': f_minus_std
            })

            gt0_data = normalized_data[combined_data['ground_truth'] == 0]
            xai_mean_gt0 = gt0_data[xai_col].mean()
            xai_std_gt0 = gt0_data[xai_col].std()
            f_plus_mean_gt0 = combined_data.loc[combined_data['ground_truth'] == 0, f_plus_col].mean()
            f_plus_std_gt0 = combined_data.loc[combined_data['ground_truth'] == 0, f_plus_col].std()
            f_minus_mean_gt0 = combined_data.loc[combined_data['ground_truth'] == 0, f_minus_col].mean()
            f_minus_std_gt0 = combined_data.loc[combined_data['ground_truth'] == 0, f_minus_col].std()

            stats_gt0.append({
                'Band': band,
                'XAI_Mean': xai_mean_gt0,
                'XAI_Std': xai_std_gt0,
                'f_plus_Mean': f_plus_mean_gt0,
                'f_plus_Std': f_plus_std_gt0,
                'f_minus_Mean': f_minus_mean_gt0,
                'f_minus_Std': f_minus_std_gt0
            })

            gt1_data = normalized_data[combined_data['ground_truth'] == 1]
            xai_mean_gt1 = gt1_data[xai_col].mean()
            xai_std_gt1 = gt1_data[xai_col].std()
            f_plus_mean_gt1 = combined_data.loc[combined_data['ground_truth'] == 1, f_plus_col].mean()
            f_plus_std_gt1 = combined_data.loc[combined_data['ground_truth'] == 1, f_plus_col].std()
            f_minus_mean_gt1 = combined_data.loc[combined_data['ground_truth'] == 1, f_minus_col].mean()
            f_minus_std_gt1 = combined_data.loc[combined_data['ground_truth'] == 1, f_minus_col].std()

            stats_gt1.append({
                'Band': band,
                'XAI_Mean': xai_mean_gt1,
                'XAI_Std': xai_std_gt1,
                'f_plus_Mean': f_plus_mean_gt1,
                'f_plus_Std': f_plus_std_gt1,
                'f_minus_Mean': f_minus_mean_gt1,
                'f_minus_Std': f_minus_std_gt1
            })

        stats_all_df = pd.DataFrame(stats_all).sort_values(by='XAI_Mean', ascending=False)
        stats_gt0_df = pd.DataFrame(stats_gt0).sort_values(by='XAI_Mean', ascending=False)
        stats_gt1_df = pd.DataFrame(stats_gt1).sort_values(by='XAI_Mean', ascending=False)

        stats_all_df.to_csv(os.path.join(output_dir, 'normalized_band_statistics_all.csv'), index=False)
        stats_gt0_df.to_csv(os.path.join(output_dir, 'normalized_band_statistics_gt0.csv'), index=False)
        stats_gt1_df.to_csv(os.path.join(output_dir, 'normalized_band_statistics_gt1.csv'), index=False)

        combined_stats = []
        for band in band_names:
            all_row = stats_all_df[stats_all_df['Band'] == band].iloc[0]
            gt0_row = stats_gt0_df[stats_gt0_df['Band'] == band].iloc[0]
            gt1_row = stats_gt1_df[stats_gt1_df['Band'] == band].iloc[0]

            combined_stats.append({
                'Band': band,
                'All_XAI_Mean': all_row['XAI_Mean'],
                'All_XAI_Std': all_row['XAI_Std'],
                'GT0_XAI_Mean': gt0_row['XAI_Mean'],
                'GT0_XAI_Std': gt0_row['XAI_Std'],
                'GT1_XAI_Mean': gt1_row['XAI_Mean'],
                'GT1_XAI_Std': gt1_row['XAI_Std']
            })

        combined_stats_df = pd.DataFrame(combined_stats)
        combined_stats_df = combined_stats_df.sort_values(by='All_XAI_Mean', ascending=False)
        combined_stats_df.to_csv(os.path.join(output_dir, 'normalized_band_statistics.csv'), index=False)

        self._create_xai_bar_charts(stats_all_df, stats_gt0_df, stats_gt1_df, output_dir)

        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"
        rank_change_dict = {}
        if comparison_model_dir and os.path.exists(comparison_model_dir):
            comparison_stats_path = os.path.join(comparison_model_dir, f'xai_explanation{mode_suffix}',
                                                 'normalized_band_statistics.csv')
            if not os.path.exists(comparison_stats_path):
                comparison_stats_path = os.path.join(comparison_model_dir, f'xai_explanation{mode_suffix}', 'band_statistics.csv')

            if os.path.exists(comparison_stats_path):
                comparison_stats_df = pd.read_csv(comparison_stats_path)
                comparison_model_ranks = comparison_stats_df.sort_values(by='All_XAI_Mean',
                                                                         ascending=False).reset_index(drop=True)
                comparison_ranks = {row['Band']: idx for idx, (_, row) in enumerate(comparison_model_ranks.iterrows())}
                current_ranks = {row['Band']: idx for idx, (_, row) in enumerate(combined_stats_df.iterrows())}

                for band in band_names:
                    current_rank = current_ranks.get(band, float('inf'))
                    comparison_rank = comparison_ranks.get(band, float('inf'))
                    rank_change = comparison_rank - current_rank

                    if rank_change > 0:
                        change_text = f"↑ {abs(rank_change)}"
                    elif rank_change < 0:
                        change_text = f"↓ {abs(rank_change)}"
                    else:
                        change_text = "="

                    rank_change_dict[band] = {
                        'change': rank_change,
                        'text': change_text
                    }

        fig, ax = plt.figure(figsize=(15, len(band_names) * 1.2 + 3)), plt.gca()
        ax.set_axis_off()

        max_all_xai_band = stats_all_df.iloc[0]['Band']
        max_gt0_xai_band = stats_gt0_df.iloc[0]['Band']
        max_gt1_xai_band = stats_gt1_df.iloc[0]['Band']

        table_data = []

        headers = ['Band', 'All Pixels XAI Mean±Std', 'GT=0 XAI Mean±Std', 'GT=1 XAI Mean±Std']
        if rank_change_dict:
            headers.append('Rank\nChange')
        table_data.append(headers)

        for _, row in combined_stats_df.iterrows():
            band = row['Band']
            all_val = f"{row['All_XAI_Mean']:.3f}±{row['All_XAI_Std']:.3f}"
            gt0_val = f"{row['GT0_XAI_Mean']:.3f}±{row['GT0_XAI_Std']:.3f}"
            gt1_val = f"{row['GT1_XAI_Mean']:.3f}±{row['GT1_XAI_Std']:.3f}"

            row_data = [band, all_val, gt0_val, gt1_val]

            if rank_change_dict and band in rank_change_dict:
                row_data.append(rank_change_dict[band]['text'])
            elif rank_change_dict:
                row_data.append("")

            table_data.append(row_data)

        table = Table(ax, bbox=[0, 0, 1, 1])

        n_rows, n_cols = len(table_data), len(table_data[0])

        col_widths = [0.15] + [0.25] * (n_cols - 2) + [0.10] if rank_change_dict else [0.15] + [0.28] * (n_cols - 1)
        cell_heights = [0.10] + [0.90 / (n_rows - 1)] * (n_rows - 1)

        if rank_change_dict:
            max_rank_change = max(
                [abs(data['change']) for data in rank_change_dict.values()]) if rank_change_dict else 1
            max_rank_change = max(1, max_rank_change)  # Avoid division by zero

        for i in range(n_rows):
            for j in range(n_cols):
                if i == 0:
                    cell_color = '#D3D3D3'
                    cell_text_props = {'fontweight': 'bold', 'fontsize': 14}
                else:
                    cell_color = 'white'
                    cell_text_props = {'fontsize': 20}

                    if j == 1 and table_data[i][0] == max_all_xai_band:
                        cell_text_props = {'fontweight': 'bold', 'color': 'red', 'fontsize': 20}
                    elif j == 2 and table_data[i][0] == max_gt0_xai_band:
                        cell_text_props = {'fontweight': 'bold', 'color': 'red', 'fontsize': 20}
                    elif j == 3 and table_data[i][0] == max_gt1_xai_band:
                        cell_text_props = {'fontweight': 'bold', 'color': 'red', 'fontsize': 20}

                    if rank_change_dict and j == 4 and i > 0:
                        band = table_data[i][0]
                        if band in rank_change_dict:
                            rank_change = rank_change_dict[band]['change']
                            if rank_change > 0:
                                intensity = min(1.0, abs(rank_change) / max_rank_change)
                                cell_color = (1 - 0.5 * intensity, 1, 1 - 0.5 * intensity)
                                cell_text_props = {'fontweight': 'bold', 'color': 'green', 'fontsize': 20}
                            elif rank_change < 0:
                                intensity = min(1.0, abs(rank_change) / max_rank_change)
                                cell_color = (1, 1 - 0.5 * intensity, 1 - 0.5 * intensity)
                                cell_text_props = {'fontweight': 'bold', 'color': 'red', 'fontsize': 20}

                table.add_cell(i, j, col_widths[j], cell_heights[i], text=table_data[i][j],
                               loc='center', facecolor=cell_color, edgecolor='black')

                table[(i, j)].get_text().set(**cell_text_props)

        ax.add_table(table)
        plt.title('Band XAI Statistics by Class (Normalized Values, Sorted by XAI Value)', fontsize=16)

        plt_path = os.path.join(output_dir, 'normalized_band_statistics_table.png')
        plt.savefig(plt_path, dpi=100, bbox_inches='tight')
        plt.close()

        if comparison_model_dir and os.path.exists(comparison_model_dir):
            self._create_band_ranking_comparison_table(combined_stats_df, band_names, output_dir, comparison_model_dir, occlusion_mode)

        print(f"Normalized band statistics saved to {output_dir} (CSVs and visualization)")
        return combined_stats_df

    def _create_band_ranking_comparison_table(self, current_model_stats, band_names, output_dir, comparison_model_dir, occlusion_mode):
        """
        Create a comparison table showing how band rankings differ between the current model
        and a comparison model.

        Args:
            current_model_stats: DataFrame with statistics for the current model
            band_names: List of band names
            output_dir: Directory to save outputs
            comparison_model_dir: Path to another model's output directory for comparison
        """
        print("Generating band ranking comparison table...")

        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"
        comparison_stats_path = os.path.join(comparison_model_dir, f'xai_explanation{mode_suffix}', 'normalized_band_statistics.csv')
        if not os.path.exists(comparison_stats_path):
            comparison_stats_path = os.path.join(comparison_model_dir, f'xai_explanation{mode_suffix}', 'band_statistics.csv')

        if not os.path.exists(comparison_stats_path):
            print(f"Warning: Could not find statistics file for comparison model at {comparison_stats_path}")
            return

        comparison_stats_df = pd.read_csv(comparison_stats_path)

        current_model_ranks = current_model_stats.sort_values(by='All_XAI_Mean', ascending=False).reset_index(drop=True)
        comparison_model_ranks = comparison_stats_df.sort_values(by='All_XAI_Mean', ascending=False).reset_index(
            drop=True)

        current_ranks = {row['Band']: idx for idx, (_, row) in enumerate(current_model_ranks.iterrows())}
        comparison_ranks = {row['Band']: idx for idx, (_, row) in enumerate(comparison_model_ranks.iterrows())}

        comparison_data = []
        for band in band_names:
            current_rank = current_ranks.get(band, float('inf'))
            comparison_rank = comparison_ranks.get(band, float('inf'))
            rank_change = comparison_rank - current_rank

            comparison_data.append({
                'Band': band,
                'Current_Model_Rank': current_rank + 1,
                'Comparison_Model_Rank': comparison_rank + 1,
                'Rank_Change': rank_change,
                'Current_Model_XAI': current_model_ranks[current_model_ranks['Band'] == band]['All_XAI_Mean'].values[0],
                'Comparison_Model_XAI':
                    comparison_model_ranks[comparison_model_ranks['Band'] == band]['All_XAI_Mean'].values[
                        0] if band in comparison_ranks else 0
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(by='Current_Model_Rank')

        comparison_df.to_csv(os.path.join(output_dir, 'band_ranking_comparison.csv'), index=False)

        fig, ax = plt.figure(figsize=(16, len(band_names) * 1.2 + 3)), plt.gca()
        ax.set_axis_off()

        table_data = []

        model_name = os.path.basename(output_dir).split('/')[0]
        comparison_model_name = os.path.basename(comparison_model_dir)

        headers = ['Band', f'{model_name}\nRank', f'{comparison_model_name}\nRank', 'Rank\nChange',
                   f'{model_name}\nXAI Value', f'{comparison_model_name}\nXAI Value']
        table_data.append(headers)

        for _, row in comparison_df.iterrows():
            band = row['Band']
            current_rank = int(row['Current_Model_Rank'])
            comparison_rank = int(row['Comparison_Model_Rank'])
            rank_change = int(row['Rank_Change'])

            if rank_change > 0:
                change_text = f"↑ {abs(rank_change)}"
            elif rank_change < 0:
                change_text = f"↓ {abs(rank_change)}"
            else:
                change_text = "="

            current_xai = f"{row['Current_Model_XAI']:.3f}"
            comparison_xai = f"{row['Comparison_Model_XAI']:.3f}"

            table_data.append([band, current_rank, comparison_rank, change_text, current_xai, comparison_xai])

        table = Table(ax, bbox=[0, 0, 1, 1])

        n_rows, n_cols = len(table_data), len(table_data[0])

        col_widths = [0.15, 0.15, 0.15, 0.15, 0.20, 0.20]
        cell_heights = [0.10] + [0.90 / (n_rows - 1)] * (n_rows - 1)

        max_rank_change = max(
            [abs(row['Rank_Change']) for _, row in comparison_df.iterrows()]) if not comparison_df.empty else 1
        max_rank_change = max(1, max_rank_change)

        for i in range(n_rows):
            for j in range(n_cols):
                if i == 0:
                    cell_color = '#D3D3D3'
                    cell_text_props = {'fontweight': 'bold', 'fontsize': 14}
                else:
                    cell_color = 'white'
                    cell_text_props = {'fontsize': 20}

                    if j == 3 and i > 0:
                        rank_change = comparison_df.iloc[i - 1]['Rank_Change']
                        if rank_change > 0:

                            intensity = min(1.0, abs(rank_change) / max_rank_change)
                            cell_color = (1 - 0.5 * intensity, 1, 1 - 0.5 * intensity)
                            cell_text_props = {'fontweight': 'bold', 'color': 'green', 'fontsize': 20}
                        elif rank_change < 0:

                            intensity = min(1.0, abs(rank_change) / max_rank_change)
                            cell_color = (1, 1 - 0.5 * intensity, 1 - 0.5 * intensity)
                            cell_text_props = {'fontweight': 'bold', 'color': 'red', 'fontsize': 20}

                table.add_cell(i, j, col_widths[j], cell_heights[i], text=table_data[i][j],
                               loc='center', facecolor=cell_color, edgecolor='black')

                table[(i, j)].get_text().set(**cell_text_props)

        ax.add_table(table)
        plt.title(f'Band Ranking Comparison: {model_name} vs {comparison_model_name}', fontsize=16)

        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(0.5, 1, 0.5), markersize=15,
                       label='Better rank in current model (↑)'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(1, 0.5, 0.5), markersize=15,
                       label='Worse rank in current model (↓)')
        ]

        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=2, fontsize=12)

        plt_path = os.path.join(output_dir, 'band_ranking_comparison_table.png')
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Band ranking comparison table saved to {plt_path}")

    def _calculate_band_averages(self):
        """
        Calculate average values for each band across the dataset.
        Returns a dictionary with averages for main images and additional images if applicable.
        """
        print("Calculating band averages across the dataset...")

        band_sums = {}
        band_counts = {}

        if self.temporal_mode == TemporalMode.TIMESERIES.value:
            band_sums['main'] = None
            band_counts['main'] = 0
            band_sums['additional'] = []
            band_counts['additional'] = []
        elif self.temporal_mode == TemporalMode.SINGLE.value:
            band_sums['main'] = None
            band_counts['main'] = 0

        for i, item in enumerate(tqdm(self.test_dataset, desc="Computing band averages")):
            if self.temporal_mode == TemporalMode.TIMESERIES.value or self.temporal_mode == TemporalMode.SINGLE.value:
                image = item['image']
                valid_mask = item['valid_mask']

                valid_pixels = valid_mask > 0

                if band_sums['main'] is None:
                    band_sums['main'] = torch.zeros(image.size(0), dtype=torch.float32)

                for band_idx in range(image.size(0)):
                    if len(valid_mask.shape) == 2:
                        band_values = image[band_idx][valid_pixels]
                    else:
                        band_values = image[band_idx][valid_pixels.squeeze()]

                    band_sums['main'][band_idx] += band_values.sum().item()
                    band_counts['main'] += band_values.numel()

                if self.temporal_mode == TemporalMode.TIMESERIES.value:
                    additional_images = item['additional_images']

                    while len(band_sums['additional']) < additional_images.size(0):
                        band_sums['additional'].append(torch.zeros(additional_images.size(1), dtype=torch.float32))
                        band_counts['additional'].append(0)

                    for img_idx in range(additional_images.size(0)):
                        for band_idx in range(additional_images.size(1)):
                            if len(valid_mask.shape) == 2:
                                band_values = additional_images[img_idx, band_idx][valid_pixels]
                            else:
                                band_values = additional_images[img_idx, band_idx][valid_pixels.squeeze()]

                            band_sums['additional'][img_idx][band_idx] += band_values.sum().item()
                            band_counts['additional'][img_idx] += band_values.numel()

        band_averages = {}

        if self.temporal_mode == TemporalMode.TIMESERIES.value:
            if band_counts['main'] > 0:
                band_averages['main'] = band_sums['main'] / band_counts['main']
            else:
                band_averages['main'] = torch.zeros_like(band_sums['main'])

            band_averages['additional'] = []
            for img_idx, img_sum in enumerate(band_sums['additional']):
                if band_counts['additional'][img_idx] > 0:
                    band_averages['additional'].append(img_sum / band_counts['additional'][img_idx])
                else:
                    band_averages['additional'].append(torch.zeros_like(img_sum))

        elif self.temporal_mode == TemporalMode.SINGLE.value:
            if band_counts['main'] > 0:
                band_averages['main'] = band_sums['main'] / band_counts['main']
            else:
                band_averages['main'] = torch.zeros_like(band_sums['main'])

        for key, value in band_averages.items():
            if isinstance(value, torch.Tensor):
                print(f"Average for {key}: {value.tolist()}")
            elif isinstance(value, list):
                print(f"Average for {key} (list of {len(value)} items)")
                for i, item in enumerate(value):
                    print(f"  Item {i}: {item.tolist()}")

        return band_averages

    def _create_xai_bar_charts(self, stats_all_df, stats_gt0_df, stats_gt1_df, output_dir):
        """
        Create bar charts showing mean XAI values for each band, sorted in descending order.
        One chart for each configuration: all pixels, GT=0 pixels, GT=1 pixels.
        """
        print("Generating XAI bar charts...")

        configs = [
            {'name': 'all', 'title': 'All Pixels', 'data': stats_all_df},
            {'name': 'gt0', 'title': 'GT=0 Pixels (Healthy)', 'data': stats_gt0_df},
            {'name': 'gt1', 'title': 'GT=1 Pixels (Damaged)', 'data': stats_gt1_df}
        ]

        for config in configs:
            plt.figure(figsize=(10, 6))

            data = config['data']
            bands = data['Band'].values
            xai_means = data['XAI_Mean'].values

            bars = plt.bar(bands, xai_means, capsize=5,
                           color='skyblue', edgecolor='black', linewidth=1)

            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            plt.xlabel('Bands', fontsize=12)
            plt.ylabel('XAI Mean Value', fontsize=12)
            plt.title(f'Mean XAI Values by Band for {config["title"]}', fontsize=14)

            plt.grid(axis='y', linestyle='--', alpha=0.7)

            '''# Add values on top of the bars
            for bar, value, std in zip(bars, xai_means, xai_stds):
                height = bar.get_height()
                if height >= 0:
                    va = 'bottom'
                    y_pos = height + 0.01
                else:
                    va = 'top'
                    y_pos = height - 0.01
                plt.text(bar.get_x() + bar.get_width() / 2, y_pos,
                         f'{value:.3f}', ha='center', va=va,
                         rotation=0, fontsize=9, fontweight='bold')'''

            plt.tight_layout()

            plt_path = os.path.join(output_dir, f'xai_barchart_{config["name"]}.png')
            plt.savefig(plt_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"XAI bar charts saved to {output_dir}")

    def _create_xai_boxplots(self, combined_data, band_names, output_dir):
        """
        Create box plots of XAI values for each band across all scenes.
        For each band, show two boxplots side by side - one for GT=0 (healthy) and one for GT=1 (damaged).
        Create separate plots for XAI values, f+ and f-.
        Shows 25-75 percentile range without outliers.
        """
        print("Generating XAI value box plots with ground truth split...")
        sns.set(style="whitegrid")

        plt.figure(figsize=(18, 10))

        positions = []
        box_data = []
        colors = []
        labels = []
        tick_positions = []

        for i, band in enumerate(band_names):
            healthy_xai = combined_data[combined_data['ground_truth'] == 0][f'xai_value_{band}'].dropna()
            damaged_xai = combined_data[combined_data['ground_truth'] == 1][f'xai_value_{band}'].dropna()

            box_data.append(healthy_xai)
            colors.append('lightgreen')
            positions.append(i * 3)

            box_data.append(damaged_xai)
            colors.append('lightcoral')
            positions.append(i * 3 + 1)

            labels.append(band)
            tick_positions.append(i * 3 + 0.5)

        box = plt.boxplot(box_data, positions=positions, patch_artist=True,
                          showfliers=False, widths=0.8)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.title('Distribution of XAI Values by Band and Ground Truth Class', fontsize=16)
        plt.xlabel('Band', fontsize=14)
        plt.ylabel('XAI Value', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        plt.gca().set_xticks(tick_positions)
        plt.gca().set_xticklabels(labels)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Healthy (GT=0)'),
            Patch(facecolor='lightcoral', label='Damaged (GT=1)')
        ]
        plt.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        xai_boxplot_path = os.path.join(output_dir, 'xai_values_boxplot_by_class.png')
        plt.savefig(xai_boxplot_path, dpi=300)
        plt.close()


    def _create_model_comparison_scatterplots(self, output_dir, occlusion_mode):
        """
        Create scatter plots comparing XAI values between ULISSE and TS models for each band.
        Pixels are split by ground truth class (GT=0 vs GT=1).

        Creates:
        1. Individual scatter plots for each band and class
        2. Two consolidated plots (one per class) with all bands
        """
        print("Generating model comparison scatter plots...")
        print("Note: This function requires XAI data from both models to be available.")
        print("If you haven't processed both models yet, this function will be skipped.")

        parent_dir = os.path.dirname(self.output_dir)

        ulisse_dir = os.path.join(output_dir, '..')
        # ts_dir = os.path.join(output_dir, '..', '..', 'timeseries_sept-apr_no_lora')
        # WARNING set the directory name according to the TS model you want to compare with
        ts_dir = os.path.join(output_dir, '..', '..', 'timeseries_september_try_12')

        if not os.path.exists(ulisse_dir) or not os.path.exists(ts_dir):
            print(f"Could not find data directories for both models.")
            print(f"Looking for ULISSE in: {ulisse_dir}")
            print(f"Looking for TS in: {ts_dir}")
            return

        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"
        ulisse_xai_dir = os.path.join(ulisse_dir, f'xai_band_occlusion{mode_suffix}')
        ts_xai_dir = os.path.join(ts_dir, f'xai_band_occlusion{mode_suffix}')

        if not os.path.exists(ulisse_xai_dir) or not os.path.exists(ts_xai_dir):
            print(f"Could not find XAI data directories for both models.")
            print(f"Looking for ULISSE XAI in: {ulisse_xai_dir}")
            print(f"Looking for TS XAI in: {ts_xai_dir}")
            return

        ulisse_files = {f.split('_xai_band_occlusion_pivot.csv')[0] for f in os.listdir(ulisse_xai_dir)
                        if f.endswith('_pivot.csv')}
        ts_files = {f.split('_xai_band_occlusion_pivot.csv')[0] for f in os.listdir(ts_xai_dir)
                    if f.endswith('_pivot.csv')}
        common_ids = ulisse_files.intersection(ts_files)

        if not common_ids:
            print("No common image IDs found between the two models.")
            return

        comparison_dir = os.path.join(output_dir, 'model_comparison_scatters')
        os.makedirs(comparison_dir, exist_ok=True)

        comparison_gt0_dir = os.path.join(comparison_dir, 'gt0_healthy')
        os.makedirs(comparison_gt0_dir, exist_ok=True)

        comparison_gt1_dir = os.path.join(comparison_dir, 'gt1_damaged')
        os.makedirs(comparison_gt1_dir, exist_ok=True)

        first_ulisse_file = os.path.join(ulisse_xai_dir, f"{next(iter(common_ids))}_xai_band_occlusion_pivot.csv")
        first_ts_file = os.path.join(ts_xai_dir, f"{next(iter(common_ids))}_xai_band_occlusion_pivot.csv")

        ulisse_df = pd.read_csv(first_ulisse_file)
        ts_df = pd.read_csv(first_ts_file)

        ulisse_bands = [col.replace('xai_value_', '') for col in ulisse_df.columns
                        if col.startswith('xai_value_')]
        ts_bands = [col.replace('xai_value_', '') for col in ts_df.columns
                    if col.startswith('xai_value_')]

        common_bands = sorted(set(ulisse_bands).intersection(set(ts_bands)))

        if not common_bands:
            print("No common bands found between the two models.")
            return

        print(f"Found {len(common_ids)} common images and {len(common_bands)} common bands.")
        print("Creating scatter plots for each band...")

        all_data = []
        for image_id in tqdm(common_ids, desc="Processing images for comparison"):
            ulisse_file = os.path.join(ulisse_xai_dir, f"{image_id}_xai_band_occlusion_pivot.csv")
            ts_file = os.path.join(ts_xai_dir, f"{image_id}_xai_band_occlusion_pivot.csv")

            if not os.path.exists(ulisse_file) or not os.path.exists(ts_file):
                continue

            ulisse_data = pd.read_csv(ulisse_file)
            ts_data = pd.read_csv(ts_file)

            merged_df = pd.merge(
                ulisse_data,
                ts_data,
                on=['row_idx', 'col_idx', 'ground_truth'],
                suffixes=('_ulisse', '_ts')
            )

            merged_df['image_id'] = image_id
            all_data.append(merged_df)

        if not all_data:
            print("No valid data found for comparison.")
            return

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data contains {len(combined_df)} pixels from {len(combined_df['image_id'].unique())} images.")

        df_gt0 = combined_df[combined_df['ground_truth'] == 0]
        df_gt1 = combined_df[combined_df['ground_truth'] == 1]

        print(f"GT=0 (healthy) pixels: {len(df_gt0)}")
        print(f"GT=1 (damaged) pixels: {len(df_gt1)}")

        n_bands = len(common_bands)
        n_cols = min(5, n_bands)
        n_rows = (n_bands + n_cols - 1) // n_cols

        fig_gt0 = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        fig_gt1 = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

        for i, band in enumerate(common_bands):
            print(f"Processing band {band}...")

            ulisse_xai_col = f'xai_value_{band}_ulisse'
            ts_xai_col = f'xai_value_{band}_ts'

            if not df_gt0.empty:
                plt.figure(figsize=(8, 8))
                plt.scatter(
                    df_gt0[ulisse_xai_col],
                    df_gt0[ts_xai_col],
                    alpha=0.5,
                    s=10,
                    c='green',
                    label=f'GT=0 (healthy)'
                )

                min_val = min(df_gt0[ulisse_xai_col].min(), df_gt0[ts_xai_col].min())
                max_val = max(df_gt0[ulisse_xai_col].max(), df_gt0[ts_xai_col].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

                plt.xlabel('ULISSE XAI Value', fontsize=12)
                plt.ylabel('TimeSeries XAI Value', fontsize=12)
                plt.title(f'Band {band} - GT=0 (Healthy)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                plt.savefig(os.path.join(comparison_gt0_dir, f'band_{band}_comparison_gt0.png'), dpi=200)
                plt.close()

                plt.figure(fig_gt0.number)
                plt.subplot(n_rows, n_cols, i + 1)
                plt.scatter(df_gt0[ulisse_xai_col], df_gt0[ts_xai_col], alpha=0.5, s=5, c='green')
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                plt.title(f'Band {band}')
                plt.grid(True, alpha=0.3)

            if not df_gt1.empty:
                plt.figure(figsize=(8, 8))
                plt.scatter(
                    df_gt1[ulisse_xai_col],
                    df_gt1[ts_xai_col],
                    alpha=0.5,
                    s=10,
                    c='red',
                    label=f'GT=1 (damaged)'
                )

                min_val = min(df_gt1[ulisse_xai_col].min(), df_gt1[ts_xai_col].min())
                max_val = max(df_gt1[ulisse_xai_col].max(), df_gt1[ts_xai_col].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

                plt.xlabel('ULISSE XAI Value', fontsize=12)
                plt.ylabel('TimeSeries XAI Value', fontsize=12)
                plt.title(f'Band {band} - GT=1 (Damaged)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()

                plt.savefig(os.path.join(comparison_gt1_dir, f'band_{band}_comparison_gt1.png'), dpi=200)
                plt.close()

                plt.figure(fig_gt1.number)
                plt.subplot(n_rows, n_cols, i + 1)
                plt.scatter(df_gt1[ulisse_xai_col], df_gt1[ts_xai_col], alpha=0.5, s=5, c='red')
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                plt.title(f'Band {band}')
                plt.grid(True, alpha=0.3)

        plt.figure(fig_gt0.number)
        plt.suptitle('Model Comparison: ULISSE vs TimeSeries - GT=0 (Healthy)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        plt.savefig(os.path.join(comparison_dir, 'consolidated_gt0_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_gt0)

        plt.figure(fig_gt1.number)
        plt.suptitle('Model Comparison: ULISSE vs TimeSeries - GT=1 (Damaged)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        plt.savefig(os.path.join(comparison_dir, 'consolidated_gt1_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_gt1)

        print(f"Model comparison scatter plots saved to {comparison_dir}")
        print(f"Individual GT=0 plots saved to {comparison_gt0_dir}")
        print(f"Individual GT=1 plots saved to {comparison_gt1_dir}")

    def _create_band_average_comparison(self, output_dir, occlusion_mode):
        """
        Create a scatter plot comparing average XAI values of each band between ULISSE and TS models.
        Each point represents a band, labeled with its name.
        """
        print("Generating band average comparison scatter plot...")

        parent_dir = os.path.dirname(self.output_dir)

        ulisse_dir = None
        ts_dir = None

        for dirname in os.listdir(parent_dir):
            dir_path = os.path.join(parent_dir, dirname)
            if os.path.isdir(dir_path):
                if 'ulisse' in dirname.lower():
                    ulisse_dir = dir_path
                elif 'timeseries' in dirname.lower() or '_ts' in dirname.lower():
                    ts_dir = dir_path

        if not ulisse_dir or not ts_dir:
            print("Could not find both model directories. Skipping band average comparison.")
            return

        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"
        ulisse_stats_path = os.path.join(ulisse_dir, f'xai_explanation{mode_suffix}', 'band_statistics.csv')
        ts_stats_path = os.path.join(ts_dir, f'xai_explanation{mode_suffix}', 'band_statistics.csv')

        if not os.path.exists(ulisse_stats_path) or not os.path.exists(ts_stats_path):
            print("Band statistics not found for both models. Skipping band average comparison.")
            return

        ulisse_stats = pd.read_csv(ulisse_stats_path)
        ts_stats = pd.read_csv(ts_stats_path)

        merged_stats = pd.merge(
            ulisse_stats, ts_stats,
            on='Band',
            suffixes=('_ulisse', '_ts')
        )

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))

        plt.scatter(
            merged_stats['XAI_Mean_ulisse'],
            merged_stats['XAI_Mean_ts'],
            s=100
        )

        for i, row in merged_stats.iterrows():
            plt.annotate(
                row['Band'],
                (row['XAI_Mean_ulisse'], row['XAI_Mean_ts']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=12
            )

        min_val = min(merged_stats['XAI_Mean_ulisse'].min(), merged_stats['XAI_Mean_ts'].min())
        max_val = max(merged_stats['XAI_Mean_ulisse'].max(), merged_stats['XAI_Mean_ts'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        plt.title('Average XAI Value Comparison by Band', fontsize=16)
        plt.xlabel('ULISSE Model Average XAI Value', fontsize=14)
        plt.ylabel('TS Model Average XAI Value', fontsize=14)
        plt.grid(True, alpha=0.3)

        avg_scatter_path = os.path.join(output_dir, 'band_average_comparison.png')
        plt.savefig(avg_scatter_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Band average comparison saved to {avg_scatter_path}")

    def _create_xai_heatmaps(self, xai_source_dir, band_names, output_dir):
        """
        Create heatmaps visualizing XAI values for each band in each scene.
        One heatmap is created for each band in each test scene.
        """
        print("Generating XAI heatmaps for each scene and band...")

        heatmaps_dir = os.path.join(output_dir, 'heatmaps')
        os.makedirs(heatmaps_dir, exist_ok=True)

        pivot_files = [f for f in os.listdir(xai_source_dir) if f.endswith('_pivot.csv')]

        for csv_file in tqdm(pivot_files, desc="Processing scenes for heatmaps"):
            image_id = csv_file.split('_xai_band_occlusion_pivot.csv')[0]
            file_path = os.path.join(xai_source_dir, csv_file)

            scene_dir = os.path.join(heatmaps_dir, image_id)
            os.makedirs(scene_dir, exist_ok=True)

            df = pd.read_csv(file_path)

            max_row = df['row_idx'].max() + 1
            max_col = df['col_idx'].max() + 1

            for band in band_names:
                heatmap_data = np.zeros((max_row, max_col)) - 999

                for _, row in df.iterrows():
                    r, c = int(row['row_idx']), int(row['col_idx'])
                    xai_value = row[f'xai_value_{band}']
                    heatmap_data[r, c] = xai_value

                valid_mask = heatmap_data != -999

                if not np.any(valid_mask):
                    continue

                plt.figure(figsize=(10, 8))

                vmin = np.min(heatmap_data[valid_mask])
                vmax = np.max(heatmap_data[valid_mask])

                abs_max = max(abs(vmin), abs(vmax))
                norm_range = [-abs_max, abs_max]

                masked_data = np.ma.masked_where(~valid_mask, heatmap_data)
                plt.imshow(masked_data, cmap='RdBu_r', vmin=norm_range[0], vmax=norm_range[1])

                plt.colorbar(label='XAI Value')
                plt.title(f'XAI Heatmap - {image_id} - Band {band}', fontsize=14)
                plt.axis('off')

                heatmap_path = os.path.join(scene_dir, f'xai_heatmap_band_{band}.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()

        print(f"XAI heatmaps saved to {heatmaps_dir}")


    def create_enhanced_xai_visualization(self, output_dir, occlusion_mode):
        """
        Create enhanced XAI visualizations that combine:
        1. Original RGB image
        2. XAI heatmaps for each band
        3. Ground truth contours
        4. Prediction contours with all bands
        5. Prediction contours with the specific band occluded

        Saves both images and an HTML viewer.
        """
        print("Generating enhanced XAI visualizations...")

        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"
        xai_source_dir = os.path.join(self.output_dir, f'xai_band_occlusion{mode_suffix}')
        heatmaps_dir = os.path.join(output_dir, 'heatmaps')
        enhanced_dir = os.path.join(output_dir, 'enhanced_heatmaps')
        os.makedirs(enhanced_dir, exist_ok=True)

        if not os.path.exists(xai_source_dir):
            print(f"XAI data directory not found: {xai_source_dir}")
            return

        pivot_files = [f for f in os.listdir(xai_source_dir) if f.endswith('_pivot.csv')]
        if not pivot_files:
            print(f"No XAI data files found in {xai_source_dir}")
            return

        first_file = pd.read_csv(os.path.join(xai_source_dir, pivot_files[0]))
        band_names = []
        for col in first_file.columns:
            if col.startswith('xai_value_'):
                band_name = col.replace('xai_value_', '')
                band_names.append(band_name)

        scene_dirs = []
        for csv_file in tqdm(pivot_files, desc="Creating enhanced visualizations"):
            image_id = csv_file.split('_xai_band_occlusion_pivot.csv')[0]
            file_path = os.path.join(xai_source_dir, csv_file)

            scene_dir = os.path.join(enhanced_dir, image_id)
            os.makedirs(scene_dir, exist_ok=True)
            scene_dirs.append(image_id)

            df = pd.read_csv(file_path)

            max_row = df['row_idx'].max() + 1
            max_col = df['col_idx'].max() + 1
            original_rgb_images = {}
            if self.test_dataset is not None:
                for i, sample in enumerate(self.test_dataset):
                    if sample.get('image_id') == image_id:
                        if self.temporal_mode == TemporalMode.SINGLE.value:
                            img = sample['image'].numpy()
                            original_rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
                            for c in range(3):
                                if c < img.shape[0]:
                                    band = img[c]
                                    band_norm = ((band - band.min()) / (band.max() - band.min() + 1e-10) * 255).astype(
                                        np.uint8)
                                    original_rgb[:, :, c] = band_norm
                            original_rgb = original_rgb[:max_row, :max_col]
                            original_rgb_images["main"] = original_rgb

                        elif self.temporal_mode == TemporalMode.TIMESERIES.value:
                            img = sample['image'].numpy()
                            original_rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
                            for c in range(3):
                                if c < img.shape[0]:
                                    band = img[c]
                                    band_norm = ((band - band.min()) / (band.max() - band.min() + 1e-10) * 255).astype(
                                        np.uint8)
                                    original_rgb[:, :, c] = band_norm
                            original_rgb = original_rgb[:max_row, :max_col]
                            original_rgb_images["main"] = original_rgb

                            if 'additional_images' in sample:
                                for t, add_img in enumerate(sample['additional_images']):
                                    add_img = add_img.numpy()
                                    add_rgb = np.zeros((add_img.shape[1], add_img.shape[2], 3), dtype=np.uint8)
                                    for c in range(3):
                                        if c < add_img.shape[0]:
                                            band = add_img[c]
                                            band_norm = ((band - band.min()) / (
                                                        band.max() - band.min() + 1e-10) * 255).astype(np.uint8)
                                            add_rgb[:, :, c] = band_norm
                                    add_rgb = add_rgb[:max_row, :max_col]
                                    original_rgb_images[f"timestamp_{t}"] = add_rgb


            ground_truth = np.zeros((max_row, max_col))
            valid_mask = np.zeros((max_row, max_col), dtype=bool)

            full_prediction = np.zeros((max_row, max_col))

            occluded_predictions = {band: np.zeros((max_row, max_col)) for band in band_names}

            for _, row in df.iterrows():
                r, c = int(row['row_idx']), int(row['col_idx'])
                ground_truth[r, c] = row['ground_truth']
                valid_mask[r, c] = True

                full_prediction[r, c] = row['predicted_value']

                for band in band_names:
                    if f'predicted_value_occluded_{band}' in row:
                        occluded_predictions[band][r, c] = row[f'predicted_value_occluded_{band}']

            for band in band_names:
                heatmap_data = np.zeros((max_row, max_col)) - 999
                valid_mask = np.zeros((max_row, max_col), dtype=bool)

                for _, row in df.iterrows():
                    r, c = int(row['row_idx']), int(row['col_idx'])
                    if f'xai_value_{band}' in row:
                        xai_value = row[f'xai_value_{band}']
                        heatmap_data[r, c] = xai_value
                        valid_mask[r, c] = True

                masked_heatmap = np.ma.masked_where(~valid_mask, heatmap_data)

                plt.figure(figsize=(12, 10))

                background_img = None
                if "main" in original_rgb_images:
                    background_img = original_rgb_images["main"]
                elif original_rgb_images:
                    background_img = list(original_rgb_images.values())[0]

                if background_img is not None:
                    plt.imshow(background_img)
                    plt.title(f'XAI Heatmap with Contours - {image_id} - Band {band}', fontsize=14)
                else:
                    background = np.zeros((max_row, max_col, 3), dtype=np.uint8) + 128
                    plt.imshow(background)
                    plt.title(f'XAI Heatmap - {image_id} - Band {band}', fontsize=14)

                vmin = np.min(masked_heatmap[~masked_heatmap.mask])
                vmax = np.max(masked_heatmap[~masked_heatmap.mask])
                abs_max = max(abs(vmin), abs(vmax))

                heatmap_plot = plt.imshow(masked_heatmap, cmap='RdBu_r',
                                          vmin=-abs_max, vmax=abs_max,
                                          alpha=0.7)
                plt.colorbar(heatmap_plot, label='XAI Value')

                ground_truth = np.zeros((max_row, max_col))
                full_prediction = np.zeros((max_row, max_col))
                occluded_prediction = np.zeros((max_row, max_col))

                for _, row in df.iterrows():
                    r, c = int(row['row_idx']), int(row['col_idx'])
                    ground_truth[r, c] = row['ground_truth']
                    full_prediction[r, c] = row['predicted_value']
                    if f'predicted_value_occluded_{band}' in row:
                        occluded_prediction[r, c] = row[f'predicted_value_occluded_{band}']

                gt_binary = ground_truth > 0.5
                gt_contour = plt.contour(gt_binary, levels=[0.5],
                                         colors=['yellow'], linewidths=4,
                                         linestyles='solid')

                pred_binary = full_prediction > 0.5
                pred_contour = plt.contour(pred_binary, levels=[0.5],
                                           colors=['red'], linewidths=4,
                                           linestyles='solid')

                occ_pred_binary = occluded_prediction > 0.5
                occ_contour = plt.contour(occ_pred_binary, levels=[0.5],
                                          colors=['orange'], linewidths=4,
                                          linestyles='solid')

                plt.legend([
                    plt.Line2D([0], [0], color='yellow', linestyle='solid', linewidth=2),
                    plt.Line2D([0], [0], color='red', linestyle='solid', linewidth=2),
                    plt.Line2D([0], [0], color='orange', linestyle='solid', linewidth=2)
                ], ['Ground Truth', 'Full Prediction', f'Without Band {band}'],
                    loc='lower right')

                plt.axis('off')

                enhanced_path = os.path.join(scene_dir, f'enhanced_heatmap_band_{band}.png')
                plt.savefig(enhanced_path, dpi=300, bbox_inches='tight')
                plt.close()

            if original_rgb_images:
                for timestamp_name, rgb_img in original_rgb_images.items():
                    if timestamp_name == "main":
                        rgb_path = os.path.join(scene_dir, 'original_rgb.png')
                    else:
                        rgb_path = os.path.join(scene_dir, f'original_rgb_{timestamp_name}.png')

                    plt.figure(figsize=(10, 8))
                    plt.imshow(rgb_img)
                    plt.axis('off')
                    if timestamp_name == "main":
                        plt.title(f'Original RGB - {image_id}', fontsize=14)
                    else:
                        plt.title(f'Original RGB - {image_id} - {timestamp_name}', fontsize=14)
                    plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
                    plt.close()

        self._create_html_viewer_with_contours(output_dir, enhanced_dir, band_names, scene_dirs)

        print(f"Enhanced XAI visualizations saved to {enhanced_dir}")
        return enhanced_dir

    def _create_html_viewer_with_contours(self, output_dir, enhanced_dir, band_names, scene_dirs):
        """
        Creates a static HTML page to display all enhanced XAI visualizations.
        Rows represent different scenes, and columns represent different bands.
        Includes all RGB images from different timestamps when available.
        Uses absolute paths to ensure images display correctly.

        Parameters:
            output_dir (str): Directory where XAI explanation results are stored
            enhanced_dir (str): Directory with enhanced heatmap images
            band_names (list): List of band names
            scene_dirs (list): List of scene IDs
        """
        print("Creating HTML viewer for enhanced XAI visualizations...")

        if not os.path.exists(enhanced_dir):
            print(f"Enhanced heatmaps directory not found: {enhanced_dir}")
            return

        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced XAI Heatmaps Viewer</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1, h2 {
                    color: #333;
                }
                .scene-container {
                    margin-bottom: 40px;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .rgb-images {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .rgb-images img {
                    max-height: 200px;
                    border: 1px solid #ddd;
                }
                .band-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 15px;
                }
                .band-container {
                    text-align: center;
                }
                .band-container img {
                    max-width: 100%;
                    border: 1px solid #ddd;
                }
                .band-title {
                    margin-top: 5px;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <h1>Enhanced XAI Visualizations</h1>
        """

        # For each scene, create a section
        for scene_id in sorted(scene_dirs):
            scene_dir = os.path.join(enhanced_dir, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            html_content += f"""
            <div class="scene-container container-fluid">
                <h2>Scene: {scene_id}</h2>

                <!-- RGB Images Section -->
                <div class="rgb-images row">
            """

            # Add original RGB images if available
            original_rgb = os.path.join(scene_dir, 'original_rgb.png')
            if os.path.exists(original_rgb):
                abs_path = os.path.abspath(original_rgb)
                html_content += f"""
                    <div class="col-md-2">
                        <img src="file://{abs_path}" alt="Original RGB">
                        <div>Main RGB</div>
                    </div>
                """

            # Check for additional RGB images (from other timestamps)
            for item in os.listdir(scene_dir):
                if item.startswith('original_rgb_') and item.endswith('.png'):
                    timestamp = item.replace('original_rgb_', '').replace('.png', '')
                    abs_path = os.path.abspath(os.path.join(scene_dir, item))
                    html_content += f"""
                        <div>
                            <img src="file://{abs_path}" alt="RGB {timestamp}">
                            <div>RGB {timestamp}</div>
                        </div>
                    """

            html_content += """
                </div>

                <!-- Band Heatmaps Grid -->
                <div class="band-grid">
            """

            # Add enhanced heatmaps for each band
            for band in band_names:
                heatmap_file = os.path.join(scene_dir, f'enhanced_heatmap_band_{band}.png')
                if os.path.exists(heatmap_file):
                    abs_path = os.path.abspath(heatmap_file)
                    html_content += f"""
                    <div class="band-container">
                        <div class="band-title">Band {band}</div>
                        <img src="file://{abs_path}" alt="Band {band} heatmap">
                    </div>
                    """

            # Close the scene container
            html_content += """
                </div>
            """

            # ====== TEST ======
            html_content += f"""
                            <div class="rgb-images images-scatter">
                        """

            # Add original RGB images if available
            scatter_f_min_f_plus = os.path.join(enhanced_dir, '..', 'individual_scatter_plots',f'{scene_id}_scatter_plot.png')
            if os.path.exists(scatter_f_min_f_plus):
                abs_path = os.path.abspath(scatter_f_min_f_plus)
                html_content += f"""
                                <div class="band-container" style="width: 100%; max-height: 600px !important;">
                                    <div class="band-title">Scatter for each band</div>
                                    <img style="width: 100%; heigth: 100%; object-fit: cover; max-height: 600px !important;" src="file://{abs_path}" alt="Scatter F-Min <> F-Plus">
                                </div>
                            """

            html_content += """
                            </div>
                        """

            html_content += """
            </div>
            """

        # Close HTML
        html_content += """
        </body>
        </html>
        """

        # Write HTML file
        html_path = os.path.join(output_dir, 'enhanced_heatmap_viewer.html')
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"Enhanced HTML viewer created at {html_path}")
        return html_path

    def _create_html_viewer_original(self, output_dir, heatmaps_dir, band_names, scene_dirs):
        """Creates an HTML viewer for original heatmaps without contours"""

        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>XAI Heatmaps Viewer - Original</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                table {
                    border-collapse: collapse;
                    margin: 0 auto;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                th, td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                    position: sticky;
                    top: 0;
                }
                th.scene-id {
                    position: sticky;
                    left: 0;
                    z-index: 2;
                    background-color: #e6e6e6;
                }
                td.scene-id {
                    position: sticky;
                    left: 0;
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                img {
                    max-width: 250px;
                    max-height: 250px;
                    display: block;
                    margin: 0 auto;
                }
                .container {
                    overflow-x: auto;
                    max-width: 100%;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>XAI Heatmaps by Scene and Band - Original</h1>
            <div class="container">
                <table>
                    <thead>
                        <tr>
                            <th class="scene-id">Scene ID</th>
        """

        # Add column headers (band names)
        for band in band_names:
            html_content += f'                        <th>Band {band}</th>\n'

        html_content += """
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add rows for each scene
        for scene_id in sorted(scene_dirs):
            scene_dir = os.path.join(heatmaps_dir, scene_id)

            # Start a new row with scene ID
            html_content += f'                    <tr>\n'
            html_content += f'                        <td class="scene-id">{scene_id}</td>\n'

            # Add images for each band
            for band in band_names:
                image_path = os.path.join(heatmaps_dir, scene_id, f'xai_heatmap_band_{band}.png')
                if os.path.exists(image_path):
                    # Use absolute path for the image
                    abs_path = os.path.abspath(image_path)
                    html_content += f'                        <td><img src="file://{abs_path}" alt="Heatmap for {scene_id}, band {band}"></td>\n'
                else:
                    html_content += f'                        <td>Image not available</td>\n'

            html_content += f'                    </tr>\n'

        # Close HTML
        html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """

        # Write HTML file
        html_path = os.path.join(output_dir, 'heatmap_viewer_original.html')
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"HTML viewer for original heatmaps created at {html_path}")

    def _create_prediction_scatterplots(self, combined_data, band_names, output_dir):
        """
        Create scatter plots showing f_plus vs f_minus for each band.
        Points are colored based on ground truth (GT) and prediction (P) combinations:
        - Green: GT=0 and P=0
        - Yellow: GT=0 and P=1
        - Orange: GT=1 and P=0
        - Red: GT=1 and P=1

        One plot is created for each band, aggregating data across all test images.
        """
        print("Generating f_plus vs f_minus scatter plots...")

        scatter_dir = os.path.join(output_dir, 'f_plus_minus_scatters')
        os.makedirs(scatter_dir, exist_ok=True)

        colors = {
            (0, 0): 'green',  # GT=0, P=0
            (0, 1): 'green',  # GT=0, P=1
            (1, 0): 'red',  # GT=1, P=0
            (1, 1): 'red'  # GT=1, P=1
        }

        for band in tqdm(band_names, desc="Creating f_plus vs f_minus scatter plots"):
            plt.figure(figsize=(10, 8))

            f_plus_col = f'f_plus_{band}'
            f_minus_col = f'f_minus_{band}'

            if f_plus_col not in combined_data.columns or f_minus_col not in combined_data.columns:
                print(f"Skipping band {band} - missing data columns")
                continue

            if 'pred_plus' not in combined_data.columns:
                combined_data['pred_plus'] = (combined_data[[f'f_plus_{b}' for b in band_names if
                                                             f'f_plus_{b}' in combined_data.columns]].iloc[:,
                                              0] > 0.5).astype(int)

            pred_minus_col = f'pred_minus_{band}'
            combined_data[pred_minus_col] = (combined_data[f_minus_col] > 0.5).astype(int)

            for (gt, pred_plus), color in colors.items():
                mask = (combined_data['ground_truth'] == gt) & (combined_data['pred_plus'] == pred_plus)
                subset = combined_data[mask]

                if len(subset) > 0:
                    plt.scatter(
                        subset[f_plus_col],
                        subset[f_minus_col],
                        c=color,
                        alpha=0.5,
                        label=f'GT={gt}, P={pred_plus}',
                        s=20
                    )

            min_val = min(combined_data[f_plus_col].min(), combined_data[f_minus_col].min())
            max_val = max(combined_data[f_plus_col].max(), combined_data[f_minus_col].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=0.5, color='gray', linestyle='-', alpha=0.3)

            plt.xlabel('f_plus (All bands present)', fontsize=14)
            plt.ylabel('f_minus (Band occluded)', fontsize=14)
            plt.title(f'f_plus vs f_minus for Band {band}', fontsize=16)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.axis('equal')

            scatter_path = os.path.join(scatter_dir, f'f_plus_minus_scatter_{band}.png')
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"f_plus vs f_minus scatter plots saved to {scatter_dir}")

    def _create_consolidated_scatter_plot(self, combined_data, band_names, output_dir):
        """
        Creates separate consolidated figures containing scatter plots for all bands,
        one for GT=0 (healthy) and one for GT=1 (damaged).
        Each subplot shows f_plus vs f_minus for a single band.
        Adds a yellow dashed diagonal line (y=x) and displays percentages of points above/below.
        """
        print("Generating separate consolidated f_plus vs f_minus scatter plots for GT=0 and GT=1...")

        scatter_dir = os.path.join(output_dir, 'consolidated_scatters')
        os.makedirs(scatter_dir, exist_ok=True)

        gt0_data = combined_data[combined_data['ground_truth'] == 0]
        gt1_data = combined_data[combined_data['ground_truth'] == 1]

        print(f"GT=0 pixels: {len(gt0_data)}")
        print(f"GT=1 pixels: {len(gt1_data)}")

        n_bands = len(band_names)
        n_cols = min(5, n_bands)  # Max 5 plots per row
        n_rows = (n_bands + n_cols - 1) // n_cols

        for gt_class, data, title in [(0, gt0_data, 'Healthy (GT=0)'), (1, gt1_data, 'Damaged (GT=1)')]:
            fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

            # Create scatter plot for each band
            for i, band in enumerate(band_names):
                ax = plt.subplot(n_rows, n_cols, i + 1)

                f_plus_col = f'f_plus_{band}'
                f_minus_col = f'f_minus_{band}'

                if f_plus_col not in data.columns or f_minus_col not in data.columns:
                    print(f"Skipping band {band} - missing data columns")
                    continue

                if 'pred_plus' not in data.columns:
                    data['pred_plus'] = (data[[f'f_plus_{b}' for b in band_names
                                               if f'f_plus_{b}' in data.columns]].iloc[:, 0] > 0.5).astype(int)

                scatter = ax.scatter(
                    data[f_plus_col],
                    data[f_minus_col],
                    s=5,  # Small point size for better visibility with many points
                    alpha=0.5,
                    c='red' if gt_class == 1 else 'green',
                    label=f"GT={gt_class}"
                )

                min_val = min(data[f_plus_col].min(), data[f_minus_col].min())
                max_val = max(data[f_plus_col].max(), data[f_minus_col].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'y--', alpha=1.0, linewidth=4)

                ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
                ax.axvline(x=0.5, color='gray', linestyle='-', alpha=0.3)

                total_points = len(data)
                if total_points > 0:
                    points_above = sum(data[f_minus_col] > data[f_plus_col])
                    points_below = sum(data[f_minus_col] < data[f_plus_col])
                    pct_above = (points_above / total_points) * 100
                    pct_below = (points_below / total_points) * 100

                    if gt_class == 0:
                        ax.text(0.20, 0.80, f"{pct_above:.1f}%", transform=ax.transAxes,
                                fontsize=20, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    else:
                        ax.text(0.50, 0.30, f"{pct_below:.1f}%", transform=ax.transAxes,
                                fontsize=20, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                ax.set_title(f'Band {band}', fontsize=12)
                ax.set_xlabel('f_plus', fontsize=10)
                ax.set_ylabel('f_minus', fontsize=10)

                ax.set_aspect('equal')

                ax.grid(True, alpha=0.3)

                if min_val >= 0 and max_val <= 1:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

            plt.tight_layout()

            plt.subplots_adjust(top=0.92)

            consolidated_path = os.path.join(scatter_dir, f'consolidated_scatter_plot_gt{gt_class}.png')
            plt.savefig(consolidated_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Consolidated scatter plots saved to {scatter_dir}")
    def _create_individual_consolidated_scatter_plots(self, combined_data, band_names, output_dir):
        """
        Creates consolidated scatter plots for each individual image.
        Each figure contains multiple subplots (one per band) showing f_plus vs f_minus.
        Points are colored based on ground truth and prediction combinations.
        """
        print("Generating individual consolidated scatter plots for each image...")

        individual_scatter_dir = os.path.join(output_dir, 'individual_scatter_plots')
        os.makedirs(individual_scatter_dir, exist_ok=True)

        colors = {
            (0, 0): 'green',  # GT=0, Pred=0 (True Negative)
            (0, 1): 'yellow',  # GT=0, Pred=1 (False Positive)
            (1, 0): 'orange',  # GT=1, Pred=0 (False Negative)
            (1, 1): 'red'  # GT=1, Pred=1 (True Positive)
        }

        image_ids = combined_data['image_id'].unique()

        n_bands = len(band_names)
        n_cols = min(5, n_bands)  # Max 5 plots per row
        n_rows = (n_bands + n_cols - 1) // n_cols

        for image_id in tqdm(image_ids, desc="Creating individual scatter plots"):
            image_data = combined_data[combined_data['image_id'] == image_id]

            if len(image_data) == 0:
                continue

            plt.figure(figsize=(5 * n_cols, 4 * n_rows))

            for i, band in enumerate(band_names):
                plt.subplot(n_rows, n_cols, i + 1)

                f_plus_col = f'f_plus_{band}'
                f_minus_col = f'f_minus_{band}'

                if f_plus_col not in image_data.columns or f_minus_col not in image_data.columns:
                    continue

                for (gt, pred), color in colors.items():
                    mask = (image_data['ground_truth'] == gt) & (image_data['predicted_value'] == pred)
                    if mask.sum() > 0:
                        plt.scatter(
                            image_data.loc[mask, f_plus_col],
                            image_data.loc[mask, f_minus_col],
                            color=color,
                            alpha=0.5,
                            s=10,
                            label=f'GT={gt}, P={pred}'
                        )

                plt.title(f'Band {band}')
                plt.xlabel('f_plus')
                plt.ylabel('f_minus')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')

                min_val = min(image_data[f_plus_col].min(), image_data[f_minus_col].min())
                max_val = max(image_data[f_plus_col].max(), image_data[f_minus_col].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.figlegend(by_label.values(), by_label.keys(),
                          loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0))

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)

            plt.suptitle(f'Image ID: {image_id}', fontsize=16, y=0.98)

            plt.savefig(os.path.join(individual_scatter_dir, f'{image_id}_scatter_plot.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Individual scatter plots saved to {individual_scatter_dir}")
        return individual_scatter_dir

    def _create_f_plus_minus_boxplots(self, combined_data, band_names, output_dir):
        """
        Create box plots of f_plus and f_minus values for each band across all scenes.
        Split by ground truth class (GT=0 vs GT=1).
        Uses same styling as XAI boxplots.
        """
        print("Generating f_plus and f_minus box plots with ground truth split...")
        sns.set(style="whitegrid")

        plt.figure(figsize=(20, 10))

        positions_plus = []
        box_data_plus = []
        colors_plus = []
        labels_plus = []
        tick_positions_plus = []

        for i, band in enumerate(band_names):
            f_plus_col = f'f_plus_{band}'

            healthy_f_plus = combined_data[combined_data['ground_truth'] == 0][f_plus_col].dropna()

            # For damaged samples (GT=1)
            damaged_f_plus = combined_data[combined_data['ground_truth'] == 1][f_plus_col].dropna()

            box_data_plus.append(healthy_f_plus)
            colors_plus.append('lightgreen')
            positions_plus.append(i * 3)

            box_data_plus.append(damaged_f_plus)
            colors_plus.append('lightcoral')
            positions_plus.append(i * 3 + 1)

            labels_plus.append(band)
            tick_positions_plus.append(i * 3 + 0.5)

        box_plus = plt.boxplot(box_data_plus, positions=positions_plus, patch_artist=True,
                               showfliers=False, widths=0.8)

        for patch, color in zip(box_plus['boxes'], colors_plus):
            patch.set_facecolor(color)

        plt.title('Distribution of f_plus Values by Band and Ground Truth Class', fontsize=16)
        plt.xlabel('Band', fontsize=14)
        plt.ylabel('f_plus Value', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        plt.gca().set_xticks(tick_positions_plus)
        plt.gca().set_xticklabels(labels_plus)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Healthy (GT=0)'),
            Patch(facecolor='lightcoral', label='Damaged (GT=1)')
        ]
        plt.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        f_plus_boxplot_path = os.path.join(output_dir, 'f_plus_boxplot_by_class.png')
        plt.savefig(f_plus_boxplot_path, dpi=300)
        plt.close()

        plt.figure(figsize=(20, 10))

        positions_minus = []
        box_data_minus = []
        colors_minus = []
        labels_minus = []
        tick_positions_minus = []

        for i, band in enumerate(band_names):
            f_minus_col = f'f_minus_{band}'

            healthy_f_minus = combined_data[combined_data['ground_truth'] == 0][f_minus_col].dropna()

            damaged_f_minus = combined_data[combined_data['ground_truth'] == 1][f_minus_col].dropna()

            box_data_minus.append(healthy_f_minus)
            colors_minus.append('lightgreen')
            positions_minus.append(i * 3)

            box_data_minus.append(damaged_f_minus)
            colors_minus.append('lightcoral')
            positions_minus.append(i * 3 + 1)

            labels_minus.append(band)
            tick_positions_minus.append(i * 3 + 0.5)

        box_minus = plt.boxplot(box_data_minus, positions=positions_minus, patch_artist=True,
                                showfliers=False, widths=0.8)

        for patch, color in zip(box_minus['boxes'], colors_minus):
            patch.set_facecolor(color)

        plt.title('Distribution of f_minus Values by Band and Ground Truth Class', fontsize=16)
        plt.xlabel('Band', fontsize=14)
        plt.ylabel('f_minus Value', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        plt.gca().set_xticks(tick_positions_minus)
        plt.gca().set_xticklabels(labels_minus)

        legend_elements = [
            Patch(facecolor='lightgreen', label='Healthy (GT=0)'),
            Patch(facecolor='lightcoral', label='Damaged (GT=1)')
        ]
        plt.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        f_minus_boxplot_path = os.path.join(output_dir, 'f_minus_boxplot_by_class.png')
        plt.savefig(f_minus_boxplot_path, dpi=300)
        plt.close()

        print(f"f_plus boxplot saved to {f_plus_boxplot_path}")
        print(f"f_minus boxplot saved to {f_minus_boxplot_path}")

    def create_polarized_xai_visualization(self, output_dir, occlusion_mode):
        """
        Create polarized XAI visualizations based on ground truth values:
        - If XAI < 0, set to 0
        - If GT = 0 (healthy), multiply XAI by -1 (invert positive values)
        - If GT = 1 (damaged), keep positive XAI values as they are

        This creates a visualization where:
        - For GT=0 pixels: negative (blue) values indicate importance
        - For GT=1 pixels: positive (red) values indicate importance
        - Values near 0 (white) indicate low importance

        Also adds ground truth contours to the visualizations.
        """
        print("Creating polarized XAI visualizations with ground truth contours...")

        # Path to XAI data directory
        mode_suffix = "_0" if occlusion_mode == 'zero' else "_avg"
        xai_source_dir = os.path.join(self.output_dir, f'xai_band_occlusion{mode_suffix}')
        if not os.path.exists(xai_source_dir):
            print(f"XAI data directory not found: {xai_source_dir}")
            return

        polarized_dir = os.path.join(output_dir, 'polarized_xai')
        os.makedirs(polarized_dir, exist_ok=True)

        pivot_files = [f for f in os.listdir(xai_source_dir) if f.endswith('_pivot.csv')]
        if not pivot_files:
            print(f"No XAI pivot data files found in {xai_source_dir}")
            return

        first_file = os.path.join(xai_source_dir, pivot_files[0])
        df = pd.read_csv(first_file)
        band_names = [col.replace('xai_value_', '') for col in df.columns if col.startswith('xai_value_')]

        print(f"Processing {len(pivot_files)} scenes for {len(band_names)} bands...")

        for csv_file in tqdm(pivot_files, desc="Processing polarized visualizations"):
            image_id = csv_file.split('_xai_band_occlusion_pivot.csv')[0]
            scene_dir = os.path.join(polarized_dir, image_id)
            os.makedirs(scene_dir, exist_ok=True)

            df = pd.read_csv(os.path.join(xai_source_dir, csv_file))

            max_row = df['row_idx'].max() + 1
            max_col = df['col_idx'].max() + 1

            ground_truth = np.zeros((max_row, max_col), dtype=np.uint8)
            for _, row in df.iterrows():
                r, c = int(row['row_idx']), int(row['col_idx'])
                ground_truth[r, c] = int(row['ground_truth'])

            for band in band_names:
                polarized_xai = np.zeros((max_row, max_col), dtype=np.float32)

                for _, row in df.iterrows():
                    r, c = int(row['row_idx']), int(row['col_idx'])
                    gt = int(row['ground_truth'])
                    xai_value = float(row[f'xai_value_{band}'])

                    if gt == 0:
                        polarized_xai[r, c] = -max(0, xai_value)  # Invert positive values
                    else:
                        polarized_xai[r, c] = max(0, xai_value)  # Keep positive values

                vmin = polarized_xai.min()
                vmax = polarized_xai.max()
                abs_max = max(abs(vmin), abs(vmax))

                plt.figure(figsize=(10, 8))
                plt.imshow(polarized_xai, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)
                plt.colorbar(label='XAI Value')
                plt.title(f'Polarized XAI for {band}')
                plt.axis('off')

                gt_binary = ground_truth > 0.5
                gt_contour = plt.contour(gt_binary, levels=[0.5],
                                         colors=['yellow'], linewidths=4,
                                         linestyles='solid')

                plt.tight_layout()
                plt.savefig(os.path.join(scene_dir, f'{band}_polarized.png'), dpi=200, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(10, 8))
                plt.imshow(polarized_xai, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)
                plt.colorbar(label='XAI Value')
                plt.title(f'Polarized XAI for {band}')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(scene_dir, f'{band}_polarized_bold.png'), dpi=200, bbox_inches='tight')
                plt.close()

        print(f"Polarized XAI visualizations saved to {polarized_dir}")

        self._create_polarized_html_viewer(polarized_dir, band_names)

        return polarized_dir

    def _create_polarized_html_viewer(self, polarized_dir, band_names):
        """
        Create an HTML viewer for polarized XAI visualizations
        """
        print("Creating HTML viewer for polarized XAI visualizations...")

        scene_dirs = [d for d in os.listdir(polarized_dir)
                      if os.path.isdir(os.path.join(polarized_dir, d))]

        if not scene_dirs:
            print("No scene directories found")
            return

        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Polarized XAI Visualizations</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                table {
                    border-collapse: collapse;
                    margin: 0 auto;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                th, td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                    position: sticky;
                    top: 0;
                }
                th.scene-id {
                    position: sticky;
                    left: 0;
                    z-index: 2;
                    background-color: #e6e6e6;
                }
                td.scene-id {
                    position: sticky;
                    left: 0;
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                img {
                    max-width: 250px;
                    max-height: 250px;
                    display: block;
                    margin: 0 auto;
                }
                .container {
                    overflow-x: auto;
                    max-width: 100%;
                    margin-top: 20px;
                }
                .legend {
                    text-align: center;
                    margin: 20px;
                    padding: 10px;
                    background-color: white;
                    border: 1px solid #ddd;
                    display: inline-block;
                }
                .legend-item {
                    margin: 5px;
                }
                .blue-box {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    background-color: #2166AC;
                    vertical-align: middle;
                }
                .red-box {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    background-color: #B2182B;
                    vertical-align: middle;
                }
                .white-box {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    background-color: white;
                    border: 1px solid #ddd;
                    vertical-align: middle;
                }
            </style>
        </head>
        <body>
            <h1>Polarized XAI Visualizations</h1>
            <div style="text-align: center">
                <div class="legend">
                    <div class="legend-item"><div class="blue-box"></div> Important for healthy pixels (GT=0)</div>
                    <div class="legend-item"><div class="red-box"></div> Important for damaged pixels (GT=1)</div>
                    <div class="legend-item"><div class="white-box"></div> Low importance</div>
                </div>
            </div>
            <div class="container">
                <table>
                    <thead>
                        <tr>
                            <th class="scene-id">Scene ID</th>
        """

        for band in band_names:
            html_content += f'                        <th>Band {band}</th>\n'

        html_content += """
                        </tr>
                    </thead>
                    <tbody>
        """

        for scene_id in sorted(scene_dirs):
            scene_dir = os.path.join(polarized_dir, scene_id)

            html_content += f'                    <tr>\n'
            html_content += f'                        <td class="scene-id">{scene_id}</td>\n'

            for band in band_names:
                image_path = os.path.join(polarized_dir, scene_id, f'{band}_polarized.png')
                if os.path.exists(image_path):
                    abs_path = os.path.abspath(image_path)
                    html_content += f'                        <td><img src="file://{abs_path}" alt="Polarized XAI for {scene_id}, band {band}"></td>\n'
                else:
                    html_content += f'                        <td>Image not available</td>\n'

            html_content += f'                    </tr>\n'

        html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """

        html_path = os.path.join(polarized_dir, 'polarized_xai_viewer.html')
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"HTML viewer created at {html_path}")
        return html_path



    def _reconstruct_image(self, tiles, positions, valid_masks, height, width):
        first_elements, second_elements = zip(*positions)
        h_rec = max(first_elements) + 1 * height
        w_rec = max(second_elements) + 1 * width

        reconstructed = torch.zeros((tiles[0].shape[0], h_rec, w_rec), dtype=tiles[0].dtype)
        reconstructed_valid_mask = torch.zeros((tiles[0].shape[0], h_rec, w_rec), dtype=tiles[0].dtype)

        for tile, (i, j), valid_mask in zip(tiles, positions, valid_masks):
            i_start, j_start = i*width, j + j*height
            i_end, j_end = i_start + tile.shape[1], j_start + tile.shape[2]
            reconstructed[:, i_start:i_end, j_start:j_end] += tile * valid_mask
            reconstructed_valid_mask[:, i_start:i_end, j_start:j_end] += valid_mask

        valid_indices = torch.nonzero(reconstructed_valid_mask)
        mic_c, min_h, min_w = valid_indices.min(dim=0)[0]
        max_c, max_h, max_w = valid_indices.max(dim=0)[0]
        cutted_img = reconstructed[:, min_h:max_h + 1, min_w:max_w + 1]

        return cutted_img

    def save_georeferenced_images(self, image_id, pred_array, true_mask):
        original_mask_path = next(
            path for path in self.test_dataset.mask_files
            if path.stem.split('_')[1] in image_id
        )

        num_id = os.path.basename(image_id).split('_')[1]
        original_mask_path = os.path.join(self.test_dataset.mask_files[0].parent, f'mask_{num_id}.tif')
        with rasterio.open(original_mask_path) as src:
            profile = src.profile.copy()

        pred_images_path = os.path.join(self.output_dir, 'predicted_images', 'images')
        os.makedirs(pred_images_path, exist_ok=True)
        pred_tifs_path = os.path.join(self.output_dir, 'predicted_images', 'tifs')
        os.makedirs(pred_tifs_path, exist_ok=True)
        pred_array_img = (pred_array > 0.5).astype(np.uint8) * 255
        true_mask_binary = (true_mask > 0.5).astype(np.uint8)
        original_height = pred_array_img.squeeze().shape[0]
        original_width = pred_array_img.squeeze().shape[1]

        if original_height < 1024:
            scale_factor = 1024 / original_height
            new_height = 1024
            new_width = int(original_width * scale_factor)
            pred_array_resized = cv2.resize(pred_array_img.squeeze(), (new_width, new_height),
                                            interpolation=cv2.INTER_LINEAR)

            true_mask_resized = cv2.resize(true_mask_binary.squeeze(), (new_width, new_height),
                                           interpolation=cv2.INTER_LINEAR)
        else:
            pred_array_resized = pred_array_img.squeeze()
            true_mask_resized = true_mask_binary.squeeze()

        pred_rgb = np.stack([pred_array_resized] * 3, axis=-1)

        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(true_mask_resized, kernel, iterations=2)
        boundaries = true_mask_resized - eroded
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)
        pred_rgb[boundaries > 0] = [255, 0, 0]
        Image.fromarray(pred_rgb).save(os.path.join(pred_images_path, f'{image_id}.jpg'))
        pred_array_tif = (pred_array > 0.5).astype(np.uint8)
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })

        with rasterio.open(os.path.join(pred_tifs_path, f'{image_id}.tif'), 'w', **profile) as dst:
            dst.write(pred_array_tif.squeeze(), 1)

    def _compute_image_metrics(self, pred_mask, true_mask):
        unique_labels = np.unique(np.concatenate([true_mask.ravel(), pred_mask.ravel()]))
        conf_matrix = confusion_matrix(true_mask.ravel(), pred_mask.ravel(),
                                       labels=[0, 1], sample_weight=None)

        if conf_matrix.shape == (2, 2):
            TN, FN, FP, TP = conf_matrix.ravel()
        else:
            TN = FP = FN = TP = 0
            if conf_matrix.shape == (1, 1):
                if 0 in unique_labels and 1 not in unique_labels:
                    TN = conf_matrix[0, 0]
                elif 1 in unique_labels and 0 not in unique_labels:
                    TP = conf_matrix[0, 0]
            elif conf_matrix.shape == (1, 2):
                if 0 in true_mask.ravel():
                    TN = conf_matrix[0, 0]
                    FP = conf_matrix[0, 1]
                else:
                    FN = conf_matrix[0, 0]
                    TP = conf_matrix[0, 1]
            elif conf_matrix.shape == (2, 1):
                if 0 in pred_mask.ravel():
                    TN = conf_matrix[0, 0]
                    FN = conf_matrix[1, 0]
                else:
                    FP = conf_matrix[0, 0]
                    TP = conf_matrix[1, 0]

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0

        precision_class1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_class1 = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_class1 = compute_f1(TP, FP, FN)

        precision_class0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        recall_class0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_class0 = compute_f1(TN, FN, FP)

        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        epoch_metrics = {
            'accuracy': accuracy,
            'precision_class1': precision_class1,
            'recall_class1': recall_class1,
            'iou': iou,
            'f1_class1': f1_class1,
            'precision_class0': precision_class0,
            'recall_class0': recall_class0,
            'f1_class0': f1_class0,
            'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP
        }

        return epoch_metrics

    def _save_aggregated_results(self, image_level_results):
        metrics_df = pd.DataFrame.from_records(
            [results['metrics'] for results in image_level_results.values()]
        )

        metrics_df.to_csv(self.output_dir / 'image_level_metrics.csv', index=False)

    def set_reproducibility(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Example usage
if __name__ == "__main__":
    print('TODO')