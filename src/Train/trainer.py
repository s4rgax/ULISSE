import os
import random
import re
import time
from pathlib import Path

import rasterio
import torch
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from collections import defaultdict
from more_itertools import chunked
from PIL import Image

from src.Model.models import ResNetUNet, MultiLoraResNetUNet, TimeseriesMultiLoraResNetUNet

import pandas as pd

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from src.Model.utils import TverskyLoss
from src.Utils.enums import TemporalMode
from src.Utils.functions import compute_metrics_from_conf_matrix, compute_f1, transform_batch_positions

import matplotlib.pyplot as plt
from matplotlib.table import Table


class UNetTrainer:
    def __init__(
            self,
            model_name,
            train_dataset,
            val_dataset,
            test_dataset=None,
            data_tile_size=224,
            model_tile_size=224,
            num_trials=30,
            max_epochs=150,
            early_stopping_patience=10,
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
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
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

    def objective(self, params):
        trial_id = len(self.trials_history)

        batch_size = int(params['batch_size'])
        print(f'Batch size: {batch_size}')
        # batch_size = 16
        learning_rate = float(params['learning_rate'])
        tversky_alpha = float(params['tversky_alpha'])
        tversky_beta = 1 - tversky_alpha
        print(f'Choosed parameters: {params}')
        peft_attr = {}
        if self.peft_encoder:
            peft_attr['lora_rank'] = params['lora_rank']
            # peft_attr['lora_rank'] = 8

        if self.temporal_mode == TemporalMode.TIMESERIES.value:
            self.model = MultiLoraResNetUNet.from_pretrained(
                self.model_name,
                data_tile_size=self.data_input_size,
                model_input_size=self.model_input_size,
                num_classes=1,
                peft=self.peft_encoder,
                peft_attr=peft_attr,
                num_additional_images=self.num_additional_images,
                fusion_mode=self.fusion_mode,
                fusion_technique=self.fusion_technique,
                # TODO RICORDATI DI DISATTIVATE
                random_init=True
            )
        elif self.temporal_mode == TemporalMode.SINGLE.value:
            self.model = ResNetUNet.from_pretrained(
                self.model_name,
                peft=self.peft_encoder,
                peft_attr=peft_attr,
                data_tile_size=self.data_input_size,
                model_input_size=self.model_input_size,
                num_classes=1
            )
        self.model = self.model.to(self.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers_dl,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers_dl,
            pin_memory=True
        )

        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers_dl,
                pin_memory=True
            )

        self.criterion = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        best_metric = float('inf') if self.optimization_metric == 'loss' else float('-inf')
        patience_counter = 0
        epoch_metrics_history = []
        best_epoch_metrics_history = []

        if hasattr(self.model, 'encoder') and self.freeze_encoder:
            print('Freezing encoder')
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        best_epoch = 0
        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_conf_matrix, train_num_samples = self.train_epoch(epoch)
            train_metrics = self.evaluate_epoch(precomputed_metrics={
                'loss': train_loss,
                'conf_matrix': train_conf_matrix,
                'num_samples': train_num_samples
            }, phase='train')
            val_metrics = self.evaluate_epoch(self.val_loader, 'val')

            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)

            epoch_metrics = {
                'trial_id': trial_id,
                'epoch': epoch,
                'starting_learning_rate': float(params['learning_rate']),
                'ending_learning_rate': self.optimizer.param_groups[0]['lr'],
                'batch_size': batch_size,
                'tversky_alpha': tversky_alpha,
                'tversky_beta': tversky_beta,
                'lora_rank': params.get('lora_rank',''),
                'time': f"{hours:02}:{minutes:02}:{seconds:02}",
                **{f'train_{k}': v for k, v in train_metrics.items() if k != 'phase'},
                **{f'val_{k}': v for k, v in val_metrics.items() if k != 'phase'}
            }

            epoch_metrics_history.append(epoch_metrics)

            current_metric = val_metrics['loss'] if self.optimization_metric == 'loss' else val_metrics['f1_class1']
            self.scheduler.step(current_metric if self.optimization_metric == 'loss' else -current_metric)
            improved = current_metric < best_metric if self.optimization_metric == 'loss' else current_metric > best_metric

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0

                trial_model_path = os.path.join(self.models_dir, f'best_model_trial.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'hyperparameters': params,
                    'epoch': epoch,
                    f'best_{self.optimization_metric}': best_metric,
                    'trial_id': trial_id
                }, trial_model_path)

                best_epoch_metrics_history = epoch_metrics_history

                print(f'Better local {self.optimization_metric}: {best_metric}')

                is_better_global = (self.optimization_metric == 'loss' and best_metric < self.best_global_metric) or \
                                   (self.optimization_metric == 'f1_class1' and best_metric > self.best_global_metric)

                if is_better_global:
                    self.best_global_metric = best_metric
                    self.best_trial_id = trial_id

                    best_model_path = Path(os.path.join(self.output_dir, 'best_model.pth'))
                    if best_model_path.exists():
                        os.remove(best_model_path)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'hyperparameters': params,
                        'epoch': epoch,
                        f'best_{self.optimization_metric}': best_metric,
                        'trial_id': trial_id
                    }, best_model_path)

                    print(f'Better global {self.optimization_metric}: {best_metric}')
            else:
                patience_counter += 1

            print(
                f'Trial {trial_id}, Epoch {epoch} | '
                f'Train - Loss: {train_metrics["loss"]:.4f}, F1 (class 1): {train_metrics["f1_class1"]:.4f}, '
                f'TN: {train_metrics["TN"]}, FP: {train_metrics["FP"]}, FN: {train_metrics["FN"]}, TP: {train_metrics["TP"]} | '
                f'Validation - Loss: {val_metrics["loss"]:.4f}, F1 (class 1): {val_metrics["f1_class1"]:.4f}, '
                f'TN: {val_metrics["TN"]}, FP: {val_metrics["FP"]}, FN: {val_metrics["FN"]}, TP: {val_metrics["TP"]}'
            )
            print('-' * 20)

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        trial_model_path = os.path.join(self.models_dir, f'best_model_trial.pth')
        checkpoint = torch.load(trial_model_path, weights_only=False, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.test_dataset is not None:
            test_metrics = self.evaluate_epoch(self.test_loader, 'test')
            best_epoch_index = next((i for i, metrics in enumerate(epoch_metrics_history)
                                     if metrics['epoch'] == best_epoch), 0)
            final_metrics = {**epoch_metrics_history[best_epoch_index],
                             **{f'test_{k}': v for k, v in test_metrics.items() if k != 'phase'}}

            test_opt_metric = test_metrics[self.optimization_metric]
            is_better_test_global = (self.optimization_metric == 'loss' and test_opt_metric < self.best_test_global_metric) or \
                                 (self.optimization_metric == 'f1_class1' and test_opt_metric > self.best_test_global_metric)

            if is_better_test_global:
                self.best_test_global_metric = test_opt_metric
                best_test_model_path = Path(os.path.join(self.output_dir, 'best_test_model.pth'))
                if best_test_model_path.exists():
                    os.remove(best_test_model_path)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'hyperparameters': params,
                    'epoch': epoch,
                    f'best_{self.optimization_metric}': test_opt_metric,
                    'trial_id': trial_id
                }, best_test_model_path)
                print(f'Better global test {self.optimization_metric}: {self.best_test_global_metric}')

        else:
            final_metrics = best_epoch_metrics_history[-1]

        self.trials_history.append(final_metrics)

        trials_df = pd.DataFrame(self.trials_history)
        trials_df.to_csv(self.output_dir / 'hyperopt_trials.csv', index=False)

        return {
            'loss': best_metric if self.optimization_metric == 'loss' else -best_metric,
            'status': STATUS_OK,
            'epoch_metrics': best_epoch_metrics_history,
            'trial_id': trial_id,
            'best_epoch': best_epoch
        }

    def optimize(self):
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.num_trials,
            trials=trials
        )

        all_trials_metrics = []
        for trial in trials.trials:
            if 'epoch_metrics' in trial['result']:
                all_trials_metrics.extend(trial['result']['epoch_metrics'])

        detailed_trials_df = pd.DataFrame(all_trials_metrics)
        detailed_trials_df.to_csv(self.output_dir / 'hyperopt_detailed_trials.csv', index=False)

        # Salva le informazioni sul miglior trial
        best_model_checkpoint = torch.load(self.output_dir / 'best_model.pth', weights_only=False, map_location=torch.device(self.device))
        best_model_info = {
            'trial_id': self.best_trial_id,
            'hyperparameters': best_model_checkpoint['hyperparameters'],
            f'best_{self.optimization_metric}': self.best_global_metric,
            'epoch': best_model_checkpoint['epoch']
        }

        pd.DataFrame([best_model_info]).to_csv(
            self.output_dir / 'best_model_info.csv',
            index=False
        )

        print("\nBest hyperparameters found:")
        for param, value in best_model_checkpoint['hyperparameters'].items():
            print(f"{param}: {value}")
        print(f"Best {self.optimization_metric}: {self.best_global_metric}")

        return best, trials

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

        # Load state dict with strict=False to ignore missing and unexpected keys
        incompatible_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Log information about the ignored keys
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

    def pad_image(self, image, mask=None):
        """
        Pad image and mask to required input size, aligning to top-left
        Returns padded image, mask, and valid region mask
        """
        _, h, w = image.shape
        pad_h = max(0, self.data_input_size[0] - h)
        pad_w = max(0, self.data_input_size[1] - w)

        padding = (0, pad_w, 0, pad_h)

        padded_image = F.pad(image, padding, mode='constant', value=0)

        valid_mask = torch.ones((1, h, w), device=image.device)
        padded_valid_mask = F.pad(valid_mask, padding, mode='constant', value=0)

        if mask is not None:
            padded_mask = F.pad(mask, padding, mode='constant', value=0)
            return padded_image, padded_mask, padded_valid_mask

        return padded_image, padded_valid_mask

    def unpad_prediction(self, pred, valid_mask):
        """Extract original size prediction from padded output"""
        # Find the non-zero region in the valid mask
        nonzero_indices = torch.nonzero(valid_mask.squeeze())
        if len(nonzero_indices) == 0:
            return pred

        h, w = nonzero_indices.max(0)[0] + 1
        return pred[:, :, :h, :w]

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

    def train_epoch(self, epoch):
        if epoch == self.sfreeze_encoder_after and hasattr(self.model,'encoder'):
            print('Sfreezing encoder....')
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            print(f"Encoder unfrozen at epoch {epoch}.")
        self.model.train()
        epoch_loss = 0
        num_samples = 0
        cumulative_conf_matrix = np.zeros((2, 2))
        with tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.max_epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                masks = batch['mask']
                valid_masks = batch['valid_mask']

                masks = masks.to(self.device)
                valid_masks = valid_masks.to(self.device)
                if self.temporal_mode == TemporalMode.TIMESERIES.value:
                    images = batch['image']
                    images = images.to(self.device)
                    additional_images = batch['additional_images'].to(self.device)
                    outputs = self.model(images, additional_images)
                    num_samples += images.size(0)
                elif self.temporal_mode == TemporalMode.SINGLE.value:
                    images = batch['image']
                    images = images.to(self.device)
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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # outputs = torch.sigmoid(outputs)
                # pred_np = (outputs.detach().cpu().numpy() > 0.5).astype(np.int64)
                # target_np = masks.cpu().numpy().astype(np.int64)
                # all_targets.extend(target_np.flatten())
                # all_predictions.extend(pred_np.flatten())
        # conf_matrix = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
        return epoch_loss, cumulative_conf_matrix, num_samples

    def test(
            self,
            calculate_forest_type_metrics: bool = False
    ):
        if self.test_dataset is None:
            print("No test dataset provided")
            return

        best_hyperparams = self.load_best_model()

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=int(best_hyperparams['batch_size']),
            shuffle=False,
            num_workers=self.num_workers_dl,
            pin_memory=True
        )

        # todo da rifare con alfa e beta ottimizzati
        self.criterion = TverskyLoss()

        test_metrics = self.evaluate_epoch(self.test_loader, 'test', calculate_forest_type_metrics=calculate_forest_type_metrics)
        test_metrics.update(best_hyperparams)
        # test_metrics[f'best_{self.optimization_metric}'] = self.best_global_metric

        test_results = pd.DataFrame([test_metrics])
        test_results.to_csv(self.output_dir / 'test_metrics.csv', index=False)

        print("\nTest Results:")
        print(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score (class 0): {test_metrics['f1_class0']:.4f}")
        print(f"F1 Score (class 1): {test_metrics['f1_class1']:.4f}")
        print(f"Loss: {test_metrics['loss']:.4f}")

    def single_image_test(self):
        if not self.test_dataset:
            print("No test dataset provided")
            return

        best_hyperparams = self.load_best_model()
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=int(best_hyperparams['batch_size']),
            shuffle=False,
            num_workers=self.num_workers_dl,
            pin_memory=True
        )

        self.model.eval()
        self.criterion = TverskyLoss()

        image_tile_registry = defaultdict(list)

        for batch in tqdm(self.test_loader, desc="Registering image tiles"):
            batch['position'] = transform_batch_positions(batch['position'])
            for i, image_id in enumerate(batch['image_id']):
                tile_info = {
                    'tile': batch['image'][i],
                    'position': batch['position'][i],
                    'valid_mask': batch['valid_mask'][i],
                    'true_mask': batch['mask'][i]
                }
                if self.temporal_mode == TemporalMode.TIMESERIES.value:
                    tile_info['additional_images'] = batch['additional_images'][i]
                image_tile_registry[image_id].append(tile_info)

        image_level_results = {}
        for image_id, tiles in tqdm(image_tile_registry.items(), desc="Processing images"):
            sorted_tiles = sorted(tiles, key=lambda x: x['position'])

            max_height = max(tile['position'][0] + tile['tile'].shape[1] for tile in sorted_tiles)
            max_width = max(tile['position'][1] + tile['tile'].shape[2] for tile in sorted_tiles)

            pred_tiles = []
            for tile_batch in chunked(sorted_tiles, 8):
                batch_tiles = [t['tile'] for t in tile_batch]
                batch_tiles = torch.stack(batch_tiles).to(self.device)
                with torch.no_grad():
                    if self.temporal_mode == TemporalMode.TIMESERIES.value:
                        batch_additional_images = [t['additional_images'] for t in tile_batch]
                        batch_additional_images = torch.stack(batch_additional_images).to(self.device)
                        preds = self.model(batch_tiles, batch_additional_images)
                    elif self.temporal_mode == TemporalMode.SINGLE.value:
                        preds = self.model(batch_tiles)
                    batch_preds = torch.sigmoid(preds).cpu()
                    pred_tiles.extend(batch_preds)
            reconstructed_pred = self._reconstruct_image(
                pred_tiles,
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

            binary_pred = (reconstructed_pred > 0.5).float()

            metrics = self._compute_image_metrics(binary_pred.squeeze(), reconstructed_true_mask.squeeze())
            metrics['image_id'] = image_id

            image_level_results[image_id] = {
                'metrics': metrics,
                'pred_array': binary_pred.numpy(),
                'true_mask_array': reconstructed_true_mask.numpy()
            }

            self.save_georeferenced_images(image_id, binary_pred.numpy(), reconstructed_true_mask.numpy())

            del reconstructed_pred, reconstructed_true_mask, binary_pred
            torch.cuda.empty_cache()

        self._save_aggregated_results(image_level_results)

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
            TN, FP, FN, TP = conf_matrix.ravel()
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
