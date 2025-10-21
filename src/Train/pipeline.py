import os
from src.Data.creator import DatasetCreator
from src.Train.explainer import UNetExplainer
from src.Train.trainer import UNetTrainer
from pathlib import Path

from src.Utils.enums import TemporalMode
from src.Utils.functions import validate_execution_config, validate_model_config


class Runner():
    def __init__(self, data_config, model_config, model_name, additional_tag=''):
        self.training_model_config = model_config['training']
        self.data_config = data_config
        self.pretrained_config = validate_model_config(model_config, model_name)
        self.run_train, self.run_test, self.run_explainer = validate_execution_config(model_config)
        self.additional_tag = additional_tag


    def run(self):
        base_dir = Path(self.data_config['base_dir'])
        temporal_mode = self.training_model_config['temporal_mode']
        if not temporal_mode:
            raise ValueError("temporal_mode must be set in the model config")
        elif temporal_mode not in [mode.value for mode in TemporalMode]:
            raise ValueError(f"temporal_mode must be one of {[mode.value for mode in TemporalMode]}")

        data = {
            'train_masks_dir': os.path.join(base_dir, self.data_config['train_masks_subdir']),
            'test_masks_dir': os.path.join(base_dir, self.data_config['test_masks_subdir']),
            'forest_type_path': os.path.join(base_dir, self.data_config['forest_type_path']),
            'mask_values_map': self.data_config['mask_values_map'],
            'train_images_dir': None,
            'test_images_dir': None,
            'additional_images_dirs_train': None,
            'additional_images_dirs_test': None,
            'timeseries1_images_dirs': {
                'train': None,
                'test': None
            },
            'timeseries2_images_dirs': {
                'train': None,
                'test': None
            }
        }
        if temporal_mode == TemporalMode.SINGLE.value:
            print('Running in single mode')
            data['train_images_dir'] = os.path.join(base_dir, self.data_config['train_images_subdir'])
            data['test_images_dir'] = os.path.join(base_dir, self.data_config['test_images_subdir'])
            print(f'Getting data from {data["train_images_dir"]} and {data["test_images_dir"]}')
        elif temporal_mode == TemporalMode.TIMESERIES.value:
            print('Running in time series mode')
            data['train_images_dir'] = os.path.join(base_dir, self.data_config['train_images_subdir'])
            data['test_images_dir'] = os.path.join(base_dir, self.data_config['test_images_subdir'])
            data['additional_images_dirs_train'] = [os.path.join(base_dir, path) for path in self.data_config['additional_images_dirs']['train']]
            data['additional_images_dirs_test'] = [os.path.join(base_dir, path) for path in self.data_config['additional_images_dirs']['test']]
            print(f'Getting data from {data["additional_images_dirs_train"]} and {data["additional_images_dirs_test"]}')

        print(f'Converting masks values {data["mask_values_map"]} into ones')

        ds_creator = DatasetCreator(data_config=self.data_config)

        print(self.training_model_config)
        peft_encoder=self.training_model_config['peft_encoder']
        fusion_mode = self.training_model_config['fusion_mode']
        fusion_technique = self.training_model_config['fusion_technique']

        base_res_dir = 'results'
        os.makedirs(base_res_dir, exist_ok=True)

        output_dir = os.path.join(base_res_dir, self.data_config['dataset_tag'], f"{self.training_model_config['output_dir']}_{self.data_config['execution_tag']}{self.additional_tag}")

        datasets = {}
        if self.run_train:
            train_dataset, val_dataset = ds_creator.create_train_val_datasets(
                tile_size=self.training_model_config['tile_size'],
                enable_augmentation=self.data_config['augmentation'],
                val_split=0.2,
                **data
            )

            datasets['train'] = train_dataset
            datasets['val'] = val_dataset

        if self.run_test:
            test_dataset = ds_creator.create_test_dataset(
                tile_size=self.training_model_config['tile_size'],
                enable_forest_type=self.data_config['remove_forest_type_on_test'],
                **data
            )
            datasets['test'] = test_dataset


        if self.training_model_config['mode'] == 'segmentation':
            trainer = UNetTrainer(
                model_name=self.pretrained_config['name'],
                train_dataset=datasets.get('train'),
                val_dataset=datasets.get('val'),
                test_dataset=datasets.get('test'),
                model_tile_size=self.pretrained_config['tile_size'],
                data_tile_size=self.training_model_config['tile_size'],
                num_trials=self.training_model_config['num_trials'],
                max_epochs=self.training_model_config['max_epochs'],
                early_stopping_patience=self.training_model_config['early_stopping_patience'],
                optimization_metric=self.training_model_config['optimization_metric'],
                sfreeze_encoder_after=self.training_model_config['sfreeze_encoder_after'],
                freeze_encoder=self.training_model_config['freeze_encoder'],
                num_workers_dl=self.training_model_config['num_workers_dl'],
                output_dir=output_dir,
                device=self.training_model_config['device'],
                temporal_mode=temporal_mode,
                peft_encoder=peft_encoder,
                num_additional_images=len(self.data_config['additional_images_dirs']['train']),
                fusion_mode = fusion_mode,
                fusion_technique = fusion_technique,
                batch_size = self.training_model_config['batch_size'],
                rank = self.training_model_config['rank'],
            )
            if self.run_train:
                trainer.optimize()
            if self.run_test:
                trainer.test(calculate_forest_type_metrics=self.data_config['remove_forest_type_on_test'])
                trainer.single_image_test()

            if self.run_explainer:
                test_dataset = ds_creator.create_test_dataset(
                    tile_size=self.training_model_config['tile_size'],
                    enable_forest_type=self.data_config['remove_forest_type_on_test'],
                    **data
                )
                datasets['test'] = test_dataset
                explainer = UNetExplainer(
                    model_name=self.pretrained_config['name'],
                    train_dataset=datasets.get('train'),
                    val_dataset=datasets.get('val'),
                    test_dataset=datasets.get('test'),
                    model_tile_size=self.pretrained_config['tile_size'],
                    data_tile_size=self.training_model_config['tile_size'],
                    optimization_metric=self.training_model_config['optimization_metric'],
                    sfreeze_encoder_after=self.training_model_config['sfreeze_encoder_after'],
                    freeze_encoder=self.training_model_config['freeze_encoder'],
                    num_workers_dl=self.training_model_config['num_workers_dl'],
                    output_dir=output_dir,
                    device=self.training_model_config['device'],
                    temporal_mode=temporal_mode,
                    peft_encoder=peft_encoder,
                    num_additional_images=len(self.data_config['additional_images_dirs']['train']),
                    fusion_mode=fusion_mode,
                    fusion_technique=fusion_technique,
                    batch_size=self.training_model_config['batch_size'],
                    rank=self.training_model_config['rank'],
                )


                explainer.xai_band_occlusion_test()
                explainer.analyze_xai_data()
        else:
            Exception('Select valid mode segmentation!')