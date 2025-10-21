import os.path

from transformers import BitsAndBytesConfig

from src.Model.utils import DecoderBlock
from src.Utils.enums import FusionType, FusionTechnique, PeftMode
from src.pretrain.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
import torch.nn.functional as F

import os.path
import torch
import torch.nn as nn
from peft import get_peft_model, TaskType, LoraConfig, HRAConfig, BoneConfig, AdaLoraConfig, VeraConfig, LoftQConfig


class ResNetUNet(nn.Module):
    def __init__(self, encoder, peft=None, peft_attr={}, data_tile_size=(224, 224), model_input_size=(224, 224),
                 num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.data_tile_size = data_tile_size
        self.model_input_size = model_input_size

        self.needs_sampling = self.data_tile_size != self.model_input_size

        self.encoder.fc = nn.Identity()
        if peft:
            rank = peft_attr.get('lora_rank', 8)
            opt_modules = ["conv1", "conv2", "conv3"]
            if peft == PeftMode.LORA.value:
                peft_config = LoraConfig(
                    r=rank,
                    use_dora=True,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=opt_modules,
                )
                print('LoRA applied to encoder')
            elif peft == PeftMode.QLORA.value:
                peft_config = LoraConfig(
                    init_lora_weights='loftq',
                    bias="none",
                    r=rank,
                    loftq_config=LoftQConfig(loftq_bits=4),
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=opt_modules,
                )
            elif peft == PeftMode.DORA.value:
                peft_config = LoraConfig(
                    r=rank,
                    use_dora=True,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=opt_modules,
                )
                print('DoRA applied to encoder')
            elif peft == PeftMode.QDORA.value:
                peft_config = LoraConfig(
                    r=rank,
                    loftq_config=LoftQConfig(loftq_bits=4),
                    use_dora=True,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=opt_modules,
                )
                print('DoRA applied to encoder')
            elif peft == PeftMode.HRA.value:
                # capire se applicare Grandt schmidt
                peft_config = HRAConfig(
                    r=rank,
                    apply_GS=peft_attr.get('apply_GS', False),
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=opt_modules,
                )
                print('HRA applied to encoder')
            else:
                raise ValueError(f"Unsupported PEFT mode: {peft}")
            self.encoder = get_peft_model(self.encoder, peft_config)

        if self.needs_sampling:
            self.input_upsampling = nn.Upsample(
                size=self.model_input_size,
                mode='bilinear',
                align_corners=True
            )
            self.output_downsampling = nn.Upsample(
                size=self.data_tile_size,
                mode='bilinear',
                align_corners=True
            )

        self.enc_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }

        self.decoder4 = DecoderBlock(2048, 1024)
        self.decoder3 = DecoderBlock(2048, 512)
        self.decoder2 = DecoderBlock(1024, 256)
        self.decoder1 = DecoderBlock(512, 64)
        self.decoder0 = DecoderBlock(128, 64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        if self.needs_sampling:
            x = self.input_upsampling(x)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        x0 = x
        x = self.encoder.maxpool(x)

        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        d4 = self.decoder4(x4)
        d4 = self.upsample(d4)
        d4 = torch.cat([d4, x3], dim=1)

        d3 = self.decoder3(d4)
        d3 = self.upsample(d3)
        d3 = torch.cat([d3, x2], dim=1)

        d2 = self.decoder2(d3)
        d2 = self.upsample(d2)
        d2 = torch.cat([d2, x1], dim=1)

        d1 = self.decoder1(d2)
        d1 = self.upsample(d1)
        d1 = torch.cat([d1, x0], dim=1)

        d0 = self.decoder0(d1)
        final_upsample = self.upsample(d0)
        out = self.final_conv(final_upsample)

        if self.needs_sampling:
            out = self.output_downsampling(out)

        return out

    @classmethod
    def from_pretrained(cls, model_name, peft=None, peft_attr={}, model_input_size=(224, 224),
                        data_tile_size=(224, 224), num_classes=1):
        # uncomment for local
        '''local_path = os.path.abspath(os.path.join('data/models', model_name))

        if not os.path.exists(local_path):
            raise ValueError(f"Model directory not found at: {local_path}")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
            local_path,
            local_files_only=True
        )'''

        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)

        encoder = model.model.vision_encoder
        print(f'Loaded vision encoder from {model_name}')
        if data_tile_size == model_input_size:
            print("Input size matches ResNet requirements, no sampling needed")
        else:
            print(f"Input size {data_tile_size} will be sampled to (224, 224)")
        return cls(encoder, peft=peft, peft_attr=peft_attr, data_tile_size=data_tile_size,
                   model_input_size=model_input_size, num_classes=num_classes)


class MultiLoraResNetUNet(nn.Module):
    def __init__(self, base_encoder, num_additional_images=1,
                 fusion_type=FusionType.MIDDLE,
                 fusion_technique=FusionTechnique.SUM,
                 peft=None,
                 peft_attr={},
                 disable_peft_indexes=None,
                 data_tile_size=(224, 224),
                 model_input_size=(224, 224),
                 num_classes=1,
                 random_init=False):
        super().__init__()
        if num_additional_images < 1:
            raise ValueError("Minimum 2 total images required")
        self.fusion_type = fusion_type
        self.fusion_technique = fusion_technique
        self.data_tile_size = data_tile_size
        self.model_input_size = model_input_size
        self.needs_sampling = data_tile_size != model_input_size
        self.base_encoder = base_encoder
        base_encoder.fc = nn.Identity()

        self.num_total_images = num_additional_images + 1

        if disable_peft_indexes is not None and not isinstance(disable_peft_indexes, list):
            disable_peft_indexes = [disable_peft_indexes]

        if disable_peft_indexes is not None and -1 in disable_peft_indexes:
            disable_peft_indexes = []

        self.encoders = nn.ModuleList()
        for i in range(self.num_total_images):
            use_peft_for_this_encoder = (peft and peft is not None and
                                         (disable_peft_indexes is None or
                                          i not in disable_peft_indexes))

            if use_peft_for_this_encoder:
                rank = peft_attr.get('lora_rank', 8)
                opt_modules = ["conv1", "conv2", "conv3"]
                print(f'Applying rank {rank} PEFT to encoder {i}')
                if peft == PeftMode.LORA.value:
                    peft_config = LoraConfig(
                        r=rank,
                        task_type=TaskType.FEATURE_EXTRACTION,
                        target_modules=opt_modules,
                    )
                    print(f'LoRA applied to encoder {i}')
                elif peft == PeftMode.HRA.value:
                    peft_config = HRAConfig(
                        r=rank,
                        task_type=TaskType.FEATURE_EXTRACTION,
                        target_modules=opt_modules,
                    )
                    print(f'HRA applied to encoder {i}')
                elif peft == PeftMode.DORA.value:
                    peft_config = LoraConfig(
                        r=rank,
                        use_dora=True,
                        task_type=TaskType.FEATURE_EXTRACTION,
                        target_modules=opt_modules,
                    )
                    print(f'DoRA applied to encoder {i}')
                else:
                    raise ValueError(f"Unsupported PEFT mode: {peft}")
                self.encoders.append(get_peft_model(base_encoder, peft_config))
            else:
                if peft:
                    print(f'PEFT disabled for encoder {i}, using base encoder')
                else:
                    print('PEFT not enabled, using base encoder')
                self.encoders.append(base_encoder)

        if random_init:
            print("Randomly initializing model weights")
            for m in self.modules():
                print('Initializing ',m)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        if self.needs_sampling:
            self.input_upsampling = nn.Upsample(
                size=model_input_size,
                mode='bilinear',
                align_corners=True
            )
            self.output_downsampling = nn.Upsample(
                size=data_tile_size,
                mode='bilinear',
                align_corners=True
            )

        if self.fusion_technique == FusionTechnique.CONCATENATION.value:
            self.encoder_compression = nn.ModuleDict({
                'x0': nn.Conv2d(64 * self.num_total_images, 64, kernel_size=1),
                'x1': nn.Conv2d(256 * self.num_total_images, 256, kernel_size=1),
                'x2': nn.Conv2d(512 * self.num_total_images, 512, kernel_size=1),
                'x3': nn.Conv2d(1024 * self.num_total_images, 1024, kernel_size=1),
                'x4': nn.Conv2d(2048 * self.num_total_images, 2048, kernel_size=1),
            })

        self._initialize_decoders()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _stack_features(self, reference_features, additional_features):
        """
        Stacks features from reference and additional images.
        Returns a list of tensors, where each tensor has shape (batch, num_images, channels, height, width)
        """
        stacked_features = []
        for pos in range(len(reference_features)):
            ref_feat = reference_features[pos]
            add_feats = [feat[pos] for feat in additional_features]
            ref_feat_expanded = ref_feat.unsqueeze(1)
            add_feats_stacked = torch.stack([f.unsqueeze(1) for f in add_feats], dim=1)
            add_feats_stacked = torch.squeeze(add_feats_stacked, dim=2)
            combined = torch.cat([ref_feat_expanded, add_feats_stacked], dim=1)
            stacked_features.append(combined)
        return stacked_features

    def _apply_fusion_technique(self, stacked_features, level):
        """
        Applies fusion technique to stacked features
        Input shape: (batch, num_images, channels, height, width)
        """
        if self.fusion_technique == FusionTechnique.CONCATENATION.value:
            batch, num_images, channels, height, width = stacked_features.size()
            reshaped = stacked_features.view(batch, num_images * channels, height, width)
            return self.encoder_compression[level](reshaped)

        elif self.fusion_technique == FusionTechnique.DIFF.value:
            ref = stacked_features[:, 0]
            diffs = [torch.norm(ref - stacked_features[:, i], p=2, dim=1, keepdim=True)
                     for i in range(1, stacked_features.size(1))]
            return torch.sum(torch.stack(diffs, dim=0), dim=0)

        elif self.fusion_technique == FusionTechnique.SUM.value:
            return torch.sum(stacked_features, dim=1)

        raise ValueError(f"Unsupported fusion technique: {self.fusion_technique}")

    def _initialize_decoders(self):
        if self.fusion_type == FusionType.MIDDLE.value:
            self.decoder4 = DecoderBlock(2048, 1024)
            self.decoder3 = DecoderBlock(2048, 512)
            self.decoder2 = DecoderBlock(1024, 256)
            self.decoder1 = DecoderBlock(512, 64)
            self.decoder0 = DecoderBlock(128, 64)
        else:
            self.decoder4_m1 = DecoderBlock(2048, 1024)
            self.decoder3_m1 = DecoderBlock(1024, 512)
            self.decoder2_m1 = DecoderBlock(1024, 256)
            self.decoder1_m1 = DecoderBlock(512, 64)

            self.decoder4_m2 = DecoderBlock(2048, 1024)
            self.decoder3_m2 = DecoderBlock(1024, 512)
            self.decoder2_m2 = DecoderBlock(1024, 256)
            self.decoder1_m2 = DecoderBlock(512, 64)

            self.decoder0 = DecoderBlock(128, 64)

    def _forward_encoder(self, x, encoder):
        if self.needs_sampling:
            x = self.input_upsampling(x)

        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.act1(x)
        x0 = x
        x = encoder.maxpool(x)

        x1 = encoder.layer1(x)
        x2 = encoder.layer2(x1)
        x3 = encoder.layer3(x2)
        x4 = encoder.layer4(x3)

        return x0, x1, x2, x3, x4

    def _forward_middle_fusion(self, reference_features, additional_features):
        stacked_features = self._stack_features(reference_features, additional_features)
        fused_features = [
            self._apply_fusion_technique(stacked_feat, f'x{i}')
            for i, stacked_feat in enumerate(stacked_features)
        ]

        x0, x1, x2, x3, x4 = fused_features
        d4 = self.decoder4(x4)
        d4 = self.upsample(d4)
        d4 = torch.cat([d4, x3], dim=1)

        d3 = self.decoder3(d4)
        d3 = self.upsample(d3)
        d3 = torch.cat([d3, x2], dim=1)

        d2 = self.decoder2(d3)
        d2 = self.upsample(d2)
        d2 = torch.cat([d2, x1], dim=1)

        d1 = self.decoder1(d2)
        d1 = self.upsample(d1)
        d1 = torch.cat([d1, x0], dim=1)

        return self.decoder0(d1)

    def _forward_late_fusion(self, reference_features, *additional_features):
        """
        Performs late fusion with proper stacking of features
        """
        stacked_features = self._stack_features(reference_features, list(additional_features))
        decoded_features = []
        d4 = self.decoder4_m1(stacked_features[4][:, 0])  # Use first image features
        d4 = self.upsample(d4)
        d4 = torch.cat([d4, stacked_features[3][:, 0]], dim=1)

        d3 = self.decoder3_m1(d4)
        d3 = self.upsample(d3)
        d3 = torch.cat([d3, stacked_features[2][:, 0]], dim=1)

        d2 = self.decoder2_m1(d3)
        d2 = self.upsample(d2)
        d2 = torch.cat([d2, stacked_features[1][:, 0]], dim=1)

        d1 = self.decoder1_m1(d2)
        d1 = self.upsample(d1)
        decoded_features.append(d1)

        for i in range(1, stacked_features[0].size(1)):
            d4 = self.decoder4_m2(stacked_features[4][:, i])
            d4 = self.upsample(d4)
            d4 = torch.cat([d4, stacked_features[3][:, i]], dim=1)

            d3 = self.decoder3_m2(d4)
            d3 = self.upsample(d3)
            d3 = torch.cat([d3, stacked_features[2][:, i]], dim=1)

            d2 = self.decoder2_m2(d3)
            d2 = self.upsample(d2)
            d2 = torch.cat([d2, stacked_features[1][:, i]], dim=1)

            d1 = self.decoder1_m2(d2)
            d1 = self.upsample(d1)
            decoded_features.append(d1)

        stacked_decoded = torch.stack(decoded_features, dim=1)
        if self.fusion_technique == FusionTechnique.CONCATENATION.value:
            batch, num_images, channels, height, width = stacked_decoded.size()
            fused = self.encoder_compression['x0'](
                stacked_decoded.view(batch, num_images * channels, height, width)
            )
        elif self.fusion_technique == FusionTechnique.DIFF.value:
            fused = torch.sum(torch.stack([
                torch.abs(stacked_decoded[:, 0] - stacked_decoded[:, i])
                for i in range(1, stacked_decoded.size(1))
            ], dim=0), dim=0)
        else:
            fused = torch.sum(stacked_decoded, dim=1)

        return self.decoder0(fused)

    def forward(self, reference_image, additional_images):
        batch_size = reference_image.size(0)
        num_additional_per_image = additional_images.size(1)

        reference_features = self._forward_encoder(reference_image, self.encoders[0])

        additional_features = []
        for j in range(num_additional_per_image):
            features = self._forward_encoder(
                additional_images[:, j],
                self.encoders[j + 1]
            )
            additional_features.append(features)

        if self.fusion_type == FusionType.MIDDLE.value:
            d0 = self._forward_middle_fusion(reference_features, additional_features)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        final_upsample = self.upsample(d0)
        out = self.final_conv(final_upsample)

        return self.output_downsampling(out) if self.needs_sampling else out

    def replace_encoder_with_base(self, encoder_index):
        """
        Replaces a PEFT-enabled encoder at the specified index with the base encoder.

        Args:
            encoder_index (int): Index of the encoder to replace (0 for reference image encoder,
                                1+ for additional image encoders)

        Returns:
            None
        """
        if encoder_index < 0 or encoder_index >= len(self.encoders):
            raise IndexError(f"Encoder index {encoder_index} out of range (0-{len(self.encoders) - 1})")

        print(f"Replacing PEFT-enabled encoder at index {encoder_index} with base encoder")
        # Replace the encoder with the base encoder
        self.encoders[encoder_index] = self.base_encoder

    @classmethod
    def from_pretrained(cls, model_name, model_input_size=(224, 224), data_tile_size=(224, 224), num_classes=1,
                        peft=None, peft_attr={}, disable_peft_indexes=None, num_additional_images=1, fusion_mode=None, fusion_technique=None,
                        random_init=False):
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
        # uncomment for local
        '''local_path = os.path.abspath(os.path.join('data/models', model_name))
        if not os.path.exists(local_path):
            raise ValueError(f"Model directory not found at: {local_path}")
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
            local_path,
            local_files_only=True
        )'''
        encoder = model.model.vision_encoder
        print(f'Loaded vision encoder from {model_name}')
        if data_tile_size == model_input_size:
            print("Input size matches ResNet requirements, no sampling needed")
        else:
            print(f"Input size {data_tile_size} will be sampled to (224, 224)")
        return cls(
            encoder,
            data_tile_size=data_tile_size,
            model_input_size=model_input_size,
            num_classes=num_classes,
            peft=peft,
            peft_attr=peft_attr,
            disable_peft_indexes=disable_peft_indexes,
            num_additional_images=num_additional_images,
            fusion_type=fusion_mode,
            fusion_technique=fusion_technique,
            random_init=random_init
        )