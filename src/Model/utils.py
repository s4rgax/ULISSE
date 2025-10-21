from enum import Enum

import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets, valid_mask=None):
        # Conversione in probabilità con controllo
        inputs = torch.sigmoid(inputs)

        # Controlli di sicurezza
        if torch.isnan(inputs).any():
            print("NaN rilevato nelle probabilità")
            inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        if torch.isinf(inputs).any():
            print("Inf rilevato nelle probabilità")
            inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        # Applica la maschera valida se fornita
        if valid_mask is not None:
            valid_mask = valid_mask.to(inputs.device)
            inputs = inputs * valid_mask
            targets = targets * valid_mask

        # Appiattisci gli input e i target
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives con epsilon
        eps = 1e-7  # Epsilon più grande per stabilità numerica
        TP = (inputs * targets).sum() + eps
        FP = ((1 - targets) * inputs).sum() + eps
        FN = (targets * (1 - inputs)).sum() + eps

        # Aggiungi controllo prima della divisione
        denominator = TP + self.alpha * FP + self.beta * FN + self.smooth

        if denominator < eps:
            print("Attenzione: denominatore troppo piccolo nella Tversky Loss")
            denominator = eps

        Tversky = TP / denominator

        # Controllo finale sul risultato
        if torch.isnan(Tversky).any() or torch.isinf(Tversky).any():
            print(f"Problema numerico nel calcolo della loss: TP={TP}, FP={FP}, FN={FN}")
            return torch.tensor(0.5, device=inputs.device, requires_grad=True)

        return 1 - Tversky



def compute_sam_distance(series1, series2):
    """
    Compute Spectral Angle Mapper distance between two temporal series

    Args:
        series1: Tensor of shape (N, T, C) where N is number of pixels, T is temporal length, C is channels
        series2: Tensor of shape (N, T, C)
    Returns:
        Tensor of shape (N, C) containing SAM distances for each channel
    """
    # Normalize the vectors
    series1_norm = series1 / (torch.norm(series1, dim=1, keepdim=True) + 1e-10)
    series2_norm = series2 / (torch.norm(series2, dim=1, keepdim=True) + 1e-10)

    # Compute cosine similarity
    cos_sim = torch.sum(series1_norm * series2_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Convert to angle in radians
    return torch.acos(cos_sim)