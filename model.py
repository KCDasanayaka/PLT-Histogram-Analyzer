from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PLTUNet(nn.Module):
    """Exact U-Net topology used by the V8 Colab model."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256, 0.10)
        self.bottleneck = ConvBlock(256, 384, 0.15)

        self.up4 = nn.ConvTranspose2d(384, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        x = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), e3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), e2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), e1], dim=1))
        return self.output(x)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        # Raw state_dict saved directly.
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a valid model state dictionary.")


def load_model(
    preferred_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[PLTUNet, Path, torch.device]:
    """Load V8 checkpoint; accepts either V8 or the legacy filename."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = []
    if preferred_path:
        candidates.append(Path(preferred_path))

    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    candidates += [
        model_dir / "plt_roi_peak50_unet_v8_best.pt",
        model_dir / "plt_curve_unet_fixed50_best.pt",  # legacy name
    ]

    existing = []
    for p in candidates:
        if p.exists() and p not in existing:
            existing.append(p)
    if not existing:
        raise FileNotFoundError(
            "No PLT model checkpoint found. Put the downloaded V8 best checkpoint in models/\n"
            "as plt_roi_peak50_unet_v8_best.pt (or keep the legacy filename)."
        )

    path = existing[0]
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(checkpoint)

    # Handle DataParallel checkpoints if ever produced.
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    model = PLTUNet().to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch.\n"
            f"Missing keys: {missing[:10]}\nUnexpected keys: {unexpected[:10]}"
        )
    model.eval()
    return model, path, device
