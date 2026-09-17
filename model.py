from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# U-NET BLOCK
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        cin: int,
        cout: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                cin,
                cout,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                cout,
                cout,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(
                nn.Dropout2d(dropout)
            )

        self.block = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


# ============================================================
# V8 PLT U-NET
# ============================================================

class PLTUNet(nn.Module):
    """
    Exact U-Net topology used by the V8 Colab model.
    """

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(
            128,
            256,
            0.10,
        )

        self.bottleneck = ConvBlock(
            256,
            384,
            0.15,
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            384,
            256,
            2,
            2,
        )

        self.dec4 = ConvBlock(
            512,
            256,
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2,
        )

        self.dec3 = ConvBlock(
            256,
            128,
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2,
        )

        self.dec2 = ConvBlock(
            128,
            64,
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            2,
        )

        self.dec1 = ConvBlock(
            64,
            32,
        )

        self.output = nn.Conv2d(
            32,
            1,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        x = self.dec4(
            torch.cat(
                [
                    self.up4(b),
                    e4,
                ],
                dim=1,
            )
        )

        x = self.dec3(
            torch.cat(
                [
                    self.up3(x),
                    e3,
                ],
                dim=1,
            )
        )

        x = self.dec2(
            torch.cat(
                [
                    self.up2(x),
                    e2,
                ],
                dim=1,
            )
        )

        x = self.dec1(
            torch.cat(
                [
                    self.up1(x),
                    e1,
                ],
                dim=1,
            )
        )

        return self.output(x)


# ============================================================
# CHECKPOINT EXTRACTION
# ============================================================

def _extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        # Standard training checkpoint
        for key in (
            "model_state_dict",
            "state_dict",
        ):

            if (
                key in checkpoint
                and isinstance(
                    checkpoint[key],
                    dict,
                )
            ):

                return checkpoint[key]

        # Raw state_dict
        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise ValueError(
        "Checkpoint does not contain "
        "a valid model state dictionary."
    )


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def find_model_checkpoint(
    preferred_path: Optional[str | Path] = None,
) -> Path:

    root = Path(__file__).resolve().parent

    models_dir = root / "models"

    candidates = []

    # Explicit preferred path
    if preferred_path is not None:

        candidates.append(
            Path(preferred_path)
        )

    # Current renamed V8 checkpoint
    candidates.append(
        models_dir
        / "plt_roi_peak50_unet_v8_best.pt"
    )

    # Additional compatible names
    candidates.extend(
        [
            models_dir
            / "plt_roi_peak50_unet_best.pt",

            models_dir
            / "plt_roi_peak50_unet_v8_last.pt",

            models_dir
            / "plt_curve_unet_fixed50_best.pt",

            models_dir / "best.pt",
        ]
    )

    # Check explicit candidates first
    existing = []

    for path in candidates:

        if (
            path.exists()
            and path.is_file()
            and path not in existing
        ):

            existing.append(path)

    if existing:

        return existing[0]

    # --------------------------------------------------------
    # Last fallback:
    # find a model checkpoint in models/
    # --------------------------------------------------------

    if models_dir.exists():

        fallback_files = sorted(
            [
                path
                for path in models_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {
                    ".pt",
                    ".pth",
                    ".bin",
                }
            ]
        )

        if fallback_files:

            return fallback_files[0]

    raise FileNotFoundError(
        "No PLT model checkpoint found.\n\n"
        "Expected:\n"
        "models/plt_roi_peak50_unet_v8_best.pt"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    preferred_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[
    PLTUNet,
    Path,
    torch.device,
]:

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint_path = find_model_checkpoint(
        preferred_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = _extract_state_dict(
        checkpoint
    )

    # --------------------------------------------------------
    # DataParallel compatibility
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Build exact V8 architecture
    # --------------------------------------------------------

    model = PLTUNet().to(device)

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing_keys or unexpected_keys:

        raise RuntimeError(
            "Checkpoint/model architecture mismatch.\n\n"
            f"Missing keys: "
            f"{missing_keys[:10]}\n\n"
            f"Unexpected keys: "
            f"{unexpected_keys[:10]}"
        )

    model.eval()

    return (
        model,
        checkpoint_path,
        device,
    )