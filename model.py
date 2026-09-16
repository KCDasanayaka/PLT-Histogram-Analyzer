# ============================================================
# model.py
# PLT ROI Peak 50% V4 Model
# ============================================================

from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# Default configuration
# ============================================================

IMAGE_SIZE = 512

DEFAULT_THRESHOLD = 0.35


# ============================================================
# Convolution block
# Must match the training notebook
# ============================================================

class ConvBlock(nn.Module):

    def __init__(self, a, b, drop=0):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                a,
                b,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(b),

            nn.SiLU(),

            nn.Conv2d(
                b,
                b,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(b),

            nn.SiLU(),

            nn.Dropout2d(drop)
            if drop
            else nn.Identity()
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# Small U-Net
# ============================================================

class SmallUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.p = nn.MaxPool2d(2)

        # Encoder
        self.e1 = ConvBlock(3, 32)

        self.e2 = ConvBlock(32, 64)

        self.e3 = ConvBlock(64, 128)

        self.e4 = ConvBlock(
            128,
            256,
            0.1
        )

        # Bottleneck
        self.b = ConvBlock(
            256,
            384,
            0.15
        )

        # Decoder
        self.u4 = nn.ConvTranspose2d(
            384,
            256,
            kernel_size=2,
            stride=2
        )

        self.d4 = ConvBlock(
            512,
            256
        )

        self.u3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.d3 = ConvBlock(
            256,
            128
        )

        self.u2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.d2 = ConvBlock(
            128,
            64
        )

        self.u1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.d1 = ConvBlock(
            64,
            32
        )

        self.out = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.e1(x)

        e2 = self.e2(
            self.p(e1)
        )

        e3 = self.e3(
            self.p(e2)
        )

        e4 = self.e4(
            self.p(e3)
        )

        b = self.b(
            self.p(e4)
        )

        x = self.d4(
            torch.cat(
                [
                    self.u4(b),
                    e4
                ],
                dim=1
            )
        )

        x = self.d3(
            torch.cat(
                [
                    self.u3(x),
                    e3
                ],
                dim=1
            )
        )

        x = self.d2(
            torch.cat(
                [
                    self.u2(x),
                    e2
                ],
                dim=1
            )
        )

        x = self.d1(
            torch.cat(
                [
                    self.u1(x),
                    e1
                ],
                dim=1
            )
        )

        return self.out(x)


# ============================================================
# Safe device selection
# ============================================================

def get_device():

    if torch.cuda.is_available():

        try:

            test = torch.zeros(
                1,
                device="cuda"
            )

            test = test + 1

            return torch.device("cuda")

        except Exception:

            pass

    return torch.device("cpu")


# ============================================================
# Load checkpoint
# ============================================================

def load_model(model_path=None):

    if model_path is None:

        raise FileNotFoundError(
            "Model path was not provided."
        )

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{model_path}"
        )

    device = get_device()

    model = SmallUNet().to(
        device
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # Checkpoint generated by V4 training
    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

        image_size = checkpoint.get(
            "image_size",
            IMAGE_SIZE
        )

        saved_threshold = checkpoint.get(
            "threshold",
            DEFAULT_THRESHOLD
        )

    else:

        state_dict = checkpoint

        image_size = IMAGE_SIZE

        saved_threshold = (
            DEFAULT_THRESHOLD
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return (
        model,
        model_path,
        device,
        int(image_size),
        float(saved_threshold)
    )