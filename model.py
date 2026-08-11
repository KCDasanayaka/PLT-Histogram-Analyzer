import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                output_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                output_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        inputs,
    ):
        return self.block(
            inputs
        )


class UpBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        skip_channels,
        output_channels,
    ):
        super().__init__()

        self.up = (
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=2,
                stride=2,
            )
        )

        self.conv = ConvBlock(
            output_channels
            + skip_channels,
            output_channels,
        )

    def forward(
        self,
        inputs,
        skip,
    ):

        inputs = self.up(
            inputs
        )

        if (
            inputs.shape[-2:]
            != skip.shape[-2:]
        ):
            inputs = F.interpolate(
                inputs,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined = torch.cat(
            [
                skip,
                inputs,
            ],
            dim=1,
        )

        return self.conv(
            combined
        )


class PLTCurveUNet(nn.Module):
    def __init__(
        self,
        base_channels=24,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        self.enc1 = ConvBlock(
            3,
            base_channels,
        )

        self.enc2 = ConvBlock(
            base_channels,
            base_channels * 2,
        )

        self.enc3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.enc4 = ConvBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.bottleneck = ConvBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.dec4 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.dec3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.dec2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.dec1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.head = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        inputs,
    ):

        e1 = self.enc1(
            inputs
        )

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        bottleneck = (
            self.bottleneck(
                self.pool(e4)
            )
        )

        d4 = self.dec4(
            bottleneck,
            e4,
        )

        d3 = self.dec3(
            d4,
            e3,
        )

        d2 = self.dec2(
            d3,
            e2,
        )

        d1 = self.dec1(
            d2,
            e1,
        )

        return self.head(
            d1
        )


def load_plt_model(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    base_channels = int(
        checkpoint.get(
            "base_channels",
            24,
        )
    )

    model = PLTCurveUNet(
        base_channels=base_channels
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    threshold = float(
        checkpoint.get(
            "segmentation_threshold",
            0.35,
        )
    )

    return (
        model,
        checkpoint,
        threshold,
    )