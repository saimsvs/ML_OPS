import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


class UNetLoMix(nn.Module):

    def __init__(
        self,
        n_channels=1,
        n_classes=1
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.inc = DoubleConv(
            n_channels,
            64
        )

        self.down1 = Down(
            64,
            128
        )

        self.down2 = Down(
            128,
            256
        )

        self.down3 = Down(
            256,
            512
        )

        self.down4 = Down(
            512,
            1024
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.up1 = Up(
            1024,
            512
        )

        self.up2 = Up(
            512,
            256
        )

        self.up3 = Up(
            256,
            128
        )

        self.up4 = Up(
            128,
            64
        )

        # -------------------------
        # Prediction heads
        # One for each decoder stage
        # -------------------------

        self.out1 = OutConv(
            512,
            n_classes
        )

        self.out2 = OutConv(
            256,
            n_classes
        )

        self.out3 = OutConv(
            128,
            n_classes
        )

        self.out4 = OutConv(
            64,
            n_classes
        )

    def forward(self, x):

        # -------------------------
        # Encoder
        # -------------------------

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        # -------------------------
        # Decoder
        # -------------------------

        d1 = self.up1(
            x5,
            x4
        )

        d2 = self.up2(
            d1,
            x3
        )

        d3 = self.up3(
            d2,
            x2
        )

        d4 = self.up4(
            d3,
            x1
        )

        # -------------------------
        # Create segmentation logits
        # -------------------------

        p1 = self.out1(d1)

        p2 = self.out2(d2)

        p3 = self.out3(d3)

        p4 = self.out4(d4)

        # -------------------------
        # Make all predictions
        # the same 256 x 256 size
        # -------------------------

        target_size = x.shape[2:]

        p1 = F.interpolate(
            p1,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        p2 = F.interpolate(
            p2,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        p3 = F.interpolate(
            p3,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        # p4 is already 256 x 256

        # Return four decoder predictions
        return [
            p1,
            p2,
            p3,
            p4
        ]


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNetLoMix(
        n_channels=1,
        n_classes=1
    ).to(device)

    # Fake BUSI image
    x = torch.randn(
        1,
        1,
        256,
        256
    ).to(device)

    outputs = model(x)

    print()
    print("Number of predictions:", len(outputs))

    for i, output in enumerate(outputs):
        print(
            "Prediction",
            i + 1,
            "shape:",
            output.shape
        )

    print()
    print("Test completed successfully.")