from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        # use PyTorch cross-entropy which combines LogSoftmax and NLLLoss
        return nn.functional.cross_entropy(logits, target)

class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Define CNN architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
                nn.MaxPool2d(2),
                # Fourth conv block (added)
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Forward through convolutional layers
        features = self.features(z)
        
        # Global average pooling
        pooled = self.adaptive_pool(features)
        
        # Flatten and pass through fully connected layers
        flattened = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flattened)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Shared encoder (downsampling) split into blocks so we can add skip connections.
        # enc1 -> outputs 32 channels (after first pool)
        # enc2 -> outputs 64 channels (after second pool)
        # enc3 -> outputs 128 channels (after third pool)
        # enc4 -> outputs 256 channels (after fourth pool)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Segmentation head (upsample to input size) with simple skip additions
        # We keep channels equal so we can add encoder features directly (no concat)
    # Now that we have an extra encoder block (256), add one more upsample step
        self.seg_up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_up4 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)

        # Depth head (upsample to input size) with same skip additions
        self.depth_up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.depth_up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.depth_up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.depth_up4 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # Normalize input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder forward (capture intermediate features for skip connections)
        e1 = self.enc1(z)   # (b, 32, h/2, w/2)
        e2 = self.enc2(e1)  # (b, 64, h/4, w/4)
        e3 = self.enc3(e2)  # (b,128,h/8,w/8)
        features = self.enc4(e3)  # (b,256,h/16,w/16)

        # Segmentation decoder with skip additions
        x = self.seg_up1(features)  # -> (b,128,h/8,w/8)
        x = F.relu(x)
        x = x + e3  # simple skip add
        x = self.seg_up2(x)  # -> (b,64,h/4,w/4)
        x = F.relu(x)
        x = x + e2
        x = self.seg_up3(x)  # -> (b,32,h/2,w/2)
        x = F.relu(x)
        x = x + e1
        logits = self.seg_up4(x)  # -> (b,num_classes,h,w)

        # Depth decoder with skip additions
        d = self.depth_up1(features)
        d = F.relu(d)
        d = d + e3
        d = self.depth_up2(d)
        d = F.relu(d)
        d = d + e2
        d = self.depth_up3(d)
        d = F.relu(d)
        d = d + e1
        depth = self.depth_up4(d).squeeze(1)  # (b, h, w)
        depth = torch.sigmoid(depth)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()