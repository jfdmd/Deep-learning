from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input: track_left (n_track, 2) + track_right (n_track, 2) = n_track * 4 features
        input_size = n_track * 4
        output_size = n_waypoints * 2

        # MLP architecture with hidden layers
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Flatten and concatenate track boundaries
        x = torch.cat([track_left, track_right], dim=2)  # (b, n_track, 4)
        x = x.reshape(batch_size, -1)  # (b, n_track * 4)

        # Pass through network
        x = self.network(x)  # (b, n_waypoints * 2)

        # Reshape to waypoints
        x = x.reshape(batch_size, self.n_waypoints, 2)  # (b, n_waypoints, 2)

        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Query embeddings for waypoints (latent array in Perceiver)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Project input features (2D coordinates) to d_model dimension
        self.input_projection = nn.Linear(2, d_model)

        # Transformer decoder for cross-attention
        # Single decoder layer with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation="relu",
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,
        )

        # Output projection from d_model to 2D coordinates
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Concatenate track boundaries
        track = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)

        # Project input features to d_model
        encoder_output = self.input_projection(track)  # (b, 2*n_track, d_model)

        # Create query embeddings for waypoints
        query_idx = torch.arange(self.n_waypoints, device=track_left.device)
        query = self.query_embed(query_idx)  # (n_waypoints, d_model)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)  # (b, n_waypoints, d_model)

        # Apply transformer decoder (cross-attention)
        # Memory is the encoder output (track boundaries)
        # Tgt is the query embeddings (waypoints)
        decoder_output = self.transformer_decoder(
            tgt=query,
            memory=encoder_output,
        )  # (b, n_waypoints, d_model)

        # Project to 2D waypoints
        waypoints = self.output_projection(decoder_output)  # (b, n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # CNN backbone - similar to typical semantic segmentation or depth prediction networks
        # Input: (B, 3, 96, 128)

        # Encoder: progressively reduce spatial dimensions while increasing channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 96x128 -> 48x64
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 48x64 -> 24x32
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 24x32 -> 12x16
        )

        # Adaptive average pooling to get a fixed-size feature vector
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Decoder: fully connected layers to predict waypoints
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)  # (b, 128, 1, 1)
        x = x.flatten(1)  # (b, 128)

        # Decoder
        x = self.fc(x)  # (b, n_waypoints * 2)

        # Reshape to waypoints
        batch_size = image.shape[0]
        x = x.reshape(batch_size, self.n_waypoints, 2)  # (b, n_waypoints, 2)

        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
