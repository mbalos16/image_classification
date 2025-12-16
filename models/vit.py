import torch
import torchvision.transforms as transforms


def create_patches(x, patch_size: int, stride: int):
    patches = torch.nn.functional.unfold(
        x, kernel_size=patch_size, dilation=1, padding=0, stride=stride
    )
    return patches


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=in_channels)

        # Attention implementation
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=2,
            batch_first=True,
            dropout=0.1,
        )

        self.norm_2 = torch.nn.LayerNorm(normalized_shape=in_channels)

        # Multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=in_channels * 2),
            torch.nn.LayerNorm(normalized_shape=in_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_channels * 2, out_features=in_channels),
        )

    def forward(self, x):
        # Apply the first norm
        norm_1 = self.norm_1(x)

        # Apply multi-head attention (inputX3 because of Q, K, V)
        # Get the first possition and add X (residual connection)
        x = self.multihead_attn(norm_1, norm_1, norm_1)[0] + x

        # Apply the second norm
        norm_2 = self.norm_2(x)

        # Apply the multiLayer perceptron and add x (residual connection)
        x = self.mlp(norm_2) + x

        return x


class Vit(torch.nn.Module):
    def __init__(self, trans_blocks=3):
        super().__init__()
        # Transformation of the patches in a fully connected layer
        self.h_vectors = torch.nn.Linear(in_features=768, out_features=128)
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock() for block in range(trans_blocks)]
        )

        self.output = torch.nn.Linear(in_features=128, out_features=6)

        # Possitional embeddings
        self.pos_embedding = torch.nn.Parameter(
            torch.empty(1, 64, 128).normal_(std=0.02)
        )

    def forward(self, x: torch.Tensor):
        # Create the patches from the images
        patches = create_patches(x, patch_size=16, stride=16)

        # Project the patches to the desired dimension: 128
        x = self.h_vectors(patches.transpose(1, 2))

        # Add positional embeddings
        x += self.pos_embedding

        # Pass through the transformer layers
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)

        # Gettting the first patch as repr of the full image
        x = x[:, 0, :]

        x = self.output(x)
        return x
