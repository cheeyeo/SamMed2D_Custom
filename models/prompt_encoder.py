import torch
from torch import nn
import numpy as np
from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            image_embedding_size,
            input_image_size,
            mask_in_chans,
            activation=nn.GELU
    ):
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)









class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies
    """

    def __init__(self, num_pos_feats=64, scale=None):
        super(PositionEmbeddingRandom, self).__init__()

        if scale is None or scale <= 0.0:
            scale = 1.0
        
        # TODO: How does register buffer work?
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )


    def _pe_encoding(self, coords):
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # coords = coords @ self.positional_encoding_gaussian_matrix
        coords = coords @ self.positional_encoding_gaussian_matrix.to(torch.float32) # todo
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    

    def forward(self, size):
        """
        Generate positional encoding for a grid of size hxw 
        """
        h, w = size

        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))

        # C x H x W
        return pe.permute(2, 0, 1)


    def forward_with_coords(self, coords_input, size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / size[1]
        coords[:, :, 1] = coords[:, :, 1] / size[0]

        # B x N x C
        return self._pe_encoding(coords.to(torch.float))