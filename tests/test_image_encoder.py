import pytest
import numpy as np
import random
import torch
from models.image_encoder import PatchEmbed
from models.image_encoder import Attention


torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
random.seed(123)
np.random.seed(123)


def test_patch_embed():
    embed = PatchEmbed(embed_dim=768)
    input = torch.rand(1, 3, 256, 256)
    res = embed(input)

    # the output channel dim should match embed dim
    assert res.shape == (1, 16, 16, 768)


def test_attention():
    attn = Attention(768, num_heads=8)
    input = torch.rand(1, 16, 16, 768)
    res = attn(input)

    assert res.shape == (1, 16, 16, 768)
    assert hasattr(attn, 'rel_pos_h') == False
    assert hasattr(attn, 'rel_pos_w') == False


def test_attention_with_relative_positional_encoding():
    # test attention with relative positional encoding
    dim = 768
    num_heads = 8

    # in context of vision transformers self-attention should output same shape as input
    attn = Attention(dim, num_heads=num_heads, use_rel_pos=True, input_size=(16, 16))
    input = torch.ones(1, 16, 16, 768)
    res = attn(input)

    assert res.shape == (1, 16, 16, 768)
    assert hasattr(attn, 'rel_pos_h') == True
    assert hasattr(attn, 'rel_pos_w') == True

    head_dim = dim // num_heads
    B, H, W, _ = input.shape
    assert attn.rel_pos_h.shape == ((2*H - 1, head_dim))
    assert attn.rel_pos_w.shape == ((2*W - 1, head_dim))

