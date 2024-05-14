import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, repeat
# from torchvision.models.vision_transformer import VisionTransformer, ViT_B_32_Weights


# class VisionTransformerEncoder(VisionTransformer):
#     def forward(self, x: torch.Tensor):
#         # Reshape and permute the input tensor
#         x = self._process_input(x)
#         n = x.shape[0]
#
#         # Expand the class token to the full batch
#         batch_class_token = self.class_token.expand(n, -1, -1)
#         x = torch.cat([batch_class_token, x], dim=1)
#
#         x = self.encoder(x)
#         return x[:, 1:, :], x[:, 0, :]


def build_vit_extractor(args, image_size=224, patch_size=32, hidden_dim=768, num_layers=12, mlp_dim=3072):
    num_heads = hidden_dim // 64
    vit = VisionTransformerEncoder(image_size=image_size, patch_size=patch_size, hidden_dim=hidden_dim,
                                   num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim)
    # load pretrained model
    if len(args['vit_checkpoint']) != 0:
        vit.load_state_dict(args['vit_checkpoint'])
    # weights = ViT_B_32_Weights.IMAGENET1K_V1
    # vit.load_state_dict(weights.get_state_dict(progress=True), strict=False)
    return vit


def default(val, d):
    return val if val is not None else d


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            context_dim=None,
            dim_head=64,
            heads=8,
            parallel_ff=False,
            ff_mult=4,
            norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if self.ff is not None:
            out = out + self.ff(x)

        return out
