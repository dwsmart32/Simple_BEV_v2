import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helpers

# attention

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# main class

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim)) # channel mixinig
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))
        # random initialization of parameters
    
    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        
        return x * self.weights + self.biases
        

class RadarTabTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.   # in paper, they reserve dimension / 8 for category shared embedding
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim) # embed with total_tokens

        # take care of shared category embed

        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)

            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous) # num_continuos 10
            self.norm = nn.LayerNorm(num_continuous)
            self.cont_class_num = 8
            self.class_weights = nn.Parameter(torch.randn(self.cont_class_num))
            self.class_embedding = nn.Parameter(torch.randn(self.cont_class_num, dim)) # channel mixinig
        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous

        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )
        
    def forward(self, x_categ, x_cont, return_attn = False):
        xs = []

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset # offset gives special numbers to categories
            categ_embed = self.category_embed(x_categ) # five values get their own embeddings => from 34 embeddings
            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1) # each class embeddings are concated
            
            xs.append(categ_embed)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std
        
            x_cont = self.numerical_embedder(x_cont) # mix all numerical values with embeddings
######################embedding separated with radar info class########################

            x_cont[0][0] += self.class_weights[0] * self.class_embedding[0]
            x_cont[0][1] += self.class_weights[0] * self.class_embedding[0]
            x_cont[0][2] += self.class_weights[0] * self.class_embedding[0]
            x_cont[0][3] += self.class_weights[1] * self.class_embedding[1]
            x_cont[0][4] += self.class_weights[2] * self.class_embedding[2]
            x_cont[0][5] += self.class_weights[2] * self.class_embedding[2]
            x_cont[0][6] += self.class_weights[3] * self.class_embedding[3] 
            x_cont[0][7] += self.class_weights[3] * self.class_embedding[3]
            x_cont[0][8] += self.class_weights[4] * self.class_embedding[4]
            x_cont[0][9] += self.class_weights[4] * self.class_embedding[4]
            x_cont[0][10] += self.class_weights[5] * self.class_embedding[5]
            x_cont[0][11] += self.class_weights[5] * self.class_embedding[5]
            x_cont[0][12] += self.class_weights[6] * self.class_embedding[6]
            
#######################################################################################

            xs.append(x_cont)
            x = torch.cat(xs, dim = 1)
            b = x.shape[0]
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)
            
            x, attns = self.transformer(x, return_attn = True)
            x = x[:, 0]
            logits = self.to_logits(x)
            
        if not return_attn:
            return logits

        return logits, attns
