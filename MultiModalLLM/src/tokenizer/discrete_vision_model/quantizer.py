import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_embed, embed_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.num_vocab = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_vocab, 1.0 / self.num_vocab)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.num_vocab} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    # def l2norm(self, t):
    #     return F.normalize(t, p = 2, dim = -1)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        bz = z.shape[0]
        z_flattened = z.view(-1, self.embed_dim)
        #print('z_flattened', z_flattened.shape)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z)**2) + torch.mean((z_q - z.detach())**2)
        else:
            loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = z_q.reshape(bz, -1, z_q.shape[-1])
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):

    def __init__(self, num_vocab, embed_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps
        if codebook_init_path == '':
            if not kmeans_init:
                weight = torch.randn(num_vocab, embed_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_vocab, embed_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_vocab), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        # self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_vocab, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_vocab):
        n = self.cluster_size.sum()
        smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + num_vocab * self.eps) * n)
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight.data.copy_(embed_normalized)


def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))


class NormEMAVectorQuantizer(nn.Module):

    def __init__(self,
                 n_embed,
                 embed_dim,
                 beta,
                 decay=0.99,
                 eps=1e-5,
                 statistic_code_usage=True,
                 kmeans_init=False,
                 codebook_init_path=''):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_vocab = n_embed
        self.beta = beta
        self.decay = decay

        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = EmbeddingEMA(self.num_vocab, self.embed_dim, decay, eps, kmeans_init, codebook_init_path)

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_vocab))
            self.cluster_size = self.cluster_size.to(device)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        # z = rearrange(z, 'b c h w -> b h w c')
        # z = l2norm(z)
        # z_flattened = z.reshape(-1, self.embed_dim)

        z = l2norm(z)
        z_flattened = z.reshape(-1, self.embed_dim)

        self.embedding.init_embed_(z_flattened)

        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)  # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)

        encodings = F.one_hot(encoding_indices, self.num_vocab).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training and self.embedding.update:
            #EMA cluster size

            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            # self.embedding.cluster_size_ema_update(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)

            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        # z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices