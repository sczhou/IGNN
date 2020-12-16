# 
# We develop our graph aggregation code based on the N3Net code from
# https://github.com/visinf/n3net/blob/master/src_denoising/models/non_local.py
# Thanks Tobias Plötz and Stefan Roth for their code.
# 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils.network_utils as net_utils
from models.lib import ops
from config import cfg


def compute_distances(xe, ye, I, train=True):
    r"""
    Computes pairwise distances for all pairs of query items and
    potential neighbors.

    :param xe: BxNxE tensor of database (son) item embeddings
    :param ye: BxMxE tensor of query (father) item embeddings
    :param I: BxMxO index tensor that selects O potential neighbors in a window for each item in ye
    :param train: whether to use tensor comprehensions for inference (forward only)

    :return D: a BxMxO tensor of distances
    """

    # xe -> b n e
    # ye -> b m e
    # I  -> b m o
    b,n,e = xe.shape
    m = ye.shape[1]

    if train or not cfg.NETWORK.WITH_WINDOW:
        # D_full -> b m n
        D = ops.euclidean_distance(ye, xe.permute(0,2,1))
        if cfg.NETWORK.WITH_WINDOW:
            # D -> b m o
            D = D.gather(dim=2, index=I) + 1e-5
    else:
        o = I.shape[2]
        # xe_ind -> b m o e
        If = I.view(b, m*o,1).expand(b,m*o,e)
        # D -> b m o
        ye = ye.unsqueeze(3)
        D = -2*ops.indexed_matmul_1_efficient(xe, ye.squeeze(3), I).unsqueeze(3)

        xe_sqs = (xe**2).sum(dim=-1, keepdim=True)
        xe_sqs_ind = xe_sqs.gather(dim=1, index=If[:,:,0:1]).view(b,m,o,1)
        D += xe_sqs_ind
        D += (ye**2).sum(dim=-2, keepdim=True) + 1e-5

        D = D.squeeze(3)
        
    return D

def hard_knn(D, k, I):
    r"""
    input D: b m n
    output Idx: b m k
    """
    score, idx = torch.topk(D, k, dim=2, largest=False, sorted=True)
    if cfg.NETWORK.WITH_WINDOW:
        idx = I.gather(dim=2, index=idx)

    return score, idx

class GraphConstruct(nn.Module):
    r"""
    Graph Construction
    """
    def __init__(self, scale, indexer, k, patchsize, stride, padding=None):
        r"""
        :param scale: downsampling factor
        :param indexer: function for creating index tensor
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphConstruct, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.indexer = indexer
        self.k = k
        self.padding = padding

    def graph_k(self, xe, ye, I):
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        n = xe.shape[1]
        b, m, e = ye.shape
        k = self.k

        # Euclidean Distance
        D = compute_distances(xe, ye, I, train=self.training)

        # hard knn
        # return: b m k
        score_k, idx_k = hard_knn(D, k, I)

        # xe -> b m e n
        # idx-> b m e k
        xe = xe.permute(0,2,1).contiguous()
        xe_e = xe.view(b,1,e,n).expand(b,m,e,n)
        idx_k_e = idx_k.view(b,m,1,k).expand(b,m,e,k)

        if cfg.NETWORK.WITH_DIFF:
            ye_e = ye.view(b,m,e,1).expand(b,m,e,k)
            diff_patch = ye_e-torch.gather(xe_e, dim=3, index=idx_k_e)
        else:
            diff_patch = None

        if cfg.NETWORK.WITH_SCORE:
            score_k = (-score_k/10.).exp()
        else:
            score_k = None

        # score_k: b m k
        # idx_k: b m k
        # diff_patch: b m e k
        return score_k, idx_k, diff_patch

    def forward(self, xe, ye):
        r"""
        :param xe: embedding of son features
        :param ye: embedding of father features

        :return score_k: similarity scores of top k nearest neighbors
        :return idx_k: indexs of top k nearest neighbors
        :return diff_patch: difference vectors between query and k nearest neighbors
        """
        # Convert everything to patches
        H, W = ye.shape[2:]
        xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        ye_patch, padding = ops.im2patch(ye, self.patchsize, self.stride, self.padding, returnpadding=True)

        I = self.indexer(xe_patch, ye_patch)

        if not self.training:
            index_neighbours_cache.clear()

        # bacth, channel, patchsize1, patchsize2, h, w
        _,_,_,_,n1,n2 = xe_patch.shape
        b,ce,e1,e2,m1,m2 = ye_patch.shape

        k = self.k
        n = n1*n2; m=m1*m2; e=ce*e1*e2
        xe_patch = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,e)
        ye_patch = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)

        # Get nearest neighbor volumes
        score_k, idx_k, diff_patch = self.graph_k(xe_patch, ye_patch, I)

        if cfg.NETWORK.WITH_DIFF:
            # diff_patch -> b,m,e,k      b m1*m2 ce e1*e2 k
            diff_patch = abs(diff_patch.view(b, m, ce, e1*e2, k))
            diff_patch = torch.sum(diff_patch,dim=3,keepdim=True)
            diff_patch = diff_patch.expand(b, m, ce, e1*self.scale*e2*self.scale, k)

            # diff_patch: b m ce e1*s*e2*s k; e1==p1, e2==p2;
            # diff_patch -> b k ce e1*s*e2*s m
            diff_patch = diff_patch.permute(0,4,2,3,1).contiguous()
            # diff_patch -> b k*ce e1*s e2*s m1 m2 
            diff_patch = diff_patch.view(b,k*ce,e1*self.scale,e2*self.scale,m1,m2)
            padding_sr = [p*self.scale for p in padding]
            # z_sr -> b k*c_y H*s W*s
            diff_patch = ops.patch2im(diff_patch, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            diff_patch = diff_patch.contiguous().view(b,k*ce,H*self.scale,W*self.scale)

        if cfg.NETWORK.WITH_SCORE:
            # score_k: b,m,k --> b,k,e1*s,e2*s,m1,m2
            score_k = score_k.permute(0,2,1).contiguous().view(b,k,1,1,m1,m2)
            score_k = score_k.view(b,k,1,1,m1,m2).expand(b,k,e1*self.scale,e2*self.scale,m1,m2)
            padding_sr = [p*self.scale for p in padding]
            score_k = ops.patch2im(score_k, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            score_k = score_k.contiguous().view(b,k,H*self.scale,W*self.scale)
        
        # score_k: b k H*s W*s
        # idx_k: b m k
        # diff_patch: b k*ce H*s W*s
        return score_k, idx_k, diff_patch


class GraphAggregation(nn.Module):
    r"""
    Graph Aggregation
    """
    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    def aggregation(self, yd, idx_k):
        r"""
        :param yd: database items, shape BxNxF
        :param idx_k: indexs of top k nearest neighbors

        :return: gathered features
        """
        # yd  -> b n f
        # I  -> b m o
        m = idx_k.shape[1]
        b, n, f = yd.shape
        k = self.k

        # yd -> b m f n
        # idx-> b m f k
        yd = yd.permute(0,2,1).contiguous()
        yd_e = yd.view(b,1,f,n).expand(b,m,f,n)
        idx_k_e = idx_k.view(b,m,1,k).expand(b,m,f,k)
        z = torch.gather(yd_e, dim=3, index=idx_k_e)

        # b m1*m2 c*p1*p2 k
        return z

    def forward(self, y, yd, idx_k):
        r"""
        :param y: query lr features
        :param yd: pixelshuffle_down features of y
        :param idx_k: indexs of top k nearest neighbors

        :return: aggregated hr features 
        """
        # Convert everything to patches
        y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)
      
        # bacth, channel, patchsize1, patchsize2, h, w
        _,_,H,W = y.shape
        _,_,_,_,m1,m2 = y_patch.shape
        b,c,p1,p2,n1,n2 = yd_patch.shape

        m = m1*m2; n = n1*n2; f=c*p1*p2
        k = self.k

        y_patch  = y_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,f//self.scale**2)
        yd_patch = yd_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)

        # Get nearest neighbor volumes
        # z_patch -> b m1*m2 c*p1*p2 k
        z_patch = self.aggregation(yd_patch, idx_k)

        # Adaptive_instance_normalization
        if cfg.NETWORK.WITH_ADAIN_NROM:
            reduce_scale = self.scale**2
            y_patch_norm = y_patch.view(b,m,c//reduce_scale,p1*p2)
            z_patch_norm = z_patch.view(b,m,c//reduce_scale,reduce_scale*p1*p2,k)
            z_patch = net_utils.adaptive_instance_normalization(y_patch_norm,z_patch_norm).view(*z_patch.size())
        
        # z_patch -> b k*c p1 p2 m1 m2
        z_patch = z_patch.permute(0,3,2,1).contiguous()

        z_patch_sr = z_patch.view(b,k,c//self.scale**2,self.scale,self.scale,p1,p2,m1,m2).permute(0,1,2,5,3,6,4,7,8).contiguous()
        z_patch_sr = z_patch_sr.view(b,k*(c//self.scale**2),p1*self.scale,p2*self.scale,m1,m2)
        padding_sr = [p*self.scale for p in padding]
        # z_sr -> b k*c_y H*s W*s
        z_sr = ops.patch2im(z_patch_sr, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        z_sr = z_sr.contiguous().view(b,k*(c//self.scale**2),H*self.scale,W*self.scale)

        return z_sr


index_neighbours_cache = {}
def index_neighbours(xe_patch, ye_patch, window_size, scale):
    r"""
    This function generates the indexing tensors that define neighborhoods for each query patch in (father) features
    It selects a neighborhood of window_size x window_size patches around each patch in xe (son) features
    Index tensors get cached in order to speed up execution time
    """
    if cfg.NETWORK.WITH_WINDOW == False:
        return None
        # dev = xe_patch.get_device()
        # key = "{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,dev)
        # if not key in index_neighbours_cache:
        #     I = torch.tensor(range(n), device=dev, dtype=torch.int64).view(1,1,n)
        #     I = I.repeat(b, m, 1)
        #     index_neighbours_cache[key] = I

        # I = index_neighbours_cache[key]
        # return Variable(I, requires_grad=False)

    b,_,_,_,n1,n2 = xe_patch.shape
    s = window_size
    
    if s>=n1 and s>=n2: 
        cfg.NETWORK.WITH_WINDOW = False
        return None

    s = min(min(s, n1), n2)
    o = s**2
    b,_,_,_,m1,m2 = ye_patch.shape

    dev = xe_patch.get_device()
    key = "{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,dev)
    if not key in index_neighbours_cache:
        I = torch.empty(1, m1 * m2, o, device=dev, dtype=torch.int64)

        ih = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,s,1)
        iw = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,1,s)*n2

        i = torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1,1,1,1)
        j = torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)

        i_s = (torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1,1,1,1)//2.0).long()
        j_s = (torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)//2.0).long()

        ch = (i_s-s//scale).clamp(0,n1-s)
        cw = (j_s-s//scale).clamp(0,n2-s)

        cidx = ch*n2+cw
        mI = cidx + ih + iw
        mI = mI.view(m1*m2,-1)
        I[0,:,:] = mI

        index_neighbours_cache[key] = I

    I = index_neighbours_cache[key]
    I = I.repeat(b,1,1)

    return Variable(I, requires_grad=False)
