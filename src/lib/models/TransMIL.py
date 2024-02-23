import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, n_heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = n_heads,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
            return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, in_sz=1024, n_heads=8):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(in_sz, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512, n_heads=n_heads)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    @staticmethod
    def calc_attn_padding(l1, l2):
        m = l1 % l2
        if m > 0:
            return l2 - m
        else:
            return 0

    def extract_attn_mat(self, inp, l1, layer, add_length):
        landm = layer.attn.num_landmarks
        n2 = l1 + add_length
        padd = self.calc_attn_padding(n2, landm)
        full_pad = padd + add_length
        #print(f'landm: {landm}, inp:{inp.shape}, padd: {padd}, orig: {l1}, add_length: {add_length}, fullpad: {full_pad}')
        return inp[:,:,full_pad:,full_pad:]

    def forward(self, **kwargs):

        results_dict = {}
        h = kwargs['data'].float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h, return_attn=False) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        if 'return_attn' in kwargs and kwargs['return_attn']:
            h, attn2 = self.layer2(h, return_attn=True) #[B, N, 512]
            results_dict['attn'] = self.extract_attn_mat(attn2, H, self.layer2, add_length).detach()
        else:
            h = self.layer2(h, return_attn=False) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        results_dict['logits'] = self._fc2(h) #[B, n_classes]
        results_dict['Y_hat'] = torch.argmax(results_dict['logits'], dim=1)
        results_dict['Y_prob'] = F.softmax(results_dict['logits'], dim = 1)
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
