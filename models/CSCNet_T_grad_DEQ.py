import torch
from torch.nn import functional
from collections import namedtuple
from models.utils import calc_pad_sizes, vec2list, list2vec
from ops.utils import soft_threshold,fastSoftThrs
import torch.nn as nn
from utility.solvers import anderson, broyden
from utility.jacobian import jac_loss_estimate, power_method
import torch.autograd as autograd
from .swin import SwinTransformerBlock
# from torchdeq import get_deq
# from torchdeq.norm import apply_norm, reset_norm
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .DeConv import DEConv, DEConv2
from .vmamba import VSSM_CSC



ListaParams = namedtuple('ListaParams', ['kernel_size_unique', 'num_filters_unique', 'stride_share','stride_unique','threshold', 'unfoldings','multi_lmbda','kernel_size_share', 'num_filters_share','noise_est','data_stride','phantom'])


class CSCNet_T_grad_DEQ(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2):
        super(CSCNet_T_grad_DEQ, self).__init__()
        torch.random.manual_seed(2021)
        self.bands=31
        # self.bands=46

        # self.params = params
        self.share_kernel_size = params.kernel_size_share
        self.share_stride = params.stride_share
        self.share_filters_count = params.num_filters_share
        self.unique_kernel_size = params.kernel_size_unique
        self.unique_stride = params.stride_unique
        self.unique_filters_count = params.num_filters_unique
        self.data_stride = params.data_stride
        self.phantom = params.phantom
        
        self.share_decoder = torch.nn.ConvTranspose2d(self.share_filters_count, self.bands, kernel_size=self.share_kernel_size,stride=self.share_stride, bias=False,padding=0)
        self.share_encoder = torch.nn.Conv2d(self.bands,  self.share_filters_count, kernel_size=self.share_kernel_size, stride=self.share_stride, bias=False,padding=0)
        self.share_filters = torch.nn.ConvTranspose2d( self.share_filters_count, self.bands, kernel_size=self.share_kernel_size,stride=self.share_stride, bias=False,padding=0)
        
        self.unique_decoder = torch.nn.ConvTranspose3d(self.unique_filters_count, 1, kernel_size=(3,self.unique_kernel_size,self.unique_kernel_size),stride=(1, self.unique_stride, self.unique_stride), bias=False, padding=0)
        self.unique_encoder = torch.nn.Conv3d(1, self.unique_filters_count, kernel_size=(3,self.unique_kernel_size,self.unique_kernel_size), stride=(1, self.unique_stride, self.unique_stride), bias=False,padding=0)
        self.unique_filters = torch.nn.ConvTranspose3d(self.unique_filters_count, 1, kernel_size=(3,self.unique_kernel_size,self.unique_kernel_size),stride=(1, self.unique_stride, self.unique_stride), bias=False, padding=0)

        nn.init.kaiming_normal_(self.share_decoder.weight, mode='fan_out')
        D =self.share_decoder.weight.data
        l=torch.FloatTensor([1000])
        D /= torch.sqrt(l)
        self.share_decoder.weight.data = D
        self.share_encoder.weight.data = torch.clone(D)
        self.share_filters.weight.data = torch.clone(D)

        nn.init.kaiming_normal_(self.unique_decoder.weight, mode='fan_out')
        D =self.unique_decoder.weight.data
        l=torch.FloatTensor([1000])
        D /= torch.sqrt(l)
        self.unique_decoder.weight.data = D
        self.unique_encoder.weight.data = torch.clone(D)
        self.unique_filters.weight.data = torch.clone(D)

        self.fastSoftThrs = fastSoftThrs
        self.soft_threshold_unique = DEBlockTrain(self.unique_filters_count)
        # self.soft_threshold_unique = nn.Identity()
        # self.soft_threshold_share  = Prox_share_swin(dim=self.share_filters_count,input_resolution=(12,12), depth=4, num_heads=16,window_size=4) #64
        # self.soft_threshold_share  = Prox_share_swin(dim=self.share_filters_count,input_resolution=(24,24), depth=4, num_heads=16,window_size=4) #128
        # self.soft_threshold_share  = Prox_share_swin(dim=self.share_filters_count,input_resolution=(32,32), depth=4, num_heads=16,window_size=4) #200
        # self.soft_threshold_share  = Prox_share_swin(dim=self.share_filters_count,input_resolution=(40,40), depth=4, num_heads=16,window_size=4) #256
        # self.soft_threshold_share  = Prox_share_swin(dim=self.share_filters_count,input_resolution=(48,48), depth=4, num_heads=16,window_size=4) #300
        self.soft_threshold_share  = VSSM_CSC(
            depths=[1,1,1,1], dims=[self.share_filters_count, self.share_filters_count, self.share_filters_count, self.share_filters_count], drop_path_rate=0.5, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v0", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln", 
            downsample_version="v1", patchembed_version="v1", 
            use_checkpoint=False, posembed=False, imgsize=224, 
        )
        # self.soft_threshold_share  = nn.Identity()
        self.share_lmbdas = nn.Parameter(torch.zeros(1, self.share_filters_count, 1, 1))
        nn.init.constant_(self.share_lmbdas, params.threshold)
        self.unique_lmbdas = nn.Parameter(torch.zeros(1, self.unique_filters_count, 1, 1, 1))
        nn.init.constant_(self.unique_lmbdas, params.threshold)

        self.beta=params.noise_est
        self.unfoldings = params.unfoldings

        self.f_solver = anderson
        self.b_solver = anderson
        self.stop_mode = 'rel'
        # self.f_thres = 30
        # self.b_thres = 40
        self.f_thres = 20
        self.b_thres = 20
        self.hook = None
        self.unrolling = True
    
    def _split_image(self, I, stride):
        if stride == 1:
            return I, torch.ones_like(I)
        left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(I, self.share_kernel_size, stride)
        I_batched_padded = torch.zeros(I.shape[0], stride ** 2, I.shape[1], top_pad + I.shape[2] + bot_pad,
                                       left_pad + I.shape[3] + right_pad).type_as(I).to(I.device)
        valids_batched = torch.zeros_like(I_batched_padded)
        for num, (row_shift, col_shift) in enumerate([(i, j) for i in range(stride) for j in range(stride)]):
            I_padded = functional.pad(I, pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
            valids = functional.pad(torch.ones_like(I), pad=(
            left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
            I_batched_padded[:, num, :, :, :] = I_padded
            valids_batched[:, num, :, :, :] = valids
        I_batched_padded = I_batched_padded.reshape(-1, *I_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return I_batched_padded, valids_batched

    def sparse_coef_init(self, I):
        # share initial
        conv_input = self.share_encoder(I)
        conv_input = self.fastSoftThrs(conv_input, self.share_lmbdas,0.5)
        # print(conv_input.shape)
        gamma_s = self.soft_threshold_share(conv_input)
        I_s = self.share_filters(gamma_s, output_size=I.size())
        
        # unique initial
        I_u = I - I_s
        conv_input = self.unique_encoder(I_u.unsqueeze(1))
        conv_input = self.fastSoftThrs(conv_input, self.unique_lmbdas,0.5)
        gamma_u = self.soft_threshold_unique(conv_input)

        return [gamma_s, gamma_u], [gamma_s.shape, gamma_u.shape]
    
    def FISTA(self, gamma, I):

        gamma_s, gamma_u = gamma

        I_u = self.unique_filters(gamma_u, output_size=I.unsqueeze(1).size()).squeeze(1)
        x_k = self.share_decoder(gamma_s, output_size=I.size())
        r_k = self.share_encoder((x_k - (I - I_u)))
        gamma_s = self.fastSoftThrs(gamma_s - r_k, self.share_lmbdas,0.5)
        gamma_s = self.soft_threshold_share(gamma_s)

        I_s = self.share_filters(gamma_s, output_size=I.size())
        x_k = self.unique_decoder(gamma_u, output_size=I.unsqueeze(1).size())
        r_k = self.unique_encoder(((x_k.squeeze(1) - (I - I_s))).unsqueeze(1))
        gamma_u = self.fastSoftThrs(gamma_u - r_k, self.unique_lmbdas,0.5)
        gamma_u = self.soft_threshold_unique(gamma_u)

        return [gamma_s, gamma_u]
    
    def forward(self, I, compute_jac_loss = False, spectral_radius_mode = False):
        
        I = I.squeeze(1)
        # print(I.shape)
        I_batched_padded, valids_batched = self._split_image(I, self.data_stride)

        jac_loss = torch.tensor(0.0).to(I)

        with torch.no_grad():
            gamma, shapes = self.sparse_coef_init(I_batched_padded)

        z1 = list2vec(gamma)
        func = lambda z: list2vec(self.FISTA(vec2list(z, shapes), I_batched_padded))
            
        if self.unfoldings:
            for i in range(self.unfoldings):
                z1 = func(z1)
        else:
            # deq
            if True:
                with torch.no_grad():
                    result = self.f_solver(func, z1, threshold=self.f_thres, stop_mode=self.stop_mode, name="forward")
                z1 = result['result']
                z1 = func(z1.requires_grad_())
                if self.training:
                    new_z1 = z1.clone().detach().requires_grad_()
                    new_f = func(new_z1.requires_grad_())
                    # if compute_jac_loss:
                    #     jac_loss += jac_loss_estimate(new_f, new_z1)
                    # def backward_hook(grad):
                    #     if self.hook is not None:
                    #             self.hook.remove()
                    #             torch.cuda.synchronize()
                    #     result = self.b_solver(lambda y: autograd.grad(new_f, new_z1, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
                    #                           threshold=self.b_thres, stop_mode=self.stop_mode, name="backward")
                    #     return result['result']
                    # self.hook = z1.register_hook(backward_hook)
            # phantom
            if self.phantom:
                lambda_ = 0.90
                for _ in range(5): 
                    z1 = (1 - lambda_) * z1 + lambda_ * func(z1)

        gamma_s, gamma_u = vec2list(z1, shapes)

        I_s = self.share_filters(gamma_s, output_size=I_batched_padded.size())
        I_u = self.unique_filters(gamma_u, output_size=I_batched_padded.unsqueeze(1).size()).squeeze(1)

        output_cropped = torch.masked_select(I_s+I_u,  valids_batched.bool()).reshape(I.shape[0], self.data_stride ** 2, *I.shape[1:])
        output_all = output_cropped.mean(dim=1, keepdim=False)

        return output_all, jac_loss
        # return output_all

class Prox_share_swin(nn.Module):
    def __init__(self,dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size

        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, input_resolution[0]*input_resolution[1], dim))
        # trunc_normal_(self.absolute_pos_embed, std=.02)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

    def forward(self, x):
        shape = x.shape
        pad_size = self.input_resolution[-1] - shape[-1]
        out = nn.functional.pad(x, (0,pad_size,0,pad_size), "constant", 0)
        out_shape = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            out = blk(out)
        out = out.transpose(1, 2).reshape(out_shape)
        out = out[:,:,:-pad_size,:-pad_size]
        # out = out * x
        return out

class Prox_unique(nn.Module):
    def __init__(self, in_channels, out_channels,act_ratio=1, act_fn=nn.ReLU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.act_fn = act_fn()
        self.local_reduce = nn.Conv3d(in_channels, reduce_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,groups=1)
        self.spatial_spectral_select = nn.Conv3d(reduce_channels, in_channels, kernel_size=3, stride=1, padding=1,groups=1)
        self.gate_fn = gate_fn()
        self.dim =in_channels
    def forward(self, x):
        ori_x = x
        attn = self.gate_fn(self.spatial_spectral_select(self.act_fn(self.local_reduce(x))))
        out = ori_x * attn
        return out

class DEBlockTrain(nn.Module):
    def __init__(self, dim):
        super(DEBlockTrain, self).__init__()
        self.conv1 = DEConv2(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.detail_attn = Prox_unique(dim, dim, act_ratio=0.5)
        # self.detail_attn = nn.Identity()

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.detail_attn(res)
        # res = self.detail_attn(x)
        return res