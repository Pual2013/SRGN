import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        if d_hid > 50:
            cycle = 10
        elif d_hid > 5:
            cycle = 100
        else:
            cycle = 10000
        cycle = 10 if d_hid > 50 else 100
        return position / np.power(cycle, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class PosEncoding1D(nn.Module):
    
    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D, self).__init__()
        print("use PosEncoding1D")
        self.sel_index = torch.tensor([0]).cuda()
        pos_enc = (get_sinusoid_encoding_table((128//pos_rfactor)+1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0 #torch.tensor([0]).cuda()
            self.max = 128//pos_rfactor #torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos # B X H X W
        pos_h = pos_h//self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3) # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long() # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            #pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long(), 
                            min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)
            #pos_h = torch.where(pos_h < self.min_tensor, self.min_tensor, pos_h)
            #pos_h = torch.where(pos_h > self.max_tensor, self.max_tensor, pos_h)

        pos_h = self.pos_layer(pos_h).transpose(1,3).squeeze(3)   # B X 1 X 48 X 80 > B X 80 X 48 X 1 

        return pos_h


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.gamma = nn.Parameter(torch.zeros(1))

        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        self.weight = Parameter(torch.Tensor(self.group_planes, self.group_planes))

        self.pos_h = torch.arange(0, 1024).unsqueeze(0).unsqueeze(2).expand(-1,-1,2048)//8  # 0 ~ 127
        self.pos_w = torch.arange(0, 2048).unsqueeze(0).unsqueeze(1).expand(-1,1024,-1)//16 # 0 ~ 127
        self.pos_h = self.pos_h[0].byte().numpy()
        self.pos_w = self.pos_w[0].byte().numpy()
        self.pos_h = Image.fromarray(self.pos_h, mode="L")
        self.pos_w = Image.fromarray(self.pos_w, mode="L")
        self.pos = PosEncoding1D(8, self.group_planes * 2)

        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        ori = x
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        
        pos_h = torch.from_numpy(np.array(self.pos_h, dtype=np.uint8))# // self.pos_rfactor
        pos_w = torch.from_numpy(np.array(self.pos_w, dtype=np.uint8))# // self.pos_rfactor
        relative = self.pos(x, (pos_h, pos_w))

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        all_embeddings = all_embeddings + relative.view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sv = torch.matmul(sv, self.weight)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        sve = torch.matmul(sve, self.weight)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        output = ori + self.gamma *  output

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))