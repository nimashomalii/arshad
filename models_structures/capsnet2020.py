import torch
import torch.nn as nn
import torch.nn.functional as F

def x_to_cnn(input) : 
    data =torch.zeros(input.shape[0] ,input.shape[1]  , 9 , 9)
    grids = [(1,3) , (2,0) , (4,2) , (3,1) , (4,0) , (6,0) , (8,3) , (8,5) , (6,8) , (4,8) , (3,7) , (2,6) , (2,8) , (1,5)]
    for g, (i, j) in enumerate(grids):
        data[:, :, i, j] = input[:, :, g]
    return data

class EmotionCaps(nn.Module):
    def __init__(self, num_emotions, out_dim, num_iterations=3):
        super().__init__()
        self.num_capsules = None
        self.in_dim = None
        self.num_emotions = num_emotions
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.W = None  # lazy init

    def squash(self, s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        scale = (norm**2) / (1 + norm**2)
        return scale * s / (norm + 1e-8)

    def forward(self, u):
        B, N, in_dim = u.size()

        if self.W is None:
            self.num_capsules = N
            self.in_dim = in_dim
            self.W = nn.Parameter(
                torch.randn(1, N, self.num_emotions, self.out_dim, in_dim, device=u.device)
            )

        u_exp = u.unsqueeze(2).unsqueeze(-1)
        u_hat = torch.matmul(self.W, u_exp).squeeze(-1)
        b = torch.zeros(B, self.num_capsules, self.num_emotions, device=u.device)

        for r in range(self.num_iterations):
            c = F.softmax(b, dim=2).unsqueeze(-1)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s)  # <--- اینجا باید squash باشه

            if r < self.num_iterations - 1:
                v_exp = v.unsqueeze(1).unsqueeze(-1)
                uv = torch.matmul(u_hat.unsqueeze(-2), v_exp).squeeze(-1).squeeze(-1)
                b = b + uv

        return v


class model(nn.Module):
    def __init__(self, num_filter, time_len, caps_len, num_emotions, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=time_len, out_channels=num_filter, kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=6, stride=1)
        # bottleneck 1x1
        self.conv3 = nn.Conv2d(in_channels=2 * num_filter, out_channels=num_filter, kernel_size=1)
        self.padd_layer = nn.ZeroPad2d((2,3,2,3))
        self.relu = nn.ReLU()
        self.caps_len = caps_len

        self.emotion_caps = EmotionCaps(num_emotions=num_emotions, out_dim=out_dim)

    def forward(self, x):
        # x: (B, 1, time_len , num_channel)
        x = x_to_cnn(x)
        x = self.conv1(x)
        x = self.relu(x)
        y = self.padd_layer(x)
        y = self.conv2(y)
        y = self.relu(y)
        x = torch.cat((x, y), dim=1)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1, self.caps_len)
        # emotion capsules
        v = self.emotion_caps(x)  # (B, M, out_dim)
        v_abs = torch.norm(v , dim=-1)
        return v_abs
