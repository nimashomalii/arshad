import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    def __init__(self, caps_len, out_dim):
        super().__init__()
        self.caps_len = caps_len
        self.out_dim = out_dim
        self.matrices = None   # بعداً initialize می‌کنیم

    def forward(self, x):
        # x شکل: (B, N, in_dim)
        B, N, in_dim = x.size()

        # اگه اولین بار باشه => initialize
        if self.matrices is None:
            self.matrices = nn.ModuleList([
                nn.Linear(in_dim, self.caps_len) for _ in range(N)
            ])
            self.matrices = self.matrices.to(x.device)

        u = []
        for i in range(N):
            u.append(self.matrices[i](x[:, i, :]))  # (B, caps_len)
        u = torch.stack(u, dim=1)  # (B, N, caps_len)
        return u




class EmotionCaps(nn.Module):
    def __init__(self, num_emotions, out_dim, num_iterations=3):
        super().__init__()
        self.num_capsules = None     # N (lazy)
        self.in_dim = None           # طول بردار اولیه (lazy)
        self.num_emotions = num_emotions   # M
        self.out_dim = out_dim             # طول بردار ثانویه
        self.num_iterations = num_iterations

        # وزن‌ها رو اینجا None می‌ذاریم
        self.W = None  

    def squash(self, s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        scale = (norm**2) / (1 + norm**2)
        return scale * s / (norm + 1e-8)

    def forward(self, u):
        # u: (B, N, in_dim)
        B, N, in_dim = u.size()

        # بار اول که forward اجرا بشه → W رو می‌سازیم
        if self.W is None:
            self.num_capsules = N
            self.in_dim = in_dim
            self.W = nn.Parameter(
                torch.randn(1, N, self.num_emotions, self.out_dim, in_dim, device=u.device)
            )

        # آماده‌سازی برای matmul
        u_exp = u.unsqueeze(2).unsqueeze(-1)  # (B, N, 1, in_dim, 1)

        # محاسبه‌ی u_hat
        u_hat = torch.matmul(self.W, u_exp).squeeze(-1)  # (B, N, M, out_dim)

        # b_ij مقدار اولیه صفر
        b = torch.zeros(B, self.num_capsules, self.num_emotions, device=u.device)

        for r in range(self.num_iterations):
            c = F.softmax(b, dim=2).unsqueeze(-1)  # (B, N, M, 1)
            s = (c * u_hat).sum(dim=1)             # (B, M, out_dim)
            v = self.squash(s)                     # (B, M, out_dim)

            if r < self.num_iterations - 1:
                uv = torch.matmul(u_hat, v.unsqueeze(-1)).squeeze(-1)  # (B, N, M)
                b = b + uv

        return v  # (B, M, out_dim)


class model(nn.Module):
    def __init__(self, num_filter, num_channel, time_len, caps_len, num_emotions, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=6, stride=1)
        # bottleneck 1x1
        self.conv3 = nn.Conv2d(in_channels=2 * num_filter, out_channels=num_filter, kernel_size=1)
        self.padd_layer = nn.ZeroPad2d((2,3,2,3))
        self.relu = nn.ReLU()
        self.caps_len = caps_len

        # PrimaryCaps: بعد از کانولوشن reshape میشه به (B, num_capsules, in_dim)
        self.primary_caps = PrimaryCaps(caps_len=caps_len, out_dim=16)
        self.emotion_caps = EmotionCaps(num_emotions=num_emotions, out_dim=out_dim)

    def forward(self, x):
        # x: (B, 1, time_len , num_channel)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        y = self.padd_layer(x)
        y = self.conv2(y)
        y = self.relu(y)
        x = torch.cat((x, y), dim=1)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1, self.caps_len)
        print(x.shape)
        u = self.primary_caps(x)  # (B, N, caps_len)

        # emotion capsules
        v = self.emotion_caps(u)  # (B, M, out_dim)
        v_abs = torch.norm(v , dim=-1)
        return v_abs
