import torch
from torch import nn
from s4 import FFTConv

class model(nn.Module):
    def __init__(
        self,
        input_dim: int,    # تعداد ویژگی ورودی (مثلاً تعداد کانال EEG)
        d_model: int = 64, # تعداد ویژگی داخلی (بعد بردار حالت)
        seq_len: int = 128 ,# طول دنباله
        output_dim = 2
    ):
        super().__init__()
        # تبدیل ورودی به فضای ویژگی داخلی
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # لایه S4
        self.s4 = FFTConv(
            d_model=d_model,
            l_max=seq_len,
            mode="s4",          # کرنل S4 کامل
            transposed=False,   # ورودی/خروجی شکل (B, L, D)
            channels=1,         # تعداد SSM مستقل
            dropout=0.0,
            d_state=d_model,    # ابعاد حالت داخلی S4
            rank=1              # رتبه پایین S4
        )
        
        # Readout نهایی به یک اسکالر (logit)
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        """
        x: (B, L, input_dim) - دنباله ورودی
        خروجی: (B,) - logits برای هر نمونه
        """
        x = self.input_proj(x)     # (B, L, d_model)
        y, _ = self.s4(x)          # (B, L, d_model)
        y_last = y[:, -1, :]       # فقط تایم‌استپ آخر
        logits = self.readout(y_last).squeeze(-1)  # (B,)
        return logits
