from typing import Dict, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Plotear():
    def Imagenes(imagenesGeneradas, filas, cols, titulo):
        fig, axes = plt.subplots(filas, cols)
        for i in range(filas*cols):
            row = i // cols
            col = i % cols
            img = imagenesGeneradas[i]/2+0.5 #-1,1
            img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
            img = img.permute(1,2,0).cpu().numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        plt.suptitle(titulo)
        plt.show()

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

blk = lambda ic, oc: nn.Sequential(nn.Conv2d(ic, oc, 7, padding=3), nn.BatchNorm2d(oc), nn.LeakyReLU(),)

class DummyEpsModel(nn.Module):
    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        return self.conv(x)

class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, betas: Tuple[float, float], n_T: int, criterion: nn.Module = nn.MSELoss(),) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x)
        x_t = (self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * eps)
        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z)
        return x_i