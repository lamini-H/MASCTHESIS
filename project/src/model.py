import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Encoder3D(nn.Module):
    """3D CNN encoder -> feature vector."""
    def __init__(self, in_channels=1, base=16, dropout=0.2, out_dim=256):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8

        self.stem = nn.Sequential(
            ConvBlock3D(in_channels, c1),
            ConvBlock3D(c1, c1),
        )
        self.down1 = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(c1, c2), ConvBlock3D(c2, c2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(c2, c3), ConvBlock3D(c3, c3))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), ConvBlock3D(c3, c4), ConvBlock3D(c4, c4))

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c4, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x  # (B, out_dim)

class GatedAttentionMIL(nn.Module):
    """
    Gated attention pooling from Ilse et al. (MIL Attention).
    Takes patch embeddings: (B, N, D) -> pooled: (B, D) and weights: (B, N)
    """
    def __init__(self, dim, attn_dim=128, dropout=0.0):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(dim, attn_dim), nn.Sigmoid())
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, H):
        # H: (B, N, D)
        V = self.v(H)          # (B, N, A)
        U = self.u(H)          # (B, N, A)
        A = self.w(self.drop(V * U)).squeeze(-1)  # (B, N)
        A = torch.softmax(A, dim=1)               # (B, N)
        M = torch.sum(H * A.unsqueeze(-1), dim=1) # (B, D)
        return M, A

class DualBranchAttentionNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, dropout=0.2,
                 emb_dim=256, attn_dim=128, num_classes=2):
        super().__init__()

        # Patch encoder used for every local patch
        self.patch_enc = Encoder3D(in_channels, base=base_channels, dropout=dropout, out_dim=emb_dim)
        self.local_attn = GatedAttentionMIL(dim=emb_dim, attn_dim=attn_dim, dropout=dropout)

        # Global encoder for whole-prostate volume
        self.global_enc = Encoder3D(in_channels, base=base_channels, dropout=dropout, out_dim=emb_dim)

        # Fusion head
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x_local, x_global, return_attn=False):
        """
        x_local:  (B, N, C, D, H, W)
        x_global: (B, C, D, H, W)
        """
        B, N, C, D, H, W = x_local.shape
        x_flat = x_local.view(B * N, C, D, H, W)
        h_flat = self.patch_enc(x_flat)          # (B*N, emb_dim)
        H_patches = h_flat.view(B, N, -1)        # (B, N, emb_dim)

        local_vec, attn_w = self.local_attn(H_patches)   # (B, emb_dim), (B, N)
        global_vec = self.global_enc(x_global)           # (B, emb_dim)

        fused = self.fuse(torch.cat([local_vec, global_vec], dim=1))
        logits = self.head(fused)

        if return_attn:
            return logits, attn_w
        return logits
