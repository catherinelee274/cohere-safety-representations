"""
Step 9 — Sparse Autoencoder (SAE) training and safety-feature analysis.

Train a SAE on residual-stream activations at a single layer, then:
  1. Identify features that activate differentially on harmful vs harmless prompts
  2. Compare feature overlap/alignment across model variants and languages
  3. Compute feature-level silhouette scores

Uses the dictionary_learning (EleutherAI) style SAE.
Install: pip install dictionary-learning

Usage:
    # Train SAE on global model, English, layer 24
    python scripts/09_sae_analysis.py \\
        --model global --lang en --layer 24 --train

    # Analyse (requires trained SAE)
    python scripts/09_sae_analysis.py \\
        --model global --lang en --layer 24 --analyse

Output:
    results/sae/{model}/{lang}/layer_{n}/
        sae_checkpoint.pt       trained SAE weights
        feature_activations.npy (N_prompts, n_features) float32
        safety_features.json    features ranked by harmful/harmless diff
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.languages import LANG_CODES
from config.models import MODELS, SAE_LAYER

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── SAE implementation (TopK sparse autoencoder) ─────────────────────────────

class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with Top-K activation sparsity.

    d_model    : input dimensionality (hidden_size of the LLM)
    n_features : dictionary size (typically 4–16× d_model)
    k          : number of active features per token
    """

    def __init__(self, d_model: int, n_features: int, k: int = 32):
        super().__init__()
        self.d_model    = d_model
        self.n_features = n_features
        self.k          = k

        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)

        # Unit-norm decoder columns
        nn.init.orthogonal_(self.decoder.weight)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        w = self.decoder.weight.data  # (d_model, n_features)
        norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data = w / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations. Shape: (batch, n_features)."""
        pre_acts = self.encoder(x)                    # (batch, n_features)
        top_vals, top_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, top_idx, top_vals.clamp(min=0))
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts)

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        x_hat = self.decode(acts)
        return x_hat, acts

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, acts: torch.Tensor,
             l1_coef: float = 1e-3) -> dict:
        recon_loss = (x - x_hat).pow(2).mean()
        l1_loss    = acts.abs().mean()
        total      = recon_loss + l1_coef * l1_loss
        return {"total": total, "recon": recon_loss, "l1": l1_loss}


# ── training ──────────────────────────────────────────────────────────────────

def train_sae(
    acts: np.ndarray,        # (N, d_model)
    out_dir: Path,
    n_features: int | None = None,
    k: int = 32,
    lr: float = 1e-4,
    n_epochs: int = 20,
    batch_size: int = 256,
    l1_coef: float = 1e-3,
    device: str | None = None,
) -> TopKSAE:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d_model    = acts.shape[1]
    n_features = n_features or (d_model * 8)   # 8× expansion

    logger.info(f"Training SAE: d={d_model}, features={n_features}, "
                f"k={k}, epochs={n_epochs}, N={len(acts)}")

    X   = torch.tensor(acts, dtype=torch.float32, device=device)
    sae = TopKSAE(d_model, n_features, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            x_hat, feat_acts = sae(batch)
            losses = sae.loss(batch, x_hat, feat_acts, l1_coef=l1_coef)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            sae._normalize_decoder()
            epoch_loss += losses["total"].item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = epoch_loss / len(loader)
            logger.info(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sae_checkpoint.pt"
    torch.save(sae.state_dict(), ckpt_path)
    logger.info(f"SAE saved → {ckpt_path}")
    return sae


# ── analysis ──────────────────────────────────────────────────────────────────

def extract_feature_acts(
    sae: TopKSAE,
    acts: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Return (N, n_features) feature activations."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae.to(device).eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = torch.tensor(acts[i:i+batch_size], dtype=torch.float32,
                                 device=device)
            feats = sae.encode(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def find_safety_features(
    feat_acts: np.ndarray,   # (N, n_features)
    labels: np.ndarray,      # (N,) 0=harmless, 1=harmful
    top_k: int = 50,
) -> list[dict]:
    """
    Rank features by |mean_harmful - mean_harmless| / pooled_std.
    Returns top_k features sorted by effect size (Cohen's d).
    """
    harmful  = feat_acts[labels == 1]
    harmless = feat_acts[labels == 0]

    mean_diff = harmful.mean(0) - harmless.mean(0)
    pooled_std = np.sqrt(
        (harmful.std(0) ** 2 + harmless.std(0) ** 2) / 2 + 1e-8
    )
    cohen_d   = mean_diff / pooled_std
    sparsity  = (feat_acts > 0).mean(0)

    indices = np.argsort(np.abs(cohen_d))[::-1][:top_k]
    return [
        {
            "feature_idx":    int(i),
            "cohen_d":        round(float(cohen_d[i]), 4),
            "mean_harmful":   round(float(harmful[:, i].mean()), 4),
            "mean_harmless":  round(float(harmless[:, i].mean()), 4),
            "sparsity":       round(float(sparsity[i]), 4),
        }
        for i in indices
    ]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=list(MODELS.keys()), default="global")
    parser.add_argument("--lang",       choices=LANG_CODES,          default="en")
    parser.add_argument("--layer",      type=int,                    default=SAE_LAYER)
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--analyse",    action="store_true")
    parser.add_argument("--n-features", type=int, default=None,
                        help="SAE dictionary size (default: 8 × d_model)")
    parser.add_argument("--k",          type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--acts-dir",   default="data/processed/activations")
    parser.add_argument("--out-dir",    default="results/sae")
    args = parser.parse_args()

    acts_dir = ROOT / args.acts_dir
    out_dir  = ROOT / args.out_dir / args.model / args.lang / f"layer_{args.layer}"

    # Load activations
    acts_path = acts_dir / args.model / args.lang / f"layer_{args.layer}.npy"
    lbl_path  = acts_dir / args.model / args.lang / "labels.npy"

    if not acts_path.exists():
        print(f"ERROR: {acts_path} not found. Run scripts/05_extract_activations.py first.")
        sys.exit(1)

    acts   = np.load(acts_path)
    labels = np.load(lbl_path)
    logger.info(f"Loaded activations: {acts.shape}, labels: {labels.shape}")

    # ── Train SAE ────────────────────────────────────────────────────────────
    sae = None
    ckpt_path = out_dir / "sae_checkpoint.pt"

    if args.train or not ckpt_path.exists():
        sae = train_sae(
            acts, out_dir,
            n_features=args.n_features,
            k=args.k,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        logger.info(f"Loading existing SAE from {ckpt_path}")
        d_model    = acts.shape[1]
        n_features = args.n_features or (d_model * 8)
        sae = TopKSAE(d_model, n_features, k=args.k)
        sae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    # ── Analyse ──────────────────────────────────────────────────────────────
    if args.analyse or args.train:
        feat_acts = extract_feature_acts(sae, acts)

        # Save feature activations
        feat_path = out_dir / "feature_activations.npy"
        np.save(feat_path, feat_acts)
        logger.info(f"Feature activations saved: {feat_acts.shape} → {feat_path}")

        # Safety features
        safety_feats = find_safety_features(feat_acts, labels, top_k=50)
        sf_path      = out_dir / "safety_features.json"
        with open(sf_path, "w") as f:
            json.dump(safety_feats, f, indent=2)

        print(f"\nTop-10 safety-discriminative features ({args.model}/{args.lang}/l{args.layer}):")
        for feat in safety_feats[:10]:
            print(f"  feat {feat['feature_idx']:>5}  "
                  f"Cohen's d={feat['cohen_d']:+.3f}  "
                  f"sparsity={feat['sparsity']:.3f}")

        print(f"\nFull results → {out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
