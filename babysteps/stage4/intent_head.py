"""Stage-4 M2a — factorized latent intent (IntentHead) + nested-CV G1 probe.

See `docs/superpowers/specs/2026-05-23-stage4-m2-slot-encoder-design.md`
for the design. M2a deliberately uses the existing 20-dim handcrafted
demo-evidence encoding as the IntentHead input `Z` (no pixel encoder
yet — that is M2b). Trains one slot per Stage-0 intent factor with
per-slot cross-entropy supervision; the cert (G1) is a frozen
LogisticRegression probe on held-out folds of the trained `G`.

Sim-free: CPU-only torch. The module does NOT touch records directly —
it operates on `(Z: np.ndarray, y: np.ndarray)` pairs supplied by the
caller. The Stage-4 firewall (see `babysteps/stage4/__init__.py`)
applies to `features.py` only; this file is the supervision side and
is allowed to read factor labels.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class IntentHead(nn.Module):
    """Map demo-evidence encoding Z -> factor-indexed slot intents G.

    Forward output `G ∈ R^{B, F, d_slot}`. The module itself has no
    knowledge of which slot index maps to which Stage-0 factor; that
    mapping is established by per-slot CE supervision at training time
    (slot `i` is supervised against factor `i`'s label via a per-slot
    linear decoder that is discarded after training).
    """

    def __init__(
        self,
        *,
        z_dim: int = 20,
        n_factors: int = 6,
        d_slot: int = 16,
        hidden: int = 64,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.n_factors = n_factors
        self.d_slot = d_slot
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_factors * d_slot),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(z.shape[0], self.n_factors, self.d_slot)


def _make_splitter(y: np.ndarray):
    """Same policy as `babysteps.stage4.probe._make_splitter`."""
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 5:
        return LeaveOneOut()
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def _train_one_slot(
    head: IntentHead,
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    factor_idx: int,
    n_classes: int,
    n_epochs: int,
    lr: float,
    seed: int,
) -> None:
    """Train `head` (in place) so slot[factor_idx] CE-predicts y_tr.

    The per-slot linear decoder is discarded; only the IntentHead weights
    persist into the cert probe.
    """
    torch.manual_seed(seed)
    decoder = nn.Linear(head.d_slot, n_classes)
    z = torch.tensor(Z_tr, dtype=torch.float32)
    t = torch.tensor(y_tr, dtype=torch.long)
    opt = torch.optim.Adam(
        list(head.parameters()) + list(decoder.parameters()), lr=lr,
    )
    head.train()
    for _ in range(n_epochs):
        logits = decoder(head(z)[:, factor_idx])
        loss = F.cross_entropy(logits, t)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_intent_head_joint(
    head: IntentHead,
    Z_tr: np.ndarray,
    labels_per_factor: dict[int, tuple[np.ndarray, int]],
    *,
    n_epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
) -> dict[int, nn.Linear]:
    """Train IntentHead with per-slot CE supervision across multiple factors.

    `labels_per_factor`: {factor_idx: (y_tr, n_classes)} for each
    non-trivial factor to supervise. The total loss is the sum of
    per-slot cross-entropy losses; gradients flow through the shared
    trunk so all supervised slots end up populated simultaneously.

    Returns the per-slot decoders (one `nn.Linear(d_slot, n_classes)`
    per supervised factor). The caller may either discard them (use a
    frozen LR probe like G1) or keep them (e.g. for inspection). The
    canonical M2a slot decoder is the centroid lookup in
    `babysteps.stage4.slot_decode`, not these per-slot classifiers.
    """
    if not labels_per_factor:
        return {}
    torch.manual_seed(seed)
    z = torch.tensor(Z_tr, dtype=torch.float32)
    decoders: dict[int, nn.Linear] = {}
    targets: dict[int, torch.Tensor] = {}
    for fi, (y, n_cls) in labels_per_factor.items():
        if n_cls < 2:
            # Trivially-constant factors carry no signal; skip
            continue
        decoders[fi] = nn.Linear(head.d_slot, n_cls)
        targets[fi] = torch.tensor(y, dtype=torch.long)
    if not decoders:
        return {}
    params = list(head.parameters())
    for dec in decoders.values():
        params += list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    head.train()
    for _ in range(n_epochs):
        G = head(z)
        loss = sum(
            F.cross_entropy(decoders[fi](G[:, fi]), targets[fi])
            for fi in decoders
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return decoders


def nested_cv_probe_one_factor(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    factor_idx: int,
    n_factors: int = 6,
    d_slot: int = 16,
    n_epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    standardize_input: bool = False,
) -> dict:
    """Outer-CV trains IntentHead per fold; inner frozen LR probes `G` on test.

    Per-fold protocol (the cert-honest version of M2a's G1 gate):

      1. fresh IntentHead seeded per fold,
      2. trained on TRAIN fold's Z + y via single-slot CE,
      3. produces G on TRAIN and TEST fold (`G_tr`, `G_te`),
      4. fresh LogisticRegression trained on G_tr → y_tr, scored on G_te → y_te.

    Returns the same keys as `babysteps.stage4.probe.train_probe` so the
    report aggregator (`babysteps.stage4.report.build_report`) ingests
    feature and G-probe outputs identically.

    ``standardize_input`` (default False — committed numbers unchanged): fit a
    StandardScaler on the TRAIN fold's Z and apply it to train+test before the
    IntentHead. The IntentHead trains Adam at a fixed lr; on encoders whose
    feature norms differ from the handcrafted/DINOv2 scale this lr underfits and
    the probe spuriously collapses (e.g. V-JEPA-2.1 features: 0.54±0.24 raw vs
    ~0.86 standardized, while DINOv2 is ~unchanged). Standardizing makes the
    probe fair across encoders. Leak-free: the scaler is fit on the train fold
    only. See reports/stage5/vjepa_object_motion/FINDINGS.md.
    """
    n_unique = int(np.unique(y).size)
    if n_unique <= 1:
        return {
            "n_episodes": int(Z.shape[0]),
            "n_unique_labels": n_unique,
            "probe_acc_mean": 1.0,
            "probe_acc_std": 0.0,
            "majority_class_acc": 1.0,
            "shuffled_features_acc": 1.0,
            "trivially_constant": True,
        }

    Z = Z.astype(np.float32, copy=False)
    splitter = _make_splitter(y)
    rng = np.random.default_rng(seed)
    fold_accs: list[float] = []
    fold_shuf: list[float] = []

    for fold_i, (tr, te) in enumerate(splitter.split(Z, y)):
        Z_tr, Z_te = Z[tr], Z[te]
        if standardize_input:
            scaler = StandardScaler().fit(Z_tr)  # leak-free: train fold only
            Z_tr = scaler.transform(Z_tr).astype(np.float32, copy=False)
            Z_te = scaler.transform(Z_te).astype(np.float32, copy=False)
        head = IntentHead(
            z_dim=Z.shape[1], n_factors=n_factors,
            d_slot=d_slot, seed=seed + fold_i,
        )
        _train_one_slot(
            head, Z_tr, y[tr],
            factor_idx=factor_idx, n_classes=n_unique,
            n_epochs=n_epochs, lr=lr, seed=seed + fold_i,
        )
        head.eval()
        with torch.no_grad():
            z_tr_t = torch.from_numpy(Z_tr)
            z_te_t = torch.from_numpy(Z_te)
            G_tr = head(z_tr_t).numpy().reshape(len(tr), -1)
            G_te = head(z_te_t).numpy().reshape(len(te), -1)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(G_tr, y[tr])
        fold_accs.append(float(clf.score(G_te, y[te])))

        # Shuffled-features baseline: permute G_tr rows before refit.
        idx = rng.permutation(len(tr))
        clf_shuf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf_shuf.fit(G_tr[idx], y[tr])
        fold_shuf.append(float(clf_shuf.score(G_te, y[te])))

    fold_accs_np = np.asarray(fold_accs)
    _, counts = np.unique(y, return_counts=True)
    return {
        "n_episodes": int(Z.shape[0]),
        "n_unique_labels": n_unique,
        "probe_acc_mean": float(fold_accs_np.mean()),
        "probe_acc_std": float(fold_accs_np.std()),
        "majority_class_acc": float(counts.max() / counts.sum()),
        "shuffled_features_acc": float(np.mean(fold_shuf)),
        "trivially_constant": False,
    }
