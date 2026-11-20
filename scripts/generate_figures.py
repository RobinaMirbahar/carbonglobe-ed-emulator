"""Generate publication-grade 300-DPI architecture and benchmark figures for CarbonGlobe ED Emulator."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures"


def make_fig0_architecture(out_dir: Path) -> None:
    """Render 300-DPI 3-stage system architecture diagram for CarbonGlobe ED Emulator (v8)."""
    fig = plt.figure(figsize=(14.5, 7.6), dpi=300, facecolor="#ffffff")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1450.0)
    ax.set_ylim(0.0, 760.0)
    ax.axis("off")

    def add_box(x, y, w, h, fc, ec, lw=1.6, r=10, ls="-", zorder=2):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder,
        )
        ax.add_patch(p)
        return p

    def add_arrow(x1, y1, x2, y2, color="#475569", lw=1.8, rad=0.0):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=7,head_width=4.2",
            color=color, linewidth=lw,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
        ax.add_patch(arr)

    # Title banner
    ax.text(725, 732, "CarbonGlobe ED Emulator (v8) — Physics-Constrained Multi-Family Architecture",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0f172a")
    ax.text(725, 707, "IEEE BigData Cup 2026  •  6,171 Global Sites × 2 Scenarios (ssp126/ssp585) × 15 Stand-Age Cohorts = 185,130 (H, AGB) Pairs",
            ha="center", va="center", fontsize=9.8, color="#475569")

    # Stage 0: Physics-Hygiene Domain Correction
    add_box(110, 585, 1230, 96, "#eef6ff", "#1a73e8", lw=2.0, r=14)
    ax.text(725, 660, "STAGE 0 — PHYSICS-HYGIENE DOMAIN CORRECTION & FEATURE SANITIZATION",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#174ea6")
    ax.text(725, 636, "Inverse-Affine Initial-State Inversion (R² > 0.999)  |  Corrupted Channel Pruning  |  Within-Site Radiative Anomaly Normalization",
            ha="center", va="center", fontsize=9.2, color="#1e293b")
    pills = [
        (150, 265, "y0 Shift Inversion (+14% Flux, +6.4% LAI)", "#dbeafe", "#1e40af"),
        (430, 260, "Drop Corrupted CO2 Ch. 8–32 & Soil Ch. 7", "#fee2e2", "#991b1b"),
        (705, 290, "Shortwave → Within-Site Yrs 0–4 Anomalies", "#fef3c7", "#92400e"),
        (1010, 290, "Non-Negative Physical Bounds (H, AGB ≥ 0)", "#dcfce7", "#166534"),
    ]
    for px, pw, ptxt, pfc, ptc in pills:
        add_box(px, 597, pw, 24, pfc, ptc, lw=1.0, r=7, zorder=3)
        ax.text(px + pw / 2.0, 609, ptxt, ha="center", va="center", fontsize=7.9, fontweight="bold", color=ptc, zorder=4)

    # Arrows Stage 0 -> 3 Model Suites
    add_arrow(410, 585, 255, 488, "#1a73e8", 2.0)
    add_arrow(725, 585, 725, 488, "#137333", 2.0)
    add_arrow(1040, 585, 1195, 488, "#8430ce", 2.0)

    # Layer 1: 3 Model Suites
    suites = [
        (
            55, "#f0f7ff", "#1a73e8", "#174ea6",
            "SUITE A: v4 SPATIAL-BLOCK (6 FAMILIES)",
            [
                "• 469 checkpoints across 6 model families",
                "• 10° × 10° Spatial-Block 5-Fold CV",
                "• High spatial OOD stability under climate shift",
                "• Warming holdout MSE: 0.02444",
                "• Contributes 0.35 prior anchor to W_cal",
            ],
            "Spatial OOF: 0.02849  |  Public LB: 0.079",
        ),
        (
            515, "#eefaf2", "#137333", "#0d5c28",
            "SUITE B: v5 CLEAN-PHYSICS (6 FAMILIES)",
            [
                "• 481 checkpoints across 6 clean-physics families",
                "• 5-Fold Stratified Site CV (Stage-1 + F12 Stack)",
                "• High-precision local stand-age emulation",
                "• Stage-1 Blend OOF: 0.01075 → F12: 0.00956",
                "• Contributes 0.65 prior anchor to W_cal",
            ],
            "Stratified OOF: 0.00956  |  481 Checkpoints",
        ),
        (
            975, "#f5effe", "#8430ce", "#5b21b6",
            "SUITE C: v7 DeepEDRolloutGRU (COUPLED)",
            [
                "• ~824k-parameter recurrent 40-step emulator",
                "• Per-site RevIN + Brandstetter rollout noise",
                "• CohortMixerBlock cross-age self-attention",
                "• Attention applied at init, yr 20, and yr 40",
                "• Independent dynamical physics verifier",
            ],
            "OOF: 0.01371  |  warm90: 0.01115  |  LB: 0.116",
        ),
    ]
    for sx, sfc, sec, stc, stitle, sbullets, sfoot in suites:
        add_box(sx, 225, 420, 260, sfc, sec, lw=1.8, r=12)
        add_box(sx + 12, 446, 396, 28, sec, sec, lw=0.8, r=7, zorder=3)
        ax.text(sx + 210, 460, stitle, ha="center", va="center", fontsize=9.3, fontweight="bold", color="#ffffff", zorder=4)
        for bi, bline in enumerate(sbullets):
            ax.text(sx + 20, 417 - bi * 31, bline, ha="left", va="center", fontsize=8.5, color="#1e293b", zorder=4)
        add_box(sx + 12, 236, 396, 28, "#ffffff", sec, lw=1.1, r=7, zorder=3)
        ax.text(sx + 210, 250, sfoot, ha="center", va="center", fontsize=8.2, fontweight="bold", color=stc, zorder=4)

    # Arrows Suites -> CalibrationHead
    add_arrow(265, 225, 420, 165, "#1a73e8", 2.0)
    add_arrow(725, 225, 725, 165, "#137333", 2.0)
    add_arrow(1185, 225, 1030, 165, "#8430ce", 1.8)

    # Stage 2: Structured Ridge CalibrationHead (W_cal)
    add_box(95, 32, 1260, 130, "#0f172a", "#10b981", lw=2.0, r=14)
    ax.text(725, 138, "STAGE 2 — STRUCTURED RIDGE CALIBRATIONHEAD (W_cal) & DETERMINISTIC SINGLE-PASS INFERENCE",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#34d399")
    ax.text(725, 112, "Tensor Shape (2, 15, 2, 55)  •  Anchored on 0.65·v5 + 0.35·v4  •  Cross-Age Smoothing (0.12, 0.76, 0.12)  •  Dual 5-Fold Rand / Temp CV",
            ha="center", va="center", fontsize=9.0, color="#e2e8f0")

    bot_pills = [
        (120, 290, "Dual-CV OOF MSE: 0.00536", "#1e293b", "#38bdf8"),
        (430, 310, "Public LB MSE: 0.052 (v8_struct)", "#1e293b", "#34d399"),
        (760, 315, "Public LB MSE: 0.052 (v8_free_lam8)", "#1e293b", "#fbbf24"),
        (1095, 235, "98.2% vs Persistence", "#1e293b", "#f87171"),
    ]
    for bx, bw, btxt, bfc, btc in bot_pills:
        add_box(bx, 52, bw, 32, bfc, btc, lw=1.2, r=8, zorder=3)
        ax.text(bx + bw / 2.0, 68, btxt, ha="center", va="center", fontsize=8.2, fontweight="bold", color=btc, zorder=4)

    fig.savefig(out_dir / "fig0_system_architecture.png", dpi=300, facecolor="#ffffff")
    plt.close(fig)


def make_fig1_progression(out_dir: Path) -> None:
    """Render 300-DPI 2-panel OOF & Public LB MSE progression chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=300, facecolor="#ffffff")
    fig.suptitle("CarbonGlobe ED Emulator — Empirical Validation & Public Leaderboard Progression",
                 fontsize=13.5, fontweight="bold", color="#0f172a", y=0.97)

    # Panel A: OOF MSE (log scale for clarity across 0.296 -> 0.00536)
    models_oof = ["Persistence", "PCHIP40", "v4 (6-Fam\nSpatial)", "v7 Rollout\nGRU", "v5 F12\nStack", "v8 CalHead\n(Final)"]
    vals_oof = [0.29622, 0.18103, 0.02849, 0.01371, 0.00956, 0.00536]
    cols_oof = ["#94a3b8", "#64748b", "#1a73e8", "#8430ce", "#137333", "#059669"]

    bars1 = ax1.bar(models_oof, vals_oof, color=cols_oof, width=0.58, edgecolor="#0f172a", linewidth=0.7)
    ax1.set_yscale("log")
    ax1.set_ylim(0.003, 0.65)
    ax1.set_ylabel("Out-of-Fold MSE (log scale, lower is better)", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax1.set_title("A. Cross-Validation OOF MSE (98.2% Reduction)", fontsize=10.5, fontweight="bold", color="#0f172a", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", labelsize=8.5)
    for b, v in zip(bars1, vals_oof):
        ax1.text(b.get_x() + b.get_width() / 2.0, v * 1.15, f"{v:.5f}", ha="center", va="bottom",
                 fontsize=8.3, fontweight="bold", color="#0f172a")

    # Panel B: Public LB MSE
    models_lb = ["v7 RolloutGRU\n(Ref 58398639)", "v4 Spatial\n(Ref 58268845)", "v8 Uncalibrated\nBase Anchor", "v8 struct\n(Ref 58398629)", "v8 free_lam8\n(Ref 58398618)"]
    vals_lb = [0.116, 0.079, 0.067, 0.052, 0.052]
    cols_lb = ["#8430ce", "#1a73e8", "#f59e0b", "#059669", "#10b981"]

    bars2 = ax2.bar(models_lb, vals_lb, color=cols_lb, width=0.56, edgecolor="#0f172a", linewidth=0.7)
    ax2.set_ylim(0.0, 0.14)
    ax2.set_ylabel("Public Leaderboard MSE (lower is better)", fontsize=9.5, fontweight="bold", color="#1e293b")
    ax2.set_title("B. Verified Kaggle Public LB MSE Progression", fontsize=10.5, fontweight="bold", color="#0f172a", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=8.2)
    for b, v in zip(bars2, vals_lb):
        ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8.8, fontweight="bold", color="#0f172a")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_dir / "fig1_benchmark_progression.png", dpi=300, facecolor="#ffffff")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_fig0_architecture(OUT_DIR)
    make_fig1_progression(OUT_DIR)
    print(f"Generated 300-DPI figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
