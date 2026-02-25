"""
Animation: MỘT ĐIỂM dữ liệu malicious bị GAN kéo dần về decision boundary

Kỹ thuật: Interpolate giữa real malicious và GAN sample:
  X(t) = (1-t)*X_malicious_real + t*X_gan_generated,  t = 0→1
  → Thấy điểm dịch chuyển từ vùng malicious → gần boundary

Output: results/summary/gan_one_point_animation.gif
"""

import sys, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized
import joblib
import PIL.Image as Image


def load_dede(model_dir):
    with open(Path(model_dir) / 'training_config.json') as f:
        cfg = json.load(f)
    m = build_dede_model(
        input_dim=cfg['input_dim'], latent_dim=cfg.get('latent_dim', 64),
        encoder_hidden_dims=[256, 128], decoder_hidden_dims=[128, 256],
        mask_ratio=cfg.get('mask_ratio', 0.5), dropout=0.2,
    )
    _ = m(tf.zeros((1, cfg['input_dim'])), training=False)
    m.load_weights(str(Path(model_dir) / 'best_model.weights.h5'))
    return m, cfg['input_dim']


def load_stacking(cache_dir, input_dim):
    d = Path(cache_dir)
    ens = create_stacking_ensemble_gan_optimized(input_dim=input_dim)
    ens.meta_model = joblib.load(d / 'meta_model.pkl')
    for name in list(ens.base_models.keys()):
        p1 = d / f'{name}_model.pkl'
        p2 = d / f'{name}_model.keras'
        if p1.exists():   ens.base_models[name] = joblib.load(p1)
        elif p2.exists():
            from tensorflow import keras
            ens.base_models[name] = keras.models.load_model(p2)
    ens.is_fitted = True
    return ens


def get_proba(ens, X):
    try:
        return ens.meta_model.predict_proba(ens._get_meta_features(X))[:, 1]
    except Exception:
        return ens.predict(X).astype(float)


def main():
    N_BEN  = 500
    N_MAL  = 500
    N_GAN  = 500
    STEPS  = 50
    FPS    = 8
    GRID_N = 80
    RNG    = np.random.RandomState(42)
    N_TRAILS = 1     # CHI 1 DIEM DUY NHAT

    raw_dir = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    out_dir = BASE_DIR / 'results/summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading models...')
    dede, input_dim = load_dede(BASE_DIR / 'experiments/dede_adapted/models_raw')
    stacking = load_stacking(
        BASE_DIR / 'results/raw/exp7_combined_matrix/ganopt_clean', input_dim
    )

    print('Loading data...')
    X_test  = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_test  = np.load(raw_dir / 'exp1_baseline/y_test.npy')
    X_gan   = np.load(raw_dir / 'exp3_gan_attack/X_test.npy')
    y_gan   = np.load(raw_dir / 'exp3_gan_attack/y_test.npy')

    X_ben = X_test[RNG.choice(np.where(y_test==0)[0], N_BEN, replace=False)]
    X_mal = X_test[RNG.choice(np.where(y_test==1)[0], N_MAL, replace=False)]
    X_gan_s = X_gan[RNG.choice(np.where(y_gan==1)[0],
                                min(N_GAN,(y_gan==1).sum()), replace=False)]

    # ── Chọn N_TRAILS điểm malicious có P(mal) cao nhất ──────────────────────
    prob_mal_all = get_proba(stacking, X_mal)
    # Chọn những điểm P cao (rõ ràng malicious) để thấy chuyển động dài nhất
    top_idx = np.argsort(prob_mal_all)[::-1][:N_TRAILS]
    X_starts = X_mal[top_idx]  # starting points (real malicious)
    print(f'Start P(mal) range: {get_proba(stacking, X_starts).min():.3f}'
          f'–{get_proba(stacking, X_starts).max():.3f}')

    # GAN endpoint: chọn GAN sample gần boundary nhất cho mỗi start
    # (P gần 0.5 nhất trong tập GAN)
    prob_gan_all = get_proba(stacking, X_gan_s)
    near_boundary_idx = np.argsort(np.abs(prob_gan_all - 0.52))[:N_TRAILS*3]
    X_near_boundary = X_gan_s[near_boundary_idx[:N_TRAILS]]
    X_ends = X_near_boundary
    print(f'End   P(mal) target: {get_proba(stacking, X_ends).mean():.3f} '
          f'(aim: close to 0.5)')

    # ── Build interpolation trajectory ───────────────────────────────────────
    ts_raw = np.linspace(0, 1, STEPS)
    trajectories_orig = np.array([
        (1 - t) * X_starts + t * X_ends for t in ts_raw
    ])  # shape: (STEPS, N_TRAILS, input_dim)

    # Compute P(malicious) for each step
    traj_prob_raw = np.array([
        get_proba(stacking, trajectories_orig[s]) for s in range(STEPS)
    ])  # shape: (STEPS, N_TRAILS)

    # ── Tìm t* nơi P ≈ 0.5 cho mỗi trail ────────────────────────────────────
    # Cắt bớt trajectory mỗi trail tại điểm dưới 0.5 đầu tiên
    stop_steps = []
    for i in range(N_TRAILS):
        cross = np.where(traj_prob_raw[:, i] < 0.5)[0]
        # Dừng tại step đầu tiên có P < 0.5 (hoặc cuối nếu không có)
        stop_steps.append(int(cross[0]) if len(cross) > 0 else STEPS - 1)
    max_stop = max(stop_steps)  # Dùng max để animation nhất quán
    ts = ts_raw[:max_stop+1]
    STEPS_ACTUAL = len(ts)

    trajectories_orig = trajectories_orig[:STEPS_ACTUAL]
    traj_prob = traj_prob_raw[:STEPS_ACTUAL]

    print(f'── Trajectory:')
    print(f'  P(mal) start : {traj_prob[0].mean():.3f}')
    print(f'  P(mal) end   : {traj_prob[-1].mean():.3f} (stop at ~boundary)')
    print(f'  Steps used   : {STEPS_ACTUAL}/{STEPS}')

    # ── PCA ──────────────────────────────────────────────────────────────────
    print('Fitting PCA...')
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([X_ben, X_mal]))
    var = pca.explained_variance_ratio_ * 100
    ben_2d = pca.transform(X_ben)
    mal_2d = pca.transform(X_mal)

    # ── Decision boundary (LogReg in 2D PCA) ─────────────────────────────────
    print('Computing decision boundary...')
    X_ref = np.vstack([X_ben, X_mal])
    y_stk = (get_proba(stacking, X_ref) >= 0.5).astype(int)
    XY_ref = np.vstack([ben_2d, mal_2d])
    clf2d = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf2d.fit(XY_ref, y_stk)

    x_min = XY_ref[:,0].min() - 2
    x_max = XY_ref[:,0].max() + 2
    y_min = XY_ref[:,1].min() - 2
    y_max = XY_ref[:,1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, GRID_N),
                         np.linspace(y_min, y_max, GRID_N))
    grid_proba = clf2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    print(f'  Boundary visible: {grid_proba.min():.3f}→{grid_proba.max():.3f} '
          f'(need to span 0.5)')

    # ── Build interpolation trajectory ───────────────────────────────────────
    # Project to 2D PCA
    traj_2d = np.array([pca.transform(trajectories_orig[s]) for s in range(STEPS_ACTUAL)])
    # DeDe error along trajectory
    thr = np.percentile(dede.get_reconstruction_error(X_test[:2000]), 99)
    traj_dede = np.array([dede.get_reconstruction_error(trajectories_orig[s]) for s in range(STEPS_ACTUAL)])

    print(f'DeDe: {traj_dede[0].mean():.4f} → {traj_dede[-1].mean():.4f}')

    # ── Colors per trail ──────────────────────────────────────────────────────
    trail_colors = plt.cm.tab10(np.linspace(0, 0.8, N_TRAILS))

    # ── Render frames ─────────────────────────────────────────────────────────
    print(f'\nRendering {STEPS_ACTUAL} frames...')
    BG = '#0d1117'
    frames = []

    for s in range(STEPS_ACTUAL):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        fig.patch.set_facecolor(BG)
        ax1, ax2 = axes
        for ax in axes:
            ax.set_facecolor('#161b22')
            ax.spines[['bottom','left','top','right']].set_color('#30363d')
            ax.tick_params(colors='white')

        t_pct = ts[s] * 100 # Scale to 0-100%

        # ── Left: PCA + boundary + moving points ──────────────────────────────
        # Background (xanh=benign, đỏ=malicious zone)
        ax1.contourf(xx, yy, grid_proba,
                     levels=[0.0, 0.5, 1.0],
                     colors=['#4CAF50', '#F44336'], alpha=0.13, zorder=0)
        # Vùng xung quanh boundary
        ax1.contourf(xx, yy, grid_proba,
                     levels=[0.4, 0.6],
                     colors=['yellow'], alpha=0.14, zorder=1)
        # Đường biên trắng rõ
        ax1.contour(xx, yy, grid_proba,
                    levels=[0.5], colors=['white'],
                    linewidths=[3], linestyles=['-'], zorder=2)
        # Label đường biên
        ax1.text((x_min+x_max)*0.68, y_max*0.92,
                 'Decision boundary\n       P = 0.5',
                 color='white', fontsize=9.5, fontweight='bold',
                 bbox=dict(boxstyle='round', fc='#111', alpha=0.75))

        # Background scatter (mờ)
        ax1.scatter(ben_2d[:,0], ben_2d[:,1],
                    c='#4CAF50', alpha=0.15, s=14, zorder=3, label='Benign')
        ax1.scatter(mal_2d[:,0], mal_2d[:,1],
                    c='#F44336', alpha=0.15, s=14, zorder=3, label='Malicious')

        # Ve toan bo duong di da qua (trail)
        if s > 0:
            ax1.plot(traj_2d[:s+1, 0, 0], traj_2d[:s+1, 0, 1],
                     '-', color='#FF9800', alpha=0.6, linewidth=2.5,
                     zorder=4, label='Trajectory')
        # Diem hien tai (lon, ro)
        ax1.scatter(traj_2d[s, 0, 0], traj_2d[s, 0, 1],
                    color='#FF9800', s=250, zorder=7,
                    edgecolors='white', linewidths=2.5,
                    marker='o')
        # Arrow chi huong
        if s >= 3:
            ax1.annotate('',
                xy=(traj_2d[s,0,0], traj_2d[s,0,1]),
                xytext=(traj_2d[s-3,0,0], traj_2d[s-3,0,1]),
                arrowprops=dict(arrowstyle='->', color='#FF9800',
                               lw=2.5, alpha=0.95), zorder=8)

        # Diem start va end
        ax1.scatter(traj_2d[0, 0, 0], traj_2d[0, 0, 1],
                    color='white', s=150, marker='x', zorder=8,
                    linewidths=3, label='Bat dau (real malicious)')
        ax1.scatter(traj_2d[-1, 0, 0], traj_2d[-1, 0, 1],
                    color='yellow', s=200, marker='*', zorder=8,
                    alpha=0.9, label='Dich (gan boundary)')

        ax1.set_xlabel(f'PC1 ({var[0]:.1f}% variance)', color='white', fontsize=10)
        ax1.set_ylabel(f'PC2 ({var[1]:.1f}% variance)', color='white', fontsize=10)
        ax1.set_title(f'Feature Space (PCA 2D)\n'
                      f't = {t_pct:.0f}%  →  P(malicious) = {traj_prob[s].mean():.3f}',
                      color='white', fontsize=12, fontweight='bold')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.legend(fontsize=8.5, facecolor='#21262d', labelcolor='white',
                   loc='lower right', framealpha=0.8)

        # Zone labels
        ax1.text(x_min + 0.5, y_max - 0.8, '🟢 BENIGN zone',
                 color='#4CAF50', fontsize=9, fontweight='bold')
        ax1.text(x_max - 2.5, y_min + 0.5, '🔴 MALICIOUS zone',
                 color='#F44336', fontsize=9, fontweight='bold')

        # DeDe status
        mean_err = traj_dede[s].mean()
        below = mean_err < thr
        ax1.text(0.02, 0.05,
                 f'DeDe error: {mean_err:.4f}\n{"✅ MISS" if below else "❌ DETECTED"} (thr={thr:.4f})',
                 transform=ax1.transAxes, fontsize=9,
                 color='#4CAF50' if below else '#F44336', fontweight='bold',
                 bbox=dict(boxstyle='round', fc='#111', alpha=0.8))

        # ── Right: P(malicious) trajectory line ───────────────────────────────
        ax2.axhline(0.5, color='white', lw=2.5, ls='--', alpha=0.9,
                    label='Decision boundary (0.5)')
        # Vung xung quanh boundary
        ax2.fill_between([0, 100], 0.45, 0.55,
                         color='yellow', alpha=0.12, label='Near boundary zone')
        ax2.axhline(get_proba(stacking, X_ben[:50]).mean(),
                    color='#4CAF50', lw=1.5, ls=':', alpha=0.7,
                    label=f'Benign avg ({get_proba(stacking, X_ben[:50]).mean():.2f})')
        ax2.axhline(get_proba(stacking, X_mal[:50]).mean(),
                    color='#F44336', lw=1.5, ls=':', alpha=0.7,
                    label=f'Malicious avg ({get_proba(stacking, X_mal[:50]).mean():.2f})')

        ts_pct = np.linspace(0, 100, STEPS_ACTUAL)
        # Plot trajectory cho từng trail
        for i in range(N_TRAILS):
            ax2.plot(ts_pct[:s+1], traj_prob[:s+1, i],
                     color=trail_colors[i], alpha=0.6, lw=1.5)
        # Mean trajectory to nổi bật
        mean_traj = traj_prob.mean(axis=1)
        ax2.plot(ts_pct[:s+1], mean_traj[:s+1],
                 'o-', color='white', lw=2.5, ms=4, zorder=5, label='Mean trajectory')
        # Điểm hiện tại
        ax2.scatter([t_pct], [mean_traj[s]],
                    color='yellow', s=150, zorder=8,
                    edgecolors='white', linewidths=2)

        ax2.text(t_pct, mean_traj[s] + 0.04,
                 f'{mean_traj[s]:.3f}',
                 color='yellow', fontsize=10, fontweight='bold', ha='center')

        ax2.set_xlabel('GAN Transformation Progress (%)', color='white', fontsize=11)
        ax2.set_ylabel('P(Malicious)', color='white', fontsize=11)
        ax2.set_title('P(malicious) giam dan den bien\nkhi GAN transform mau',
                      color='white', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.05, 1.08)
        ax2.legend(fontsize=8.5, facecolor='#21262d', labelcolor='white',
                   loc='upper right', framealpha=0.8)

        # Progress bar
        progress_x = np.linspace(0, t_pct, 100)
        ax2.fill_between(progress_x, -0.05, -0.02,
                         color='#FF9800', alpha=0.8)
        ax2.fill_between(np.linspace(t_pct, 100, 100), -0.05, -0.02,
                         color='#333', alpha=0.5)

        fig.suptitle(
            f'GAN Attack: Mẫu malicious tiến dần đến Decision Boundary   '
            f'[{"▓" * int(t_pct/5)}{"░" * (20-int(t_pct/5))}] {t_pct:.0f}%',
            color='white', fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        fname = out_dir / f'_frm_{s:04d}.png'
        fig.savefig(fname, dpi=110, bbox_inches='tight', facecolor=BG)
        frames.append(fname)
        plt.close(fig)

        if s % 10 == 0:
            print(f'  Frame {s+1}/{STEPS_ACTUAL}  t={t_pct:.0f}%  P={traj_prob[s].mean():.3f}')

    # ── Build GIF ─────────────────────────────────────────────────────────────
    print('\nBuilding GIF...')
    imgs = [Image.open(f) for f in frames]
    # Dừng 3s ở cuối
    pause = [imgs[-1]] * (FPS * 3)

    gif_path = out_dir / 'gan_one_point_animation.gif'
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:] + pause,
        optimize=False,
        duration=int(1000 / FPS),
        loop=0
    )
    for f in frames:
        f.unlink()

    print(f'\n✅ Saved: {gif_path}')
    print(f'   {STEPS_ACTUAL} frames  •  FPS={FPS}  •  ~{(STEPS_ACTUAL + FPS*3)/FPS:.0f}s')


if __name__ == '__main__':
    main()
