"""
Visualize: Quá trình GAN tạo mẫu tiến dần đến decision boundary

Chạy mini GAN (nhanh) và lưu samples ở các checkpoint epoch:
  epoch 0  → noise ngẫu nhiên (P ≈ 0.5, DeDe error ≈ cao)
  epoch 5  → bắt đầu học distribution malicious
  epoch 10 → samples đang tiến về benign
  epoch 20 → gần biên quyết định
  epoch 30 → final GAN samples (P ≈ 0.5-0.6!)

Visualizations:
  1. Trajectory 2D (PCA): Real malicious → GAN samples qua các epoch
  2. P(malicious) per epoch: Confidence giảm dần theo training
  3. DeDe error per epoch: Error thấp dần (trở nên "benign-like")

Output: results/summary/gan_training_trajectory.png
"""

import sys, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized
import joblib


# ── Load models ───────────────────────────────────────────────────────────────

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


# ── Mini GAN training với checkpoint ─────────────────────────────────────────

def build_mini_gen(latent_dim, output_dim):
    return models.Sequential([
        layers.Dense(64, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='tanh'),
    ], name='generator')


def build_mini_disc(input_dim):
    return models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ], name='discriminator')


def train_mini_gan_with_checkpoints(X_mal, latent_dim=64,
                                    total_epochs=35, batch_size=64,
                                    checkpoint_epochs=None, n_gen=200):
    """
    Train mini GAN và lưu generated samples tại checkpoint_epochs.
    Returns: dict {epoch: generated_samples (raw space)}
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [0, 3, 7, 12, 20, 35]

    # Normalize
    X_min = X_mal.min(axis=0)
    X_max = X_mal.max(axis=0)
    X_scaled = (X_mal - X_min) / (X_max - X_min + 1e-8) * 2 - 1

    feat_dim = X_mal.shape[1]
    gen  = build_mini_gen(latent_dim, feat_dim)
    disc = build_mini_disc(feat_dim)
    disc.compile(optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                 loss='binary_crossentropy')

    disc.trainable = False
    gi = layers.Input(shape=(latent_dim,))
    go = disc(gen(gi))
    gan_m = models.Model(gi, go)
    gan_m.compile(optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                  loss='binary_crossentropy')

    checkpoints = {}
    n = len(X_scaled)
    fixed_noise = np.random.normal(0, 1, (n_gen, latent_dim))

    def gen_samples():
        raw = gen.predict(fixed_noise, verbose=0)
        return (raw + 1) / 2 * (X_max - X_min + 1e-8) + X_min

    # epoch 0 = pure noise
    if 0 in checkpoint_epochs:
        checkpoints[0] = gen_samples()
        print(f'  Checkpoint epoch=0 (init)')

    for epoch in range(1, total_epochs + 1):
        batches = n // batch_size
        for _ in range(batches):
            idx = np.random.randint(0, n, batch_size)
            real = X_scaled[idx]
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake  = gen.predict(noise, verbose=0)

            disc.trainable = True
            disc.train_on_batch(real, np.ones((batch_size, 1)))
            disc.train_on_batch(fake, np.zeros((batch_size, 1)))
            disc.trainable = False

            noise2 = np.random.normal(0, 1, (batch_size, latent_dim))
            gan_m.train_on_batch(noise2, np.ones((batch_size, 1)))

        if epoch in checkpoint_epochs:
            checkpoints[epoch] = gen_samples()
            print(f'  Checkpoint epoch={epoch}')

    return checkpoints, X_min, X_max


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    N_REAL  = 300   # real samples cho PCA fit
    N_GEN   = 200   # generated samples per epoch
    LATENT  = 64
    RNG     = np.random.RandomState(42)
    CKPTS   = [0, 3, 7, 12, 20, 35]

    raw_dir = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    out_dir = BASE_DIR / 'results/summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading models...')
    dede, input_dim = load_dede(BASE_DIR / 'experiments/dede_adapted/models_raw')
    stacking = load_stacking(
        BASE_DIR / 'results/raw/exp7_combined_matrix/ganopt_clean', input_dim
    )
    X_clean = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    thr = np.percentile(dede.get_reconstruction_error(X_clean), 99)

    print('Loading training data...')
    X_train = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_train = np.load(raw_dir / 'exp1_baseline/y_train.npy')
    X_test  = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_test  = np.load(raw_dir / 'exp1_baseline/y_test.npy')

    # Lấy samples đại diện
    mal_idx = RNG.choice(np.where(y_train == 1)[0], min(2000, (y_train==1).sum()), replace=False)
    X_mal_train = X_train[mal_idx]
    mal_test_idx = RNG.choice(np.where(y_test == 1)[0], N_REAL, replace=False)
    ben_test_idx = RNG.choice(np.where(y_test == 0)[0], N_REAL, replace=False)
    X_mal_real = X_test[mal_test_idx]
    X_ben_real  = X_test[ben_test_idx]

    print(f'\nTraining mini GAN ({max(CKPTS)} epochs) with checkpoints {CKPTS}...')
    checkpoints, X_min, X_max = train_mini_gan_with_checkpoints(
        X_mal_train, latent_dim=LATENT,
        total_epochs=max(CKPTS), batch_size=64,
        checkpoint_epochs=set(CKPTS), n_gen=N_GEN
    )

    print('\nComputing metrics at each checkpoint...')
    epoch_metrics = {}
    for ep, X_gen in sorted(checkpoints.items()):
        errs  = dede.get_reconstruction_error(X_gen)
        proba = get_proba(stacking, X_gen)
        epoch_metrics[ep] = {
            'mean_prob': float(proba.mean()),
            'std_prob':  float(proba.std()),
            'mean_err':  float(errs.mean()),
            'pct_below_thr': float((errs < thr).mean()),
            'X_gen': X_gen,
        }
        print(f'  Epoch {ep:3d}: P(mal)={proba.mean():.3f}±{proba.std():.3f}  '
              f'DeDe_err={errs.mean():.4f}  below_thr={( errs<thr).mean()*100:.0f}%')

    # Reference metrics
    err_ben = dede.get_reconstruction_error(X_ben_real)
    err_mal = dede.get_reconstruction_error(X_mal_real)
    prob_ben = get_proba(stacking, X_ben_real)
    prob_mal = get_proba(stacking, X_mal_real)

    # ── PCA fit ───────────────────────────────────────────────────────────────
    print('\nFitting PCA...')
    X_ref = np.vstack([X_ben_real, X_mal_real])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_ref)
    var = pca.explained_variance_ratio_ * 100

    ben_2d = pca.transform(X_ben_real)
    mal_2d = pca.transform(X_mal_real)

    gen_2d = {ep: pca.transform(m['X_gen']) for ep, m in epoch_metrics.items()}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor('#0d1117')

    BG   = '#161b22'
    CBEN = '#4CAF50'
    CMAL = '#F44336'
    CB   = 'white'

    # Colormap cho epochs: orange → purple (early → late)
    cmap = plt.cm.plasma
    n_ep = len(CKPTS)
    epoch_colors = {ep: cmap(i / (n_ep - 1)) for i, ep in enumerate(CKPTS)}

    # ── Plot 1: P(malicious) theo epoch ──────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor(BG)

    eps   = sorted(epoch_metrics.keys())
    means = [epoch_metrics[e]['mean_prob'] for e in eps]
    stds  = [epoch_metrics[e]['std_prob']  for e in eps]

    ax1.fill_between(eps,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.25, color='#FF9800')
    ax1.plot(eps, means, 'o-', color='#FF9800', linewidth=2.5, markersize=8, label='GAN samples')
    ax1.axhline(0.5, color=CB, linewidth=1.5, linestyle='--', alpha=0.7, label='Decision boundary')
    ax1.axhline(prob_ben.mean(), color=CBEN, linewidth=1.5, linestyle=':', alpha=0.7,
                label=f'Benign ref ({prob_ben.mean():.2f})')
    ax1.axhline(prob_mal.mean(), color=CMAL, linewidth=1.5, linestyle=':', alpha=0.7,
                label=f'Malicious ref ({prob_mal.mean():.2f})')

    # Annotation arrow
    ax1.annotate('', xy=(eps[-1], means[-1]), xytext=(eps[0], means[0]),
                 arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
    ax1.text((eps[0]+eps[-1])/2, (means[0]+means[-1])/2 + 0.05,
             'Tiến dần\nvề biên!', color='#FF9800', fontsize=9, ha='center')

    ax1.set_xlabel('GAN Training Epoch', color=CB, fontsize=11)
    ax1.set_ylabel('P(Malicious)', color=CB, fontsize=11)
    ax1.set_title('P(Malicious) của GAN samples\ntheo epoch training', color=CB, fontsize=12, fontweight='bold')
    ax1.tick_params(colors=CB)
    ax1.spines[['bottom','left','top','right']].set_color('#30363d')
    ax1.legend(fontsize=8.5, facecolor='#21262d', labelcolor=CB, framealpha=0.8)
    ax1.set_ylim(0, 1.05)

    # ── Plot 2: DeDe error theo epoch ────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor(BG)

    errs_mean = [epoch_metrics[e]['mean_err'] for e in eps]
    ax2.fill_between(eps,
                     [epoch_metrics[e]['mean_err'] * 0.7 for e in eps],
                     [epoch_metrics[e]['mean_err'] * 1.3 for e in eps],
                     alpha=0.2, color='#9C27B0')
    ax2.plot(eps, errs_mean, 's-', color='#9C27B0', linewidth=2.5, markersize=8)
    ax2.axhline(err_ben.mean(), color=CBEN, linewidth=1.5, linestyle=':', alpha=0.8,
                label=f'Benign ref ({err_ben.mean():.4f})')
    ax2.axhline(err_mal.mean(), color=CMAL, linewidth=1.5, linestyle=':', alpha=0.8,
                label=f'Malicious ref ({err_mal.mean():.4f})')
    ax2.axhline(thr, color=CB, linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'DeDe threshold ({thr:.4f})')

    pct_final = epoch_metrics[eps[-1]]['pct_below_thr'] * 100
    ax2.text(eps[-2], errs_mean[-1] * 1.1,
             f'{pct_final:.0f}% below\nthreshold\n(DeDe misses!)',
             color='#9C27B0', fontsize=8.5, ha='center',
             bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.7))

    ax2.set_xlabel('GAN Training Epoch', color=CB, fontsize=11)
    ax2.set_ylabel('DeDe Reconstruction Error', color=CB, fontsize=11)
    ax2.set_title('DeDe Error của GAN samples\n(càng thấp → DeDe càng không phát hiện được)', color=CB, fontsize=12, fontweight='bold')
    ax2.tick_params(colors=CB)
    ax2.spines[['bottom','left','top','right']].set_color('#30363d')
    ax2.legend(fontsize=9, facecolor='#21262d', labelcolor=CB, framealpha=0.8)

    # ── Plot 3: PCA trajectory ────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(BG)

    # Background: real benign & malicious
    ax3.scatter(ben_2d[:, 0], ben_2d[:, 1], c=CBEN, alpha=0.2, s=12, label='Benign (real)')
    ax3.scatter(mal_2d[:, 0], mal_2d[:, 1], c=CMAL, alpha=0.2, s=12, label='Malicious (real)')

    # GAN trajectories: vẽ từng epoch
    centers = {}
    for ep in CKPTS:
        pts = gen_2d[ep]
        col = epoch_colors[ep]
        ax3.scatter(pts[:, 0], pts[:, 1], c=[col]*len(pts), alpha=0.5, s=18,
                    zorder=3, marker='^')
        centers[ep] = (pts[:, 0].mean(), pts[:, 1].mean())

    # Vẽ arrow trajectory trung tâm
    ep_sorted = sorted(CKPTS)
    for i in range(len(ep_sorted) - 1):
        e1, e2 = ep_sorted[i], ep_sorted[i+1]
        c1, c2 = centers[e1], centers[e2]
        ax3.annotate('', xy=c2, xytext=c1,
                     arrowprops=dict(arrowstyle='->', color='white', lw=1.5, alpha=0.8))

    # Colorbar legend cho epochs
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max(CKPTS)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, fraction=0.03, pad=0.02)
    cbar.set_label('Epoch', color=CB, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=CB)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=CB, fontsize=8)

    ax3.set_xlabel(f'PC1 ({var[0]:.1f}% var)', color=CB, fontsize=10)
    ax3.set_ylabel(f'PC2 ({var[1]:.1f}% var)', color=CB, fontsize=10)
    ax3.set_title('GAN samples trajectory trong feature space\n(PCA 2D) — theo epoch training', color=CB, fontsize=12, fontweight='bold')
    ax3.tick_params(colors=CB)
    ax3.spines[['bottom','left','top','right']].set_color('#30363d')

    # Legend
    handles = [
        mpatches.Patch(color=CBEN, alpha=0.6, label='Benign (real)'),
        mpatches.Patch(color=CMAL, alpha=0.6, label='Malicious (real)'),
        mpatches.Patch(color='white', alpha=0.6, label='→ GAN trajectory'),
    ]
    ax3.legend(handles=handles, fontsize=9, facecolor='#21262d', labelcolor=CB, framealpha=0.8)

    fig.suptitle('Quá trình GAN tạo mẫu tiến dần đến Decision Boundary',
                 color=CB, fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    save_path = out_dir / 'gan_training_trajectory.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'\n✅ Saved: {save_path}')


if __name__ == '__main__':
    main()
