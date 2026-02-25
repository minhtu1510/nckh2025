"""
Animation: GAN samples tiến dần đến decision boundary qua từng epoch

Output: results/summary/gan_animation.gif
"""

import sys, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.dede_adapted.dede_model import build_dede_model
from models.ensemble.stacking import create_stacking_ensemble_gan_optimized
import joblib


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


def build_mini_gen(latent_dim, output_dim):
    return models.Sequential([
        layers.Dense(64, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(output_dim, activation='tanh'),
    ])


def build_mini_disc(input_dim):
    return models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

from sklearn.linear_model import LogisticRegression


def main():
    N_BEN   = 400
    N_MAL   = 400
    N_GEN   = 300
    LATENT  = 64
    GRID_N  = 80        # Grid resolution cho decision boundary
    CKPTS   = list(range(0, 36, 1))   # Mỗi epoch 1 frame
    FPS     = 6
    RNG     = np.random.RandomState(42)

    raw_dir = BASE_DIR / 'datasets/splits/3.0_raw_from_latent'
    out_dir = BASE_DIR / 'results/summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading models...')
    dede, input_dim = load_dede(BASE_DIR / 'experiments/dede_adapted/models_raw')
    stacking = load_stacking(
        BASE_DIR / 'results/raw/exp7_combined_matrix/ganopt_clean', input_dim
    )

    print('Loading data...')
    X_train = np.load(raw_dir / 'exp1_baseline/X_train.npy')
    y_train = np.load(raw_dir / 'exp1_baseline/y_train.npy')
    X_test  = np.load(raw_dir / 'exp1_baseline/X_test.npy')
    y_test  = np.load(raw_dir / 'exp1_baseline/y_test.npy')

    ben_idx = RNG.choice(np.where(y_test == 0)[0], N_BEN, replace=False)
    mal_idx = RNG.choice(np.where(y_test == 1)[0], N_MAL, replace=False)
    X_ben   = X_test[ben_idx]
    X_mal   = X_test[mal_idx]

    mal_train_idx = RNG.choice(np.where(y_train==1)[0], min(3000,(y_train==1).sum()), replace=False)
    X_mal_train = X_train[mal_train_idx]

    # ── PCA fit ───────────────────────────────────────────────────────────────
    print('Fitting PCA on benign + malicious...')
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([X_ben, X_mal]))
    var = pca.explained_variance_ratio_ * 100

    ben_2d = pca.transform(X_ben)
    mal_2d = pca.transform(X_mal)

    # ── Decision boundary: fit LogReg trong 2D PCA space ─────────────────────
    # Lấy stacking prediction làm label → fit LogReg trong PCA 2D
    # → boundary rõ ràng, không cần inverse_transform
    print('Computing decision boundary (LogReg in PCA 2D)...')
    X_all_ref = np.vstack([X_ben, X_mal])
    y_stk     = (get_proba(stacking, X_all_ref) >= 0.5).astype(int)
    XY_all    = np.vstack([ben_2d, mal_2d])

    from sklearn.linear_model import LogisticRegression
    clf2d = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf2d.fit(XY_all, y_stk)

    x_min = XY_all[:,0].min() - 1.5
    x_max = XY_all[:,0].max() + 1.5
    y_min = XY_all[:,1].min() - 1.5
    y_max = XY_all[:,1].max() + 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, GRID_N),
        np.linspace(y_min, y_max, GRID_N)
    )
    grid_proba = clf2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    acc2d = clf2d.score(XY_all, y_stk)
    print(f'  LogReg 2D acc={acc2d:.3f}  '
          f'proba range=[{grid_proba.min():.3f}, {grid_proba.max():.3f}]  '
          f'boundary visible: {grid_proba.min() < 0.5 < grid_proba.max()}')


    # ── Mini GAN với checkpoint tất cả epoch ─────────────────────────────────
    print(f'\nTraining mini GAN ({max(CKPTS)} epochs, saving each epoch)...')
    X_min = X_mal_train.min(axis=0)
    X_max = X_mal_train.max(axis=0)
    X_scaled = (X_mal_train - X_min) / (X_max - X_min + 1e-8) * 2 - 1

    gen  = build_mini_gen(LATENT, input_dim)
    disc = build_mini_disc(input_dim)
    disc.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 loss='binary_crossentropy')
    disc.trainable = False
    gi = layers.Input(shape=(LATENT,))
    go = disc(gen(gi))
    gan_m = models.Model(gi, go)
    gan_m.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  loss='binary_crossentropy')
    disc.trainable = True

    fixed_noise = RNG.normal(0, 1, (N_GEN, LATENT))
    n = len(X_scaled)
    bs = 64

    def sample_gen():
        raw = gen.predict(fixed_noise, verbose=0)
        return (raw + 1) / 2 * (X_max - X_min + 1e-8) + X_min

    all_frames = {}  # epoch → (gen_2d, mean_prob, mean_err)

    # epoch 0
    X_g0 = sample_gen()
    prob0 = get_proba(stacking, X_g0)
    err0  = dede.get_reconstruction_error(X_g0)
    all_frames[0] = (pca.transform(X_g0), prob0.mean(), err0.mean())
    print(f'  ep=0  P={prob0.mean():.3f}  err={err0.mean():.4f}')

    for epoch in range(1, max(CKPTS)+1):
        batches = n // bs
        for _ in range(batches):
            idx  = RNG.randint(0, n, bs) if bs < n else np.arange(n)
            real = X_scaled[idx]
            noise = RNG.normal(0, 1, (bs, LATENT))
            fake  = gen.predict(noise, verbose=0)
            disc.trainable = True
            disc.train_on_batch(real, np.ones((bs, 1)))
            disc.train_on_batch(fake, np.zeros((bs, 1)))
            disc.trainable = False
            noise2 = RNG.normal(0, 1, (bs, LATENT))
            gan_m.train_on_batch(noise2, np.ones((bs, 1)))

        if epoch in CKPTS:
            X_g = sample_gen()
            p   = get_proba(stacking, X_g)
            e   = dede.get_reconstruction_error(X_g)
            all_frames[epoch] = (pca.transform(X_g), p.mean(), e.mean())
            print(f'  ep={epoch:3d}  P={p.mean():.3f}  err={e.mean():.4f}')

    # ── Animation ─────────────────────────────────────────────────────────────
    print('\nRendering animation...')
    BG    = '#0d1117'
    CBEN  = '#4CAF50'
    CMAL  = '#F44336'
    CGAN  = '#FF9800'
    thr   = np.percentile(dede.get_reconstruction_error(X_test[:2000]), 99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    ax_main = axes[0]   # PCA + boundary
    ax_info = axes[1]   # P(mal) timeline

    epochs_so_far = []
    probs_so_far  = []

    def draw_frame(ep):
        ax_main.clear()
        ax_info.clear()
        for ax in [ax_main, ax_info]:
            ax.set_facecolor('#161b22')
            ax.spines[['bottom','left','top','right']].set_color('#30363d')
            ax.tick_params(colors='white')

        gen_2d, mean_prob, mean_err = all_frames[ep]
        epochs_so_far.append(ep)
        probs_so_far.append(mean_prob)

        # ── Main: PCA + decision boundary ────────────────────────────────────
        # Decision boundary contour
        # Vùng xanh = benign zone, vùng đỏ = malicious zone
        ax_main.contourf(xx, yy, grid_proba,
                         levels=[0.0, 0.5, 1.0],
                         colors=[CBEN, CMAL], alpha=0.15)
        # Đường biên rõ ràng
        cs = ax_main.contour(xx, yy, grid_proba,
                             levels=[0.5], colors=['white'], linewidths=[3.0])
        # Vùng mờ xung quanh biên (±0.1)
        ax_main.contourf(xx, yy, grid_proba,
                         levels=[0.4, 0.6],
                         colors=['yellow'], alpha=0.12)
        ax_main.text(x_max * 0.6, y_max * 0.85,
                     'Decision\nboundary\n(P = 0.5)',
                     color='white', fontsize=9, ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='#111', alpha=0.7))

        # Background: benign & malicious
        ax_main.scatter(ben_2d[:,0], ben_2d[:,1],
                        c=CBEN, alpha=0.25, s=15, label='Benign')
        ax_main.scatter(mal_2d[:,0], mal_2d[:,1],
                        c=CMAL, alpha=0.25, s=15, label='Malicious')

        # GAN samples
        sc = ax_main.scatter(gen_2d[:,0], gen_2d[:,1],
                             c=[mean_prob]*len(gen_2d),
                             cmap='RdYlGn_r', vmin=0, vmax=1,
                             alpha=0.8, s=35, marker='^',
                             edgecolors='white', linewidths=0.4,
                             label=f'GAN (epoch {ep})', zorder=5)

        ax_main.set_xlabel(f'PC1 ({var[0]:.1f}%)', color='white', fontsize=10)
        ax_main.set_ylabel(f'PC2 ({var[1]:.1f}%)', color='white', fontsize=10)
        ax_main.set_title(f'Feature Space (PCA 2D) — Epoch {ep:3d}\n'
                          f'GAN  P(malicious)={mean_prob:.3f}  DeDe_err={mean_err:.4f}',
                          color='white', fontsize=11, fontweight='bold')
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.legend(fontsize=9, facecolor='#21262d', labelcolor='white', loc='upper left')

        # DeDe threshold label
        below = (mean_err < thr)
        status = '✅ Below DeDe thr' if below else '❌ Above DeDe thr'
        color_s = '#4CAF50' if below else '#F44336'
        ax_main.text(0.02, 0.04, f'DeDe: {status}',
                     transform=ax_main.transAxes, fontsize=9,
                     color=color_s, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

        # ── Right: P(malicious) timeline ──────────────────────────────────────
        all_eps = sorted(all_frames.keys())
        all_ps  = [all_frames[e][1] for e in all_eps]

        ax_info.plot(all_eps, all_ps, color='#555', linewidth=1.5, alpha=0.5)
        ax_info.plot(epochs_so_far, probs_so_far,
                     'o-', color=CGAN, linewidth=2.5, markersize=5)
        ax_info.scatter([ep], [mean_prob], color='white', s=120,
                        zorder=6, edgecolors=CGAN, linewidths=2)

        ax_info.axhline(0.5, color='white', linewidth=2, linestyle='--', alpha=0.9)
        ax_info.axhline(get_proba(stacking, X_ben).mean(),
                        color=CBEN, linewidth=1.5, linestyle=':', alpha=0.7)
        ax_info.axhline(get_proba(stacking, X_mal).mean(),
                        color=CMAL, linewidth=1.5, linestyle=':', alpha=0.7)

        ax_info.fill_between([0, max(CKPTS)], 0.45, 0.55,
                             alpha=0.1, color='white', label='Danger zone (near boundary)')
        ax_info.text(max(CKPTS)*0.5, 0.5, 'Decision boundary',
                     color='white', fontsize=9, ha='center', va='center', alpha=0.7)

        ax_info.set_xlabel('Training Epoch', color='white', fontsize=11)
        ax_info.set_ylabel('Mean P(Malicious)', color='white', fontsize=11)
        ax_info.set_title('GAN tiến dần về biên quyết định\ntheo epoch training',
                          color='white', fontsize=11, fontweight='bold')
        ax_info.set_xlim(0, max(CKPTS))
        ax_info.set_ylim(-0.05, 1.05)

        fig.suptitle('GAN Attack: Training Trajectory đến Decision Boundary',
                     color='white', fontsize=13, fontweight='bold')
        plt.tight_layout()

    # Pre-compute benign/malicious reference proba (only once)
    _prob_ben_ref = get_proba(stacking, X_ben).mean()
    _prob_mal_ref = get_proba(stacking, X_mal).mean()

    sorted_epochs = sorted(all_frames.keys())

    # Vẽ tất cả frames
    print(f'  Rendering {len(sorted_epochs)} frames...')
    frames = []
    for ep in sorted_epochs:
        epochs_so_far.clear()
        probs_so_far.clear()
        for e in sorted_epochs:
            if e <= ep:
                epochs_so_far.append(e)
                probs_so_far.append(all_frames[e][1])
        draw_frame(ep)
        fname = out_dir / f'_frame_{ep:03d}.png'
        fig.savefig(fname, dpi=100, bbox_inches='tight', facecolor=BG)
        frames.append(fname)

    plt.close(fig)

    # Tạo GIF từ frames
    print('  Building GIF...')
    import PIL.Image as Image
    imgs = [Image.open(f) for f in frames]

    # Thêm frame cuối lặp lại 3 lần để dừng
    imgs_out = imgs + [imgs[-1]] * (FPS * 3)

    gif_path = out_dir / 'gan_animation.gif'
    imgs_out[0].save(
        gif_path,
        save_all=True,
        append_images=imgs_out[1:],
        optimize=False,
        duration=int(1000 / FPS),
        loop=0
    )

    # Xóa frames tạm
    for f in frames:
        f.unlink()

    print(f'\n✅ Animation saved: {gif_path}')
    print(f'   Frames: {len(sorted_epochs)}  FPS: {FPS}  Duration: {len(sorted_epochs)/FPS:.1f}s')


if __name__ == '__main__':
    main()
