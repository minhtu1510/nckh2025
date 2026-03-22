import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path('/run/media/mtu/4AE886A9E886933D/NCKH2025/NCKH_code/ids_research')

raw_dir = BASE_DIR / "datasets/splits/3.0_raw_from_latent/exp3_gan_attack"
out_dir = BASE_DIR / "datasets/splits/3.2_latent_single_enc"

# Load new raw GAN data
print("Loading new RAW GAN data...")
X_gan_test = np.load(raw_dir / 'X_test.npy').astype(np.float32)
y_gan_test = np.load(raw_dir / 'y_test.npy')
if (raw_dir / 'X_train.npy').exists():
    X_gan_tr = np.load(raw_dir / 'X_train.npy').astype(np.float32)
    y_gan_tr = np.load(raw_dir / 'y_train.npy')
else:
    # use baseline if not exist
    base_dir = BASE_DIR / "datasets/splits/3.0_raw_from_latent/exp1_baseline"
    X_gan_tr = np.load(base_dir / 'X_train.npy').astype(np.float32)
    y_gan_tr = np.load(base_dir / 'y_train.npy')

# Load existing clean Single Encoder
enc_path = out_dir / 'models/single_encoder.h5'
print(f"Loading existing Single Encoder from {enc_path}...")
encoder = tf.keras.models.load_model(str(enc_path))

# Encode GAN test data
print("Encoding GAN data to latent space...")
Z_test = encoder.predict(X_gan_test, verbose=1)
Z_train = encoder.predict(X_gan_tr, verbose=1)

# Save
save_dir = out_dir / 'exp11_gan_attack'
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / 'X_test.npy', Z_test)
np.save(save_dir / 'y_test.npy', y_gan_test)
np.save(save_dir / 'X_train.npy', Z_train)
np.save(save_dir / 'y_train.npy', y_gan_tr)

print(f"Successfully updated Exp11 GAN Latent Data in {save_dir}!")
