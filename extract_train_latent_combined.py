#!/usr/bin/env python3
"""
Extract latent features từ TRAIN malicious + benign của CICIDS2017 và lưu vào 1 file

Script này:
1. Load train_malicious_X.npy và train_benign_X.npy
2. Train một Autoencoder trên cả 2 loại
3. Extract latent features cho tất cả train samples
4. Lưu vào 1 file CSV/NPY duy nhất với cả features và labels
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# ===== TensorFlow / Keras imports =====
# Sử dụng TensorFlow nếu có, nếu không thì dùng PyTorch
BACKEND = None
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    BACKEND = 'tensorflow'
    print("✓ Using TensorFlow backend")
except ImportError:
    pass

if BACKEND is None:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        BACKEND = 'pytorch'
        print("✓ Using PyTorch backend")
    except ImportError:
        pass

if BACKEND is None:
    print("❌ Neither TensorFlow nor PyTorch found!")
    print("   Please install: pip install tensorflow  OR  pip install torch")
    exit(1)

# Paths
BASE_DIR = Path(__file__).parent
SPLITS_DIR = BASE_DIR / "datasets" / "splits" / "cicids2017"
OUTPUT_DIR = BASE_DIR / "datasets" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===== Autoencoder với TensorFlow =====
def build_autoencoder_tf(input_dim, latent_dim=32):
    """Build autoencoder using TensorFlow/Keras"""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    latent = Dense(latent_dim, activation='relu', name='latent')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = Model(input_layer, output_layer, name='autoencoder')
    encoder = Model(input_layer, latent, name='encoder')
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder, encoder


def train_autoencoder_tf(autoencoder, X_train, epochs=50, batch_size=256):
    """Train autoencoder using TensorFlow"""
    print("\n🔥 Training Autoencoder (TensorFlow)...")
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )
    print("✓ Training complete!")
    return history


# ===== Autoencoder với PyTorch =====
if BACKEND == 'pytorch':
    class AutoencoderPyTorch(nn.Module):
        def __init__(self, input_dim, latent_dim=32):
            super(AutoencoderPyTorch, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed
        
        def encode(self, x):
            return self.encoder(x)


    def train_autoencoder_pytorch(model, X_train, epochs=50, batch_size=256, device='cpu'):
        """Train autoencoder using PyTorch"""
        print(f"\n🔥 Training Autoencoder (PyTorch on {device})...")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_train).to(device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("✓ Training complete!")
        return model


    def extract_latent_pytorch(model, X, device='cpu'):
        """Extract latent features using PyTorch"""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            latent = model.encode(X_tensor)
            return latent.cpu().numpy()


def main():
    print("\n" + "="*70)
    print("🎯 EXTRACT TRAIN LATENT FEATURES (MALICIOUS + BENIGN)")
    print("="*70)
    
    # ===== 1. Load train data =====
    print("\n📂 Loading train data...")
    train_mal_X = np.load(SPLITS_DIR / "train_malicious_X.npy")
    train_mal_y = np.load(SPLITS_DIR / "train_malicious_y.npy")
    train_ben_X = np.load(SPLITS_DIR / "train_benign_X.npy")
    train_ben_y = np.load(SPLITS_DIR / "train_benign_y.npy")
    
    print(f"✓ Train malicious: {train_mal_X.shape}")
    print(f"✓ Train benign:    {train_ben_X.shape}")
    
    # Combine
    X_train = np.vstack([train_ben_X, train_mal_X])
    y_train = np.hstack([train_ben_y, train_mal_y])
    
    print(f"\n✓ Combined train: {X_train.shape}")
    print(f"  - Benign:    {(y_train==0).sum():,}")
    print(f"  - Malicious: {(y_train==1).sum():,}")
    
    input_dim = X_train.shape[1]
    latent_dim = 32
    
    # ===== 2. Train Autoencoder =====
    if BACKEND == 'tensorflow':
        autoencoder, encoder = build_autoencoder_tf(input_dim, latent_dim)
        train_autoencoder_tf(autoencoder, X_train, epochs=50, batch_size=256)
        
        # Save models
        autoencoder_path = OUTPUT_DIR / f"autoencoder_{TIMESTAMP}.h5"
        encoder_path = OUTPUT_DIR / f"encoder_{TIMESTAMP}.h5"
        autoencoder.save(autoencoder_path)
        encoder.save(encoder_path)
        print(f"\n💾 Saved autoencoder: {autoencoder_path}")
        print(f"💾 Saved encoder: {encoder_path}")
        
        # Extract latent
        print("\n🔍 Extracting latent features...")
        X_train_latent = encoder.predict(X_train, verbose=0)
        
    else:  # PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoencoderPyTorch(input_dim, latent_dim)
        model = train_autoencoder_pytorch(model, X_train, epochs=50, batch_size=256, device=device)
        
        # Save model
        model_path = OUTPUT_DIR / f"autoencoder_{TIMESTAMP}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 Saved autoencoder: {model_path}")
        
        # Extract latent
        print("\n🔍 Extracting latent features...")
        X_train_latent = extract_latent_pytorch(model, X_train, device=device)
    
    print(f"✓ Latent features shape: {X_train_latent.shape}")
    
    # ===== 3. Create DataFrame =====
    print("\n📊 Creating DataFrame...")
    
    # Create column names for latent features
    latent_cols = [f'latent_{i}' for i in range(latent_dim)]
    
    df = pd.DataFrame(X_train_latent, columns=latent_cols)
    df['label'] = y_train
    
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # ===== 4. Save results =====
    print("\n💾 Saving results...")
    
    # Save as CSV
    csv_path = OUTPUT_DIR / f"cicids2017_train_latent_combined_{TIMESTAMP}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV: {csv_path}")
    
    # Save as NPY (separate X and y)
    npy_X_path = OUTPUT_DIR / f"cicids2017_train_latent_X_{TIMESTAMP}.npy"
    npy_y_path = OUTPUT_DIR / f"cicids2017_train_latent_y_{TIMESTAMP}.npy"
    np.save(npy_X_path, X_train_latent)
    np.save(npy_y_path, y_train)
    print(f"✓ NPY X: {npy_X_path}")
    print(f"✓ NPY y: {npy_y_path}")
    
    # Save metadata
    metadata = {
        'timestamp': TIMESTAMP,
        'backend': BACKEND,
        'original_feature_dim': int(input_dim),
        'latent_dim': int(latent_dim),
        'total_samples': int(len(X_train)),
        'benign_samples': int((y_train==0).sum()),
        'malicious_samples': int((y_train==1).sum()),
        'csv_file': str(csv_path),
        'npy_X_file': str(npy_X_path),
        'npy_y_file': str(npy_y_path),
        'autoencoder_model': str(autoencoder_path) if BACKEND == 'tensorflow' else str(model_path),
        'encoder_model': str(encoder_path) if BACKEND == 'tensorflow' else None,
        'source_files': {
            'train_malicious_X': str(SPLITS_DIR / "train_malicious_X.npy"),
            'train_benign_X': str(SPLITS_DIR / "train_benign_X.npy"),
        }
    }
    
    meta_path = OUTPUT_DIR / f"cicids2017_train_latent_metadata_{TIMESTAMP}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata: {meta_path}")
    
    # ===== 5. Summary =====
    print("\n" + "="*70)
    print("✅ EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"   {csv_path.name}")
    print(f"   {npy_X_path.name}")
    print(f"   {npy_y_path.name}")
    print(f"   {meta_path.name}")
    
    print(f"\n📊 Statistics:")
    print(f"   Original dimension: {input_dim}")
    print(f"   Latent dimension:   {latent_dim}")
    print(f"   Compression ratio:  {input_dim/latent_dim:.1f}x")
    print(f"   Total samples:      {len(X_train):,}")
    print(f"   Benign:             {(y_train==0).sum():,}")
    print(f"   Malicious:          {(y_train==1).sum():,}")
    
    print("\n💡 Usage:")
    print("   # Load CSV")
    print(f"   df = pd.read_csv('{csv_path}')")
    print("   X = df.drop('label', axis=1).values")
    print("   y = df['label'].values")
    print()
    print("   # Or load NPY")
    print(f"   X = np.load('{npy_X_path}')")
    print(f"   y = np.load('{npy_y_path}')")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
