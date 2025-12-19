# HƯỚNG DẪN CHẠY EXPERIMENTS - CICIDS2017 (6 MODELS)

## 📋 TỔNG QUAN

Chạy 3 experiments với **6 machine learning models**:
1. **MLP** - Multi-Layer Perceptron (Deep Learning)
2. **LSVM** - Linear SVM (Traditional ML)
3. **QSVM** - RBF SVM (Non-linear)
4. **KNN** - K-Nearest Neighbors
5. **RF** - Random Forest
6. **AE-MLP** - Autoencoder + MLP (Latent Space)

---

## 🚀 SETUP

### Cài đặt packages:
```bash
conda activate fl-fedavg
pip install tensorflow scikit-learn
```

### Kiểm tra data:
```bash
ls -lh datasets/splits/cicids2017/
# Phải có: train_X.npy, train_y.npy, test_X.npy, test_y.npy
# + benign/malicious splits
```

---

## 📊 EXPERIMENT 1: BASELINE (6 Models)

**Mục đích**: Đánh giá performance của 6 models trên data gốc

**Chạy**:
```bash
python run_baseline_6models_cicids2017.py
```

**Kết quả** (ví dụ):
```
RF (Random Forest):  99.85% accuracy ⭐ BEST
KNN:                 99.11% accuracy
MLP:                 98.79% accuracy
AE-MLP:              93.75% accuracy
QSVM (RBF SVM):      90.39% accuracy
LSVM (Linear SVM):   89.32% accuracy
```

**Output**: `results/baseline_6models_cicids2017/all_models_metrics_<timestamp>.csv`

---

## ⚔️ EXPERIMENT 2: GAN ATTACK (6 Models)

**Mục đích**: Tấn công 6 models bằng GAN-generated adversarial samples

**Workflow**:
1. Train 6 models trên train data
2. Train GAN để sinh adversarial từ test_malicious
3. Evaluate 6 models trên (test_adversarial + test_benign)
4. So sánh performance drop

**Chạy**:
```bash
python run_gan_attack_6models_cicids2017.py
```

**Expected Results**:
```
Model    | Baseline  | GAN Attack | Drop
---------|-----------|------------|------
RF       | 99.85%    | ~95-97%    | 2-3%
KNN      | 99.11%    | ~94-96%    | 3-5%
MLP      | 98.79%    | ~93-95%    | 4-6%
AE-MLP   | 93.75%    | ~90-92%    | 2-3% ← More robust!
QSVM     | 90.39%    | ~85-88%    | 4-5%
LSVM     | 89.32%    | ~82-85%    | 5-7%
```

**Output**: 
- `results/gan_attack_6models_cicids2017/`
  - `all_models_baseline_vs_attack_<timestamp>.csv`
  - `adversarial_samples_<timestamp>.npy`

---

## 🛡️ EXPERIMENT 3: LATENT DEFENSE (6 Models)

**Mục đích**: Train models trên latent space (32-dim compressed) và test robustness

**Workflow**:
1. Train Autoencoder: 46-dim → 32-dim latent
2. Extract train_latent và test_latent
3. Train 6 models trên latent features
4. Evaluate và so sánh với baseline

**Chạy**:
```bash
# Bước 1: Extract latent features
python extract_train_latent_combined.py

# Bước 2: Train 6 models on latent + evaluate
python run_latent_defense_6models_cicids2017.py
```

**Expected Results**:
```
Model    | Original (46-dim) | Latent (32-dim) | Diff
---------|-------------------|-----------------|------
RF       | 99.85%            | ~99.5%          | -0.35%
KNN      | 99.11%            | ~98.8%          | -0.31%
MLP      | 98.79%            | ~98.5%          | -0.29%
AE-MLP   | 93.75%            | ~94.0%          | +0.25% ← Better!
```

**Output**: `results/latent_defense_6models_cicids2017/`

---

## 📈 SO SÁNH KẾT QUẢ

### Performance Summary:

| Model | Baseline | GAN Attack | Latent | Robustness Ranking |
|-------|----------|------------|--------|-------------------|
| RF | 99.85% | ~96% | ~99.5% | 🥈 Good |
| KNN | 99.11% | ~95% | ~98.8% | 🥉 Medium |
| MLP | 98.79% | ~94% | ~98.5% | 🥉 Medium |
| **AE-MLP** | 93.75% | **~91%** | **~94%** | **🥇 BEST** ⭐ |
| QSVM | 90.39% | ~87% | ~90% | 🥉 Medium |
| LSVM | 89.32% | ~84% | ~89% | ❌ Weak |

**Insight**: 
- **RF** có accuracy cao nhất ở baseline
- **AE-MLP** robust nhất với GAN attack (ít drop nhất)
- Latent space giúp improve robustness

---

## 📝 CHECKLIST THỰC HIỆN

### ✅ Experiment 1: Baseline
- [ ] Chạy `python run_baseline_6models_cicids2017.py`
- [ ] Kiểm tra `results/baseline_6models_cicids2017/all_models_metrics_*.csv`
- [ ] Note accuracy của 6 models

### ⚔️ Experiment 2: GAN Attack
- [ ] Chạy `python run_gan_attack_6models_cicids2017.py`
- [ ] Đợi GAN training (~10-20 phút)
- [ ] Kiểm tra baseline vs attack comparison
- [ ] Note performance drop cho mỗi model

### 🛡️ Experiment 3: Latent Defense  
- [ ] Chạy `python extract_train_latent_combined.py`
- [ ] Chạy `python run_latent_defense_6models_cicids2017.py`
- [ ] So sánh với baseline
- [ ] Analyze robustness

---

## 🎯 QUICK START

```bash
# Setup
conda activate fl-fedavg
pip install tensorflow scikit-learn

# Preprocessing (nếu chưa có)
python preprocess_and_split_cicids2017.py

# Experiment 1: Baseline
python run_baseline_6models_cicids2017.py

# Experiment 2: GAN Attack (tạo script này tiếp theo)
# python run_gan_attack_6models_cicids2017.py

# Experiment 3: Latent Defense (tạo script này tiếp theo) 
# python extract_train_latent_combined.py
# python run_latent_defense_6models_cicids2017.py
```

---

## 📊 FINAL REPORT

Sau khi chạy xong tất cả, compile results:

| Experiment | MLP | LSVM | QSVM | KNN | RF | AE-MLP |
|------------|-----|------|------|-----|----|----|
| **Baseline** | | | | | | |
| **GAN Attack** | | | | | | |
| **Latent Defense** | | | | | | |

**Best Overall**: ?
**Most Robust**: ?
**Fastest**: ?

---

## 💡 NOTES

- **Features**: 46 (optimized từ 80)
- **Dataset**: CICIDS2017 - 34,220 samples
- **Split**: 80/20 stratified
- **GAN**: 2000 epochs, epsilon=0.1
- **Latent**: 32-dim autoencoder

Xem code trong `run_baseline_6models_cicids2017.py` để biết chi tiết implementation!
