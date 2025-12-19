"""
Train a GAN-based adversarial generator to fool the IDS classifier on TRAbID 2017.

The script:
1. Loads and preprocesses the dataset (min-max scaling, stratified split).
2. Trains the baseline MLP classifier (same architecture as adversarial_ml.py).
3. Trains a generator + discriminator pair where the generator learns perturbations
   that make malicious samples appear benign to the classifier.
4. Evaluates the classifier on original and GAN-crafted adversarial samples, saving
   reports to the Results/ directory.

This implementation uses TensorFlow/Keras 1.x APIs to remain compatible with the
original code base.
"""

from __future__ import print_function

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

# Keras imports from tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "datasets" / "raw"
RESULTS_ROOT = BASE_DIR / "results"
METRICS_DIR = RESULTS_ROOT / "metrics"
ADV_SAMPLES_DIR = RESULTS_ROOT / "adversarial_samples"
LOGS_DIR = RESULTS_ROOT / "logs"
FIGURES_DIR = RESULTS_ROOT / "figures"
for directory in (METRICS_DIR, ADV_SAMPLES_DIR, LOGS_DIR, FIGURES_DIR):
    os.makedirs(directory, exist_ok=True)

CURRENT_RUN_ID = None


def get_run_id(proposed: Optional[str] = None) -> str:
    global CURRENT_RUN_ID
    if proposed is not None:
        CURRENT_RUN_ID = proposed
    if CURRENT_RUN_ID is None:
        CURRENT_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CURRENT_RUN_ID


def ensure_results_dir():
    for directory in (METRICS_DIR, ADV_SAMPLES_DIR, LOGS_DIR, FIGURES_DIR):
        os.makedirs(directory, exist_ok=True)


def build_results_path(filename):
    ensure_results_dir()
    return os.path.join(METRICS_DIR, filename)


def save_dataframe(df, filename):
    path = Path(build_results_path(filename))
    df.to_csv(path, index=False)
    print("Saved {}".format(path))
    return path


def load_tribid_dataset():
    """Load TRAbID 2017 dataset and perform min-max scaling."""
    from scipy.io import arff

    data = arff.loadarff(str(DATA_DIR / "TRAbID2017_dataset.arff"))
    dataset = pd.DataFrame(data[0])

    X = dataset.iloc[:, 0:43].values
    Y = pd.read_csv(DATA_DIR / "TRAbID2017_dataset_Y_class.csv").iloc[:, 0].values

    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print("Data loaded. Train shape: {}, Test shape: {}".format(X_train.shape, X_test.shape))
    return (X_train, X_test, Y_train, Y_test), scaler


def build_classifier(input_dim):
    """Baseline MLP classifier identical to adversarial_ml.py."""
    model = Sequential(name="ids_classifier")
    model.add(
        Dense(
            units=round(input_dim / 2),
            kernel_initializer="uniform",
            activation="relu",
            input_dim=input_dim,
        )
    )
    model.add(Dense(units=round(input_dim / 2), kernel_initializer="uniform", activation="relu"))
    model.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_classifier(model, X_train, Y_train, validation_split=0.1, batch_size=64, epochs=100):
    callbacks = [EarlyStopping(monitor="val_loss", patience=2)]
    history = model.fit(
        X_train,
        Y_train,
        validation_split=validation_split,
        callbacks=callbacks,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=2,
    )
    print("Classifier training complete.")
    return history


def evaluate_classifier(model, X, Y, label):
    """Evaluate classifier and store reports."""
    Y_pred_prob = model.predict(X)
    Y_pred = (Y_pred_prob > 0.5).astype(int).ravel()
    report = classification_report(Y, Y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(Y, Y_pred)
    metrics_summary = {
        "accuracy": accuracy_score(Y, Y_pred),
        "precision": precision_score(Y, Y_pred),
        "recall": recall_score(Y, Y_pred),
        "f1": f1_score(Y, Y_pred),
    }

    print("[{}] metrics: {}".format(label, metrics_summary))
    metrics_path = save_dataframe(
        pd.DataFrame([metrics_summary]),
        "{}_metrics_{}.csv".format(label, get_run_id()),
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = save_dataframe(
        report_df,
        "{}_classification_report_{}.csv".format(label, get_run_id()),
    )

    cm_df = pd.DataFrame(
        cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]
    )
    cm_path = save_dataframe(
        cm_df,
        "{}_confusion_matrix_{}.csv".format(label, get_run_id()),
    )

    return {
        "metrics_path": metrics_path,
        "report_path": report_path,
        "confusion_matrix_path": cm_path,
        "summary": metrics_summary,
    }


def build_generator(feature_dim, noise_dim=32, epsilon=0.1):
    """Conditional generator that returns clipped adversarial samples."""
    orig_input = Input(shape=(feature_dim,), name="orig_sample")
    noise_input = Input(shape=(noise_dim,), name="noise_vector")
    x = Concatenate()([orig_input, noise_input])
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    perturbation = Dense(feature_dim, activation="tanh")(x)
    perturbation = Lambda(lambda p: epsilon * p, name="scaled_perturbation")(perturbation)
    adv_sample = Lambda(
        lambda inputs: K.clip(inputs[0] + inputs[1], 0.0, 1.0),
        name="adv_sample",
    )([orig_input, perturbation])
    return Model([orig_input, noise_input], adv_sample, name="generator")


def build_discriminator(feature_dim):
    model = Sequential(name="discriminator")
    model.add(Dense(128, activation="relu", input_dim=feature_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_combined(generator, discriminator, classifier, feature_dim, noise_dim=32, cls_weight=5.0):
    """Create model to train generator against discriminator + fixed classifier."""
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False

    classifier.trainable = False
    for layer in classifier.layers:
        layer.trainable = False

    orig_input = Input(shape=(feature_dim,), name="combined_orig")
    noise_input = Input(shape=(noise_dim,), name="combined_noise")

    adv_sample = generator([orig_input, noise_input])
    validity = discriminator(adv_sample)
    benign_pred = classifier(adv_sample)

    combined = Model(
        inputs=[orig_input, noise_input],
        outputs=[validity, benign_pred],
        name="combined_model",
    )
    combined.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=["binary_crossentropy", "binary_crossentropy"],
        loss_weights=[1.0, cls_weight],
    )
    return combined


def train_gan(
    generator,
    discriminator,
    combined,
    classifier,
    X_train,
    Y_train,
    noise_dim=32,
    epochs=2000,
    batch_size=128,
):
    """Train GAN to craft adversarial examples from malicious samples."""
    y_train = Y_train.astype(int)
    malicious_mask = y_train == 1
    benign_mask = y_train == 0

    X_malicious = X_train[malicious_mask]
    X_benign = X_train[benign_mask]

    if X_malicious.shape[0] == 0:
        raise ValueError("No malicious samples found in training data.")

    valid_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))
    target_benign = np.zeros((batch_size, 1))

    for epoch in range(1, epochs + 1):
        # Train discriminator
        malicious_idx = np.random.randint(0, X_malicious.shape[0], batch_size)
        benign_idx = np.random.randint(0, X_benign.shape[0], batch_size)

        malicious_samples = X_malicious[malicious_idx]
        benign_samples = X_benign[benign_idx]

        noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_dim))
        generated_samples = generator.predict([malicious_samples, noise], verbose=0)

        d_loss_real = discriminator.train_on_batch(benign_samples, valid_label)
        d_loss_fake = discriminator.train_on_batch(generated_samples, fake_label)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        g_loss = combined.train_on_batch(
            [malicious_samples, noise], [valid_label, target_benign]
        )

        if epoch % 100 == 0 or epoch == 1:
            benign_success = classifier.predict(generated_samples, verbose=0)
            benign_success = (benign_success < 0.5).mean()
            print(
                "Epoch {:04d} | D loss: {:.4f} acc: {:.4f} | G loss: {:.4f} | Benign success: {:.2%}".format(
                    epoch, d_loss[0], d_loss[1], g_loss[0], benign_success
                )
            )


def craft_adversarial_samples(generator, X_source, noise_dim=32):
    noise = np.random.uniform(-1.0, 1.0, size=(X_source.shape[0], noise_dim))
    return generator.predict([X_source, noise], verbose=0)


def run_gan_attack(
    processed_csv: Optional[Path] = None,
    model_outputs: Optional[Dict] = None,
    run_id: Optional[str] = None,
    epochs: int = 2000,
    batch_size: int = 128,
    noise_dim: int = 32,
    epsilon: float = 0.1,
):
    """
    Execute GAN-based attack pipeline and return artefact paths.
    
    Parameters
    ----------
    processed_csv : Path
        Path to processed dataset CSV
    model_outputs : dict
        Outputs from baseline model training (contains trained model)
    run_id : str, optional
        Run identifier
    epochs : int
        Number of GAN training epochs
    batch_size : int
        Batch size for GAN training
    noise_dim : int
        Noise vector dimension
    epsilon : float
        Perturbation budget
        
    Returns
    -------
    dict
        Paths to generated outputs and metrics
    """
    ensure_results_dir()
    run_id = get_run_id(run_id)
    
    # Load data from processed CSV (CICIDS2017 or other)
    if processed_csv is None:
        # Fallback to old TRAbID for backward compatibility
        print("⚠️  No processed_csv provided, using TRAbID dataset")
        (X_train, X_test, Y_train, Y_test), _ = load_tribid_dataset()
    else:
        print(f"✓ Loading data from: {processed_csv}")
        data = pd.read_csv(processed_csv)
        X = data.drop(columns=['label']).values
        y = data['label'].values
        
        # IMPORTANT: Use SAME random_state as baseline training!
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,  # ← MUST match baseline!
            stratify=y
        )
        print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"✓ Using random_state=42 (same as baseline)")
    
    feature_dim = X_train.shape[1]
    
    # Build and train classifier
    classifier = build_classifier(feature_dim)
    history = train_classifier(classifier, X_train, Y_train)
    
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = "epoch"
    history_path = save_dataframe(
        hist_df, f"gan_classifier_history_{run_id}.csv"
    )
    
    # Evaluate baseline
    baseline_outputs = evaluate_classifier(classifier, X_test, Y_test, label="baseline")
    
    # Build GAN
    generator = build_generator(feature_dim, noise_dim=noise_dim, epsilon=epsilon)
    discriminator = build_discriminator(feature_dim)
    combined = build_combined(
        generator, discriminator, classifier, feature_dim, noise_dim=noise_dim
    )
    
    # Train GAN on TRAIN malicious samples
    print("\n🔥 Training GAN...")
    train_gan(
        generator,
        discriminator,
        combined,
        classifier,
        X_train,  # GAN trains on train set
        Y_train,
        noise_dim=noise_dim,
        epochs=epochs,
        batch_size=batch_size,
    )
    
    # Generate adversarial from TEST malicious samples ← CRITICAL!
    print("\n⚔️  Generating adversarial samples from TEST malicious...")
    malicious_mask = Y_test == 1
    print(f"   Test malicious samples: {malicious_mask.sum():,}")
    
    X_adv = X_test.copy()
    X_adv_malicious = craft_adversarial_samples(
        generator, X_test[malicious_mask], noise_dim=noise_dim
    )
    X_adv[malicious_mask] = X_adv_malicious
    
    # Evaluate on adversarial test set
    adversarial_outputs = evaluate_classifier(
        classifier, X_adv, Y_test, label="gan_adversarial"
    )
    
    # Save adversarial samples
    adv_path = ADV_SAMPLES_DIR / f"gan_adversarial_samples_{run_id}.npy"
    np.save(adv_path, X_adv)
    
    # Compute perturbation statistics
    perturbations = X_adv_malicious - X_test[malicious_mask]
    perturbation_df = pd.DataFrame(
        {
            "l2_norm": np.linalg.norm(perturbations, axis=1),
            "linf_norm": np.max(np.abs(perturbations), axis=1),
        }
    )
    perturbation_path = save_dataframe(
        perturbation_df, f"gan_perturbation_stats_{run_id}.csv"
    )
    
    print("\n✅ GAN attack workflow complete.")
    print(f"   Adversarial samples: {adv_path}")
    print(f"   Baseline accuracy: {baseline_outputs['summary']['accuracy']:.4f}")
    print(f"   Under GAN attack: {adversarial_outputs['summary']['accuracy']:.4f}")
    print(f"   Accuracy drop: {baseline_outputs['summary']['accuracy'] - adversarial_outputs['summary']['accuracy']:.4f}")
    
    return {
        "run_id": run_id,
        "history_path": history_path,
        "baseline": baseline_outputs,
        "adversarial": adversarial_outputs,
        "adversarial_samples": adv_path,
        "perturbation_stats": perturbation_path,
    }


def main():
    run_gan_attack()


if __name__ == "__main__":
    main()


__all__ = ["run_gan_attack"]
