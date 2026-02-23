"""
DeDe-Adapted: Encoder-Decoder Architecture cho Network Traffic Data

Cải tiến từ DeDe (CVPR 2025) để phù hợp với dữ liệu mạng (tabular):
- Thay Vision Transformer → MLP-based Encoder
- Thay Masked Patches → Masked Features (feature-level masking)
- Decoder để reconstruct masked features
- Sử dụng reconstruction error để detect adversarial/backdoor samples

Ý tưởng chính:
1. Mask random features trong input
2. Encoder tạo latent representation
3. Decoder reconstruct original features
4. Samples với high reconstruction error = adversarial/backdoor
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class FeatureMasker(layers.Layer):
    """
    Mask random features trong tabular data
    Similar to masked patches in DeDe, nhưng cho features
    """
    def __init__(self, mask_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch_size, num_features)
        Returns:
            masked_inputs: inputs with random features masked to 0
            mask: binary mask (1 = keep, 0 = masked)
        """
        if not training:
            # During inference, don't mask
            return inputs, tf.ones_like(inputs)
        
        batch_size = tf.shape(inputs)[0]
        num_features = tf.shape(inputs)[1]
        
        # Random mask for each sample
        mask_prob = tf.ones((batch_size, num_features)) * (1 - self.mask_ratio)
        mask = tf.cast(
            tf.random.uniform((batch_size, num_features)) < (1 - self.mask_ratio),
            tf.float32
        )
        
        masked_inputs = inputs * mask
        
        return masked_inputs, mask
    
    def get_config(self):
        config = super().get_config()
        config.update({"mask_ratio": self.mask_ratio})
        return config


class TabularEncoder(keras.Model):
    """
    MLP-based Encoder cho tabular network data
    Giống ViT trong DeDe nhưng dùng MLP thay vì attention
    """
    def __init__(self, 
                 input_dim,
                 latent_dim=64,
                 hidden_dims=[256, 128],
                 dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # Build encoder layers
        self.encoder_layers = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder_layers.append(layers.Dense(
                hidden_dim,
                activation='relu',
                name=f'encoder_dense_{i}'
            ))
            self.encoder_layers.append(layers.BatchNormalization(
                name=f'encoder_bn_{i}'
            ))
            self.encoder_layers.append(layers.Dropout(
                dropout,
                name=f'encoder_dropout_{i}'
            ))
        
        # Latent representation
        self.latent_layer = layers.Dense(
            latent_dim,
            activation='relu',
            name='latent'
        )
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch_size, input_dim) - masked features
        Returns:
            latent: (batch_size, latent_dim)
        """
        x = inputs
        
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        
        latent = self.latent_layer(x, training=training)
        
        return latent
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout_rate
        }


class TabularDecoder(keras.Model):
    """
    MLP-based Decoder để reconstruct original features
    """
    def __init__(self,
                 latent_dim,
                 output_dim,
                 hidden_dims=[128, 256],
                 dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # Build decoder layers
        self.decoder_layers = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.decoder_layers.append(layers.Dense(
                hidden_dim,
                activation='relu',
                name=f'decoder_dense_{i}'
            ))
            self.decoder_layers.append(layers.BatchNormalization(
                name=f'decoder_bn_{i}'
            ))
            self.decoder_layers.append(layers.Dropout(
                dropout,
                name=f'decoder_dropout_{i}'
            ))
        
        # Reconstruction output
        self.output_layer = layers.Dense(
            output_dim,
            activation='linear',  # Linear for reconstruction
            name='reconstruction'
        )
    
    def call(self, latent, training=None):
        """
        Args:
            latent: (batch_size, latent_dim)
        Returns:
            reconstructed: (batch_size, output_dim)
        """
        x = latent
        
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        
        reconstructed = self.output_layer(x, training=training)
        
        return reconstructed
    
    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout_rate
        }


class DeDeAdapted(keras.Model):
    """
    DeDe-Adapted: Complete Encoder-Decoder model cho network traffic
    
    Training:
        - Mask random features
        - Encode → Decode → Reconstruct
        - Minimize reconstruction loss
    
    Inference (Detection):
        - Forward pass without masking
        - Calculate reconstruction error
        - High error → Adversarial/Backdoor sample
    """
    def __init__(self,
                 input_dim,
                 latent_dim=64,
                 encoder_hidden_dims=[256, 128],
                 decoder_hidden_dims=[128, 256],
                 mask_ratio=0.5,
                 dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        
        # Components
        self.masker = FeatureMasker(mask_ratio=mask_ratio)
        
        self.encoder = TabularEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout
        )
        
        self.decoder = TabularDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout
        )
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch_size, input_dim)
        Returns:
            reconstructed: (batch_size, input_dim)
            mask: (batch_size, input_dim) - binary mask
        """
        # Mask features during training
        masked_inputs, mask = self.masker(inputs, training=training)
        
        # Encode
        latent = self.encoder(masked_inputs, training=training)
        
        # Decode
        reconstructed = self.decoder(latent, training=training)
        
        return reconstructed, mask
    
    def train_step(self, data):
        """Custom training step with reconstruction loss"""
        x, _ = data  # We don't use labels for reconstruction
        
        with tf.GradientTape() as tape:
            # Forward pass
            reconstructed, mask = self(x, training=True)
            
            # Reconstruction loss - only on MASKED features
            # (DeDe paper focuses on reconstructing masked parts)
            masked_positions = 1 - mask  # 1 where masked, 0 where kept
            reconstruction_loss = tf.reduce_mean(
                masked_positions * tf.square(x - reconstructed)
            )
            
            # Add regularization
            total_loss = reconstruction_loss + sum(self.losses)
        
        # Gradient update
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(x, reconstructed)
        
        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss
        }
    
    def test_step(self, data):
        """Custom test step"""
        x, _ = data
        
        # Forward pass without masking (training=False)
        reconstructed, _ = self(x, training=False)
        
        # Full reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        
        return {'loss': reconstruction_loss}
    
    def get_reconstruction_error(self, inputs):
        """
        Calculate reconstruction error for anomaly detection
        
        Args:
            inputs: (batch_size, input_dim)
        Returns:
            errors: (batch_size,) - MSE for each sample
        """
        reconstructed, _ = self(inputs, training=False)
        
        # MSE per sample
        errors = tf.reduce_mean(
            tf.square(inputs - reconstructed),
            axis=1
        )
        
        return errors.numpy()
    
    def detect_adversarial(self, inputs, threshold=None, percentile=95):
        """
        Detect adversarial samples using reconstruction error
        
        Args:
            inputs: (batch_size, input_dim)
            threshold: custom threshold, or None to use percentile
            percentile: percentile to use as threshold if threshold=None
        
        Returns:
            is_adversarial: (batch_size,) - binary predictions
            errors: (batch_size,) - reconstruction errors
        """
        errors = self.get_reconstruction_error(inputs)
        
        if threshold is None:
            threshold = np.percentile(errors, percentile)
        
        is_adversarial = errors > threshold
        
        return is_adversarial, errors, threshold
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'mask_ratio': self.mask_ratio
        }


def build_dede_model(input_dim,
                     latent_dim=64,
                     encoder_hidden_dims=[256, 128],
                     decoder_hidden_dims=[128, 256],
                     mask_ratio=0.5,
                     dropout=0.2,
                     learning_rate=0.001):
    """
    Build and compile DeDe-Adapted model
    
    Args:
        input_dim: number of input features
        latent_dim: dimension of latent representation
        encoder_hidden_dims: hidden layer sizes for encoder
        decoder_hidden_dims: hidden layer sizes for decoder
        mask_ratio: ratio of features to mask during training
        dropout: dropout rate
        learning_rate: learning rate for Adam optimizer
    
    Returns:
        model: compiled DeDe-Adapted model
    """
    model = DeDeAdapted(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        mask_ratio=mask_ratio,
        dropout=dropout
    )
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['mse']
    )
    
    return model
