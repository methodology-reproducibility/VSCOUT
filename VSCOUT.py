import numpy as np
import tensorflow as tf
from keras import layers, Model, callbacks
import ruptures as rpt
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.kde import KDE
from scipy.stats import kurtosis, skew, chi2, entropy

class T2Detector:
    """Hotelling's T-squared multivariate statistical control detector."""
    def __init__(self, alpha=0.005):
        self.alpha = alpha
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.threshold_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        if np.ndim(self.cov_) == 0:
            self.cov_ = np.array([[self.cov_]])
        self.inv_cov_ = np.linalg.pinv(self.cov_)
        df = X.shape[1]
        self.threshold_ = chi2.ppf(1 - self.alpha, df=df)
        return self

    def predict(self, X):
        diff = X - self.mean_
        md2 = np.sum(diff @ self.inv_cov_ * diff, axis=1)
        return (md2 > self.threshold_).astype(int)

class BoxplotOutlier1D:
    """1D boxplot outlier detector (IQR method)."""
    def fit(self, X, y=None):
        X = np.asarray(X)
        x = X[:, 0] if (X.ndim == 2 and X.shape[1] == 1) else X.ravel()
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        self.low_ = q1 - 1.5 * iqr
        self.high_ = q3 + 1.5 * iqr
        return self

    def predict(self, X):
        X = np.asarray(X)
        x = X[:, 0] if (X.ndim == 2 and X.shape[1] == 1) else X.ravel()
        out = (x < self.low_) | (x > self.high_)
        return out.astype(int)

class VSCOUT:
    """
    Variational Self-Correcting Outlier Uncovering Technique (VSCOUT).
    Combines ARD-VAE, PELT changepoint detection, and ensemble methods for anomaly detection.
    """
    def __init__(
        self,
        encoder_neurons=(64,), decoder_neurons=(64,), latent_dim=32,
        learning_rate=1e-4, alpha=0.005, penalty=40, kl_threshold=1,
        flag_rule="any", n_jobs=1, kurtosis_threshold=5.0
    ):
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.penalty = penalty
        self.kl_threshold = kl_threshold
        self.flag_rule = flag_rule
        self.n_jobs = n_jobs
        self.kurtosis_threshold = kurtosis_threshold

        self.vae = None
        self.encoder = None
        self.decoder = None
        self.latent_encoder = None
        self.orig_dim = None
        self._X_train = None
        self.z_inliers = None
        self.change_points = None
        self.inlier_mask = None
        self.kl_divs = None
        self.relevant_latents = None
        self.ere1_threshold = None
        self.latent_mean_inlier = None
        self.latent_cov_inlier = None
        self.t2_threshold = None
        self.base_detectors = None

    @staticmethod
    def suggest_flag_rule(X):
        """Analyzes data distribution to suggest 'any' or 'majority' voting for the ensemble."""
        X = np.nan_to_num(np.asarray(X))
        avg_kurt = np.mean(np.abs(kurtosis(X, axis=0, fisher=False)))
        avg_skew = np.mean(np.abs(skew(X, axis=0)))
        n_samples, n_features = X.shape
        entropies = [entropy(np.histogram(X[:,i], bins=10)[0] + 1e-8) for i in range(n_features)]
        avg_entropy = np.mean(entropies)
        
        multimodal = False
        try:
            from scipy.signal import find_peaks
            for i in range(n_features):
                counts, _ = np.histogram(X[:,i], bins=20)
                peaks, props = find_peaks(counts, height=np.max(counts)*0.4, prominence=np.max(counts)*0.3, distance=2)
                if len(peaks) > 1:
                    sorted_h = np.sort(props['peak_heights'])
                    if sorted_h[-2] > 0.5 * sorted_h[-1]:
                        multimodal = True
                        break
        except Exception: pass

        if avg_kurt > 5 or avg_skew > 5 or avg_entropy > 2.0:
            return "any"
        return "majority"

    def _build_model(self, X, force_structure=False):
        if self.flag_rule is None:
            self.flag_rule = self.suggest_flag_rule(X)

        enc_in = layers.Input(shape=(X.shape[1],), name="enc_in")
        h = enc_in
        for neurons in self.encoder_neurons:
            h = layers.Dense(neurons, activation="relu", kernel_initializer="he_normal")(h)
            h = layers.BatchNormalization()(h)
            h = layers.Dropout(0.2)(h)
        
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(h)

        def sample(args):
            m, lv = args
            return m + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(m))
        
        z = layers.Lambda(sample, name="z")([z_mean, z_log_var])
        self.encoder = Model(enc_in, [z_mean, z_log_var, z], name="encoder")

        lat_in = layers.Input(shape=(self.latent_dim,), name="z_sampling")
        h2 = lat_in
        for neurons in self.decoder_neurons:
            h2 = layers.Dense(neurons, activation="relu", kernel_initializer="he_normal")(h2)
            h2 = layers.BatchNormalization()(h2)
            h2 = layers.Dropout(0.2)(h2)
        
        dec_out = layers.Dense(X.shape[1], activation="linear")(h2)
        self.decoder = Model(lat_in, dec_out, name="decoder")

        class ARDVAE(Model):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder, self.decoder = encoder, decoder
                self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
                self.recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

            def call(self, inputs, training=False):
                _, _, z = self.encoder(inputs, training=training)
                return self.decoder(z, training=training)

            @property
            def metrics(self):
                return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

            def train_step(self, data):
                x = data[0] if isinstance(data, tuple) else data
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(x, training=True)
                    x_recon = self.decoder(z, training=True)
                    recon_loss = tf.reduce_sum(tf.square(x - x_recon), axis=1)
                    kl_loss = 0.5 * tf.reduce_sum(-1 - z_log_var + tf.square(z_mean) + tf.exp(z_log_var), axis=1)
                    total_loss = tf.reduce_mean(recon_loss + kl_loss)
                
                grads = tape.gradient(total_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                self.total_loss_tracker.update_state(total_loss)
                self.recon_loss_tracker.update_state(tf.reduce_mean(recon_loss))
                self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss))
                return {m.name: m.result() for m in self.metrics}

            def compute_latent_statistics(self, z_means, inlier_mask):
                z_in = z_means[inlier_mask]
                self.latent_mean_normal = np.median(z_in, axis=0)
                self.latent_var_normal = (np.median(np.abs(z_in - self.latent_mean_normal), axis=0) + 1e-6) ** 2

        self.vae = ARDVAE(self.encoder, self.decoder)
        self.latent_encoder = Model(self.encoder.input, self.encoder.get_layer("z_mean").output)

    def _fit_ensemble(self, z_mean_relevant):
        z = np.asarray(z_mean_relevant)
        detectors = []
        if z.ndim == 2 and z.shape[1] == 1:
            detectors.append(BoxplotOutlier1D())
        
        detectors.extend([
            IForest(), LOF(), HBOS(), ECOD(), KNN(n_neighbors=2), KDE(),
            T2Detector(alpha=self.alpha)
        ])
        for clf in detectors:
            clf.fit(z_mean_relevant)
        self.base_detectors = detectors

    def _ensemble_predict(self, z_mean_relevant, rule='majority'):
        base_preds = np.array([clf.predict(z_mean_relevant) for clf in self.base_detectors])
        votes = np.sum(base_preds, axis=0)
        if rule == 'majority':
            return votes >= (len(self.base_detectors) // 2 + 1)
        return np.any(base_preds, axis=0)

    def fit(self, X_train, epochs=30, batch_size=32, verbose=1):
        """Fits the model using a two-stage approach to filter outliers before final training."""
        self.orig_dim = X_train.shape[1]
        self._X_train = X_train.copy()
        self._build_model(X_train)
        
        early_stop = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        
        # Stage 1: Initial training to identify noise/changepoints
        self.vae.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stop])
        
        initial_enc_w = self.encoder.get_weights()
        initial_dec_w = self.decoder.get_weights()
        
        z_mean, z_log_var, _ = self.encoder.predict(X_train, batch_size=batch_size)
        kl_divs = np.mean(0.5 * (-1 - z_log_var + np.square(z_mean) + np.exp(z_log_var)), axis=0)
        relevant_indices = np.where(kl_divs > self.kl_threshold)[0]
        self.relevant_latents = relevant_indices if len(relevant_indices) > 0 else np.array([0])
        
        z_relevant = z_mean[:, self.relevant_latents]
        l2_norm = np.linalg.norm(z_relevant, axis=1)
        self.change_points = rpt.Pelt(model="rbf").fit(l2_norm.reshape(-1, 1)).predict(pen=self.penalty)
        
        cp_mask = np.zeros(len(z_mean), dtype=bool)
        if len(self.change_points) > 0:
            cp_mask[self.change_points[0]:] = True
            
        self._fit_ensemble(z_relevant)
        suod_mask = self._ensemble_predict(z_relevant, rule=self.flag_rule)
        
        self.inlier_mask = ~(np.logical_or(cp_mask, suod_mask))
        X_inliers = X_train[self.inlier_mask]
        
        # Stage 2: Refined training on inliers
        self._build_model(X_inliers, force_structure=True)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        self.vae.fit(X_inliers, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[early_stop])
        
        z_in_mean, _, _ = self.encoder.predict(X_inliers, batch_size=batch_size)
        self.z_inliers = z_in_mean[:, self.relevant_latents]
        self.latent_mean_inlier = np.mean(self.z_inliers, axis=0)
        
        cov = np.cov(self.z_inliers, rowvar=False)
        self.latent_cov_inlier = np.array([[cov]]) if np.ndim(cov) == 0 else cov
        self.t2_threshold = chi2.ppf(1 - self.alpha, df=len(self.relevant_latents))
        self._fit_ensemble(self.z_inliers)

    def is_outlier(self, data, batch_size=32):
        """Returns outlier masks based on ensemble, T2, changepoints, and reconstruction error."""
        z_mean, z_log_var, z_sample = self.encoder.predict(data, batch_size=batch_size)
        z_rel = z_mean[:, self.relevant_latents]

        cp_mask = np.zeros(len(z_rel), dtype=bool)
        if self.change_points and len(self.change_points) > 0:
            cp_mask[self.change_points[0]:] = True

        suod_mask = self._ensemble_predict(z_rel, rule=self.flag_rule)
        
        inv_cov = np.linalg.pinv(self.latent_cov_inlier)
        diff = z_rel - self.latent_mean_inlier
        mahal_sq = np.sum(diff @ inv_cov * diff, axis=1)
        t2_mask = mahal_sq > self.t2_threshold

        x_recon = self.decoder.predict(z_sample, batch_size=batch_size)
        recon_errors = np.sum((data - x_recon) ** 2, axis=1)
        # Threshold is the 95th percentile of errors seen during training inliers
        recon_threshold = np.percentile(np.sum((self._X_train[self.inlier_mask] - 
                                         self.decoder.predict(self.encoder.predict(self._X_train[self.inlier_mask])[2]))**2, axis=1), 95)
        recon_mask = recon_errors > recon_threshold

        votes = np.stack([suod_mask, t2_mask, cp_mask, recon_mask], axis=1)
        final_outlier = np.sum(votes, axis=1) >= 2
        
        return final_outlier, cp_mask, suod_mask, t2_mask, recon_mask, mahal_sq, self.t2_threshold, recon_threshold

    def plot_control_chart(self, data, batch_size=32, phase="Phase I", show_plot=True):
            """
            Generates statistical control charts for Phase I (cleaning/baseline) or Phase II (real-time monitoring).
            """
            x_vals = np.arange(len(data))

            if phase == "Phase I":
                is_out, cp, suod, t2, rec, m2, thresh, _ = self.is_outlier(data, batch_size)
                
                if show_plot:
                    plt.figure(figsize=(16, 7))
                    # Plot continuous distance line
                    plt.plot(x_vals, m2, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Mahalanobis²')
                    
                    # Markers for specific detection sources
                    plt.plot(x_vals[is_out], m2[is_out], 'ro', markersize=8, label='Out-of-Control')
                    plt.plot(x_vals[cp], m2[cp], 'D', markersize=4, linestyle='None', label='ChangePoint Flag')
                    plt.plot(x_vals[suod], m2[suod], 'kx', markersize=4, linestyle='None', label='Ensemble Flag')
                    plt.plot(x_vals[t2], m2[t2], 'g+', markersize=4, linestyle='None', label='T² Flag')
                    plt.plot(x_vals[rec], m2[rec], 'y^', markersize=4, linestyle='None', label='Reconstruction Flag')
                    
                    plt.axhline(y=thresh, color='green', linestyle='--', linewidth=2, label='T² Threshold')
                    
                    if self.change_points is not None:
                        for p in self.change_points:
                            plt.axvline(p, color='magenta', linestyle=':', alpha=0.6, label='Changepoint' if p == self.change_points[0] else "")
                    
                    plt.title('VSCOUT: Phase I Control Chart')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Mahalanobis² Value')
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                return is_out

            elif phase == "Phase II":
                    z_m, _, _ = self.encoder.predict(data, batch_size=batch_size)
                    z_rel = z_m[:, self.relevant_latents]
                    diff = z_rel - self.latent_mean_inlier
                    T2 = np.sum(diff @ np.linalg.pinv(self.latent_cov_inlier) * diff, axis=1)
                    out_mask = T2 > self.t2_threshold
                    
                    if show_plot:
                        plt.figure(figsize=(16, 7))
                        
                        # Consistent gray path for distances
                        plt.plot(x_vals, T2, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Mahalanobis² (T²)')
                        
                        # Consistent Out-of-control marker (Red Circle)
                        plt.plot(x_vals[out_mask], T2[out_mask], 'ro', markersize=8, label='Out-of-Control')
                        
                        # Consistent green threshold line
                        plt.axhline(y=self.t2_threshold, color='green', linestyle='--', linewidth=2, label='T² Threshold')
                        
                        plt.title('VSCOUT: Phase II Latent Space Monitoring')
                        plt.xlabel('Sample Index')
                        plt.ylabel('Hotelling T² Distance')
                        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    return out_mask
                
            else:
                raise ValueError("phase must be 'Phase I' or 'Phase II'")

    @property
    def latent_mean(self): return self.latent_mean_inlier

    @property
    def latent_cov(self): return self.latent_cov_inlier







# --- EXAMPLE USAGE ---
# import numpy as np
# !pip install ruptures pyod

# 1. Generate dummy data: 1000 samples of normal data with 10 features
# X_train = np.random.normal(0, 1, (1000, 200))
# X_train[900:] += 2  # Adding anomalies

# 2. Initialize and Fit VSCOUT (Phase I)
# This will automatically handle changepoint detection and ensemble voting
# detector = VSCOUT()
# detector.fit(X_train)

# 3. Saving Predictions Training Data (Phase I)
# y_pred = detector.plot_control_chart(X_train, phase="Phase I", show_plot=False)
# check performance
# from sklearn.metrics import recall_score, precision_score, ConfusionMatrixDisplay, confusion_matrix
# y_true = np.zeros(len(X_train))
# y_true[900:] = 1  # True anomalies    
# print("Recall (Detection Rate):", recall_score(y_true, y_pred))
# print("Precision:", precision_score(y_true, y_pred))
# FalsePositivesRate = confusion_matrix(y_true, y_pred)[0][1] / sum(confusion_matrix(y_true, y_pred)[0])
# print("False Positives Rate:", FalsePositivesRate)



# 4. Visualize Training (Phase I) Results
# detector.plot_control_chart(X_train, phase="Phase I", show_plot=True)

# 5. Simulate new incoming data for Phase II monitoring
# X_new = np.random.normal(0, 1, (300, 200))      
# X_new[250:] += 2  # Adding anomalies in new data    
# detector.plot_control_chart(X_new, phase="Phase II", show_plot=True)

