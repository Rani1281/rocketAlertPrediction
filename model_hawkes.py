import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pymc as pm
import config


class SpatiotemporalHawkes:
    def __init__(self, mu=config.BACKGROUND_RATE, alpha=config.TRIGGER_WEIGHT,
                 beta=config.DECAY_RATE_TIME, sigma=config.DECAY_RATE_SPACE):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.history = None

    def fit(self, df):
        """Loads historical data into the model."""
        self.history = df[['time_sec', 'latitude', 'longitude']].values

    def calculate_intensity(self, target_time, target_lat, target_lon, time_window=86400):
        """
        Calculates the probability of an alert at a specific location and time.
        """
        if self.history is None or len(self.history) == 0:
            return self.mu

        past_events = self.history[(self.history[:, 0] < target_time) &
                                   (self.history[:, 0] >= (target_time - time_window))]

        if len(past_events) == 0:
            return self.mu

        dt = target_time - past_events[:, 0]

        target_coords = np.array([[target_lat, target_lon]])
        event_coords = past_events[:, 1:3]

        distances = cdist(target_coords, event_coords, metric='euclidean').flatten()

        temporal_decay = np.exp(-self.beta * dt)
        spatial_decay = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))

        excitation = self.alpha * np.sum(temporal_decay * spatial_decay)

        return self.mu + excitation

    def learn_parameters(self, df, max_events=300):
        """
        Uses PyMC to perform fast MAP estimation to learn optimal parameters.
        """
        recent_data = df.tail(max_events).copy()

        t = recent_data['time_sec'].values

        # --- THE FIX: Add microscopic jitter so no two events happen at the exact same 0.0 millisecond ---
        t = t + np.random.uniform(0, 0.01, size=len(t))
        t = np.sort(t)  # Re-sort to maintain strict chronological order

        # Normalize time to prevent math overflow
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)

        interarrival_times = np.diff(t)

        # Ensure no exact zeros slip through
        interarrival_times = np.clip(interarrival_times, 1e-5, None)

        with pm.Model() as hawkes_model:
            mu_prior = pm.Exponential('mu', lam=1.0)
            alpha_prior = pm.Uniform('alpha', lower=0.0, upper=1.0)
            beta_prior = pm.Exponential('beta', lam=1.0)

            lambda_t = pm.Deterministic('lambda_t', mu_prior + alpha_prior * np.exp(-beta_prior * interarrival_times))

            obs = pm.Exponential('obs', lam=lambda_t, observed=interarrival_times)

            # --- THE SPEED BOOST: Find Maximum A Posteriori (takes ~1 second) instead of heavy MCMC ---
            map_estimate = pm.find_MAP(progressbar=False)

        # Extract the optimized values
        self.mu = float(map_estimate['mu'])
        self.alpha = float(map_estimate['alpha'])
        self.beta = float(map_estimate['beta'])

        return {
            'mu': self.mu,
            'alpha': self.alpha,
            'beta': self.beta
        }