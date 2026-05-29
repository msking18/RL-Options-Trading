import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from backend.train.ppo_config import STATIC_PER_SYMBOL_FEATURES

class Trading1DCNN(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Trading Environment that uses 1D Convolutions
    over the lookback window for each symbol slot.
    """
    def __init__(self, observation_space, features_dim=512, lookback_window=30, total_slots=6):
        super().__init__(observation_space, features_dim)
        
        self.lookback_window = lookback_window
        self.total_slots = total_slots
        self.ohlcv_size = lookback_window * 8 # Expanded to 8 channels
        self.static_per_symbol_size = STATIC_PER_SYMBOL_FEATURES 
        self.per_symbol_segment_size = self.ohlcv_size + self.static_per_symbol_size
        
        # Validation: Ensure the observation space matches our expectations
        expected_min_size = (self.per_symbol_segment_size * total_slots) + 2
        actual_size = observation_space.shape[0]
        if actual_size < expected_min_size:
            raise ValueError(
                f"Observation space size {actual_size} is too small. "
                f"Expected at least {expected_min_size} for {total_slots} slots "
                f"with lookback {lookback_window} (per-slot size: {self.per_symbol_segment_size})."
            )
        
        # 1D CNN for OHLCV + Indicators data
        # input shape: [batch, 8, 30] (channels, length)
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        
        # Compute the size after CNN
        # 30 -> 15 (maxpool 1) -> 7 (maxpool 2)
        # result: 64 * 7 = 448
        cnn_output_size = 64 * 7
        
        # Combined size per symbol (CNN features + static Greeks/tech)
        self.combined_per_symbol_size = cnn_output_size + self.static_per_symbol_size
        
        # Global features size
        self.global_features_size = observation_space.shape[0] - (self.per_symbol_segment_size * total_slots)
        
        # Final output MLP
        total_intermediate_size = (self.combined_per_symbol_size * total_slots) + self.global_features_size
        self.linear = nn.Sequential(
            nn.Linear(total_intermediate_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        # 1. Extract symbol segments
        total_slots_size = self.total_slots * self.per_symbol_segment_size
        obs_slice = observations[:, 2 : 2 + total_slots_size]
        
        # Guard against mismatch between Environment and Feature Extractor
        if obs_slice.shape[1] != total_slots_size:
            actual_size = obs_slice.shape[1]
            raise RuntimeError(
                f"Feature Extractor Mismatch: Expected total_slots_size {total_slots_size} "
                f"({self.total_slots} slots * {self.per_symbol_segment_size} per symbol), "
                f"but got slice of size {actual_size}."
            )

        symbol_data = obs_slice.view(
            batch_size, self.total_slots, self.per_symbol_segment_size
        ) 
        
        # 2. Split into OHLCV+Indicators (history) and Static features
        ohlcv_data = symbol_data[:, :, :self.ohlcv_size] 
        static_data = symbol_data[:, :, self.ohlcv_size:] 
        
        # 3. Vectorized CNN Pass
        # Reshape for CNN: [B*slots, length, 8] -> [B*slots, 8, length]
        ohlcv_reshaped = ohlcv_data.reshape(batch_size * self.total_slots, self.lookback_window, 8).permute(0, 2, 1)
        cnn_feats_all = self.cnn(ohlcv_reshaped) 
        
        # 4. Reconstruct Combined Features
        cnn_feats = cnn_feats_all.view(batch_size, self.total_slots, -1)
        symbol_combined = th.cat([cnn_feats, static_data], dim=2) 
        
        # Flatten Slots into feature dimension
        symbol_flat = symbol_combined.view(batch_size, -1)
        
        # 5. Extract global features
        global_portfolio = observations[:, :2]
        external_signals = observations[:, 2 + total_slots_size:]
        global_feats = th.cat([global_portfolio, external_signals], dim=1)
        
        # 6. Final Concatenation and Linear Layers
        all_features = th.cat([symbol_flat, global_feats], dim=1)
        return self.linear(all_features)
