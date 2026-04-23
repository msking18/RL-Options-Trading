import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Trading1DCNN(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Trading Environment that uses 1D Convolutions
    over the lookback window for each symbol slot.
    """
    def __init__(self, observation_space, features_dim=512, lookback_window=30, total_slots=6):
        super().__init__(observation_space, features_dim)
        
        self.lookback_window = lookback_window
        self.total_slots = total_slots
        self.ohlcv_size = lookback_window * 6
        # Matches TradingEnv: (2*5*4) Greeks + (2*2) Prices + 3 Pos + 5 Tech + 3 Temp + 2 Risk = 57
        self.static_per_symbol_size = 57 
        self.per_symbol_segment_size = self.ohlcv_size + self.static_per_symbol_size
        
        # 1D CNN for OHLCV data
        # input shape: [batch, 6, 30] (channels, length)
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
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
        # total_obs = (per_symbol * 10) + global
        self.global_features_size = observation_space.shape[0] - (self.per_symbol_segment_size * total_slots)
        
        # Final output MLP
        total_intermediate_size = (self.combined_per_symbol_size * total_slots) + self.global_features_size
        self.linear = nn.Sequential(
            nn.Linear(total_intermediate_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        # 1. Extract symbol segments
        # observations shape: [B, total_obs_size]
        total_slots_size = self.total_slots * self.per_symbol_segment_size
        symbol_data = observations[:, :total_slots_size].view(
            batch_size, self.total_slots, self.per_symbol_segment_size
        ) # [B, 10, 208]
        
        # 2. Split into OHLCV and Static features
        # ohlcv data is first 180 elements (30 * 6)
        ohlcv_data = symbol_data[:, :, :self.ohlcv_size] # [B, 10, 180]
        static_data = symbol_data[:, :, self.ohlcv_size:] # [B, 10, 28]
        
        # 3. Vectorized CNN Pass
        # Flatten Batch and Slots to process all together
        # Reshape for CNN: [B*10, 6, 30]
        ohlcv_reshaped = ohlcv_data.reshape(batch_size * self.total_slots, self.lookback_window, 6).permute(0, 2, 1)
        cnn_feats_all = self.cnn(ohlcv_reshaped) # [B*10, 448]
        
        # 4. Reconstruct Combined Features
        # Reshape CNN output back to [B, 10, 448]
        cnn_feats = cnn_feats_all.view(batch_size, self.total_slots, -1)
        symbol_combined = th.cat([cnn_feats, static_data], dim=2) # [B, 10, 476]
        
        # Flatten Slots into feature dimension: [B, 4760]
        symbol_flat = symbol_combined.view(batch_size, -1)
        
        # 5. Extract global features
        global_feats = observations[:, total_slots_size:] # [B, G]
        
        # 6. Final Concatenation and Linear Layers
        all_features = th.cat([symbol_flat, global_feats], dim=1)
        return self.linear(all_features)
