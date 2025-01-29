import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import gc


class HoGDataset(Dataset):
    def __init__(self, features_dir, masks_dir):
        self.features_dir = Path(features_dir)
        self.masks_dir = Path(masks_dir)
        self.feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.txt.npy')])
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        
        # Load the feature and mask
        feature = np.load(self.features_dir / feature_file).astype(np.float16)
        mask = np.load(self.masks_dir / feature_file).astype(np.float16)
        
        # Print shapes of the feature and mask to debug
        print(f"Loaded feature shape: {feature.shape}")
        print(f"Loaded mask shape: {mask.shape}")
        
        return torch.tensor(feature, dtype=torch.float16), torch.tensor(mask, dtype=torch.float16)


class LaneTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=128, nhead=4, num_layers=3, dim_feedforward=512):
        super().__init__()

        # Reduced model dimensions
        self.input_dim = input_dim
        self.patch_embedding = nn.Linear(input_dim, d_model)

        # Simplified positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 58*89, d_model) * 0.02)

        # Reduced transformer size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Ensure input is in float32 (single precision)
        x = x.float()

        # Process in smaller chunks if needed
        x = x.view(batch_size, 58*89, -1)  # Flatten the spatial dimensions (58*89)
        x = self.patch_embedding(x)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Process through transformer
        x = self.transformer_encoder(x)
        x = self.output_proj(x)

        # Reshape back to (batch_size, 58, 89, 9)
        x = x.view(batch_size, 58, 89, 9)

        return x


def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (features, masks) in enumerate(train_loader):
            # Move to device and ensure both features and masks are in float32
            features = features.to(device).float()  # Convert to float32
            masks = masks.to(device).float()  # Convert to float32

            # Print the data types of the features and masks
            print(f"Features dtype: {features.dtype}, Masks dtype: {masks.dtype}")

            optimizer.zero_grad()

            # Forward pass
            output = model(features)

            # Print the data type of the output
            print(f"Output dtype: {output.dtype}")

            loss = criterion(output, masks)

            # Print the data type of the loss
            print(f"Loss dtype: {loss.dtype}")

            # Backward pass
            loss.backward()
            optimizer.step()

            # Free up memory
            del output
            gc.collect()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')


def main():
    device = torch.device('cpu')  # Ensure CPU is being used
    dataset = HoGDataset('./hog_features', './hog_masks')
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    model = LaneTransformer().to(device)  # Model will be in float32 by default
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    num_epochs = 50
    
    try:
        train_model(model, train_loader, optimizer, criterion, num_epochs, device)
        torch.save(model.state_dict(), 'lane_transformer_light.pth')
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        gc.collect()


if __name__ == '__main__':
    main()
