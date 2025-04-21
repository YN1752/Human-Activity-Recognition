import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(CNNLSTM, self).__init__()

        # Pretrained CNN (ResNet18) for feature extraction
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        self.feature_dim = resnet.fc.in_features  # 512 features from ResNet18

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,  # 512 features from CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape  # (B, T, C, H, W)
        
        # Extract CNN features for each frame
        cnn_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]  # Extract frame (B, C, H, W)
            features = self.cnn(frame)  # CNN output (B, 512, 1, 1)
            features = features.view(batch_size, -1)  # Flatten to (B, 512)
            cnn_features.append(features)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # (B, T, 512)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # (B, T, hidden_size)

        # Take the last LSTM output
        last_output = lstm_out[:, -1, :]  # (B, hidden_size)

        # Final classification
        out = self.fc(last_output)  # (B, num_classes)
        
        return out