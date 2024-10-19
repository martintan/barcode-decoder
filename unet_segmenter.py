import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(1, 8)
        self.enc2 = self.conv_block(8, 8)
        self.enc3 = self.conv_block(8, 8)
        self.enc4 = self.conv_block(8, 8)

        # Decoder (upsampling)
        self.dec1 = self.upconv_block(8, 8)
        self.dec2 = self.upconv_block(8, 8)
        self.dec3 = self.upconv_block(8, 8)
        self.dec4 = nn.Conv2d(8, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections
        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        output = self.dec4(torch.cat([d3, e1], dim=1))

        return torch.sigmoid(output)


def train_unet(images, masks, num_epochs=100, batch_size=100, learning_rate=0.001):
    # Convert numpy arrays to PyTorch tensors
    images_tensor = (
        torch.from_numpy(images).float().unsqueeze(1)
    )  # Add channel dimension
    masks_tensor = torch.from_numpy(masks).float().unsqueeze(1)  # Add channel dimension

    # Create dataset and dataloader
    dataset = TensorDataset(images_tensor, masks_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_images, batch_masks in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    return model


# Example usage
if __name__ == "__main__":
    # Generate dummy data (replace with your actual data)
    num_images = 1000
    image_size = 128
    images = np.random.rand(num_images, image_size, image_size)
    masks = np.random.randint(0, 2, size=(num_images, image_size, image_size))

    trained_model = train_unet(images, masks)

    # Save the trained model
    torch.save(trained_model.state_dict(), "unet_model.pth")
