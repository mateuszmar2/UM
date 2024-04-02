import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import FVCDataset, get_data_loaders

# metadata
# 100 epochs is too low, 200 is better, maybe more is even better
num_epochs = 200
# smaller batch size is better
batch_size = 2
validation_split = batch_size / 8 if batch_size < 8 else 0.25


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# custom dataset
data_loader, test_data_loader, _, test_dataset_size = get_data_loaders(
    single_class=False,
    batch_size=batch_size,
    validation_split=validation_split,
    img_dim=(300, 300),
    fingerprint_database="1",
    pca_transform=False,
    train_class=101,
    test_class=101,
    shuffle_dataset=False,
)

# check if values are in range [0, 1]
dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.Size of image = batch_size (N), color channel, height, width
        # torch.Size([2, 1, 300, 300])
        self.encoder = nn.Sequential(
            # input_channel = 1 (grayscale)
            # output_channel = 16
            # kernel_size = 4 (4x4 filter)
            # stride = 2 (move 3 pixels at a time)
            # padding = 1 (add 1 pixel of padding to each side)
            nn.Conv2d(1, 16, 6, stride=2, padding=1),  # N, 16, 300, 300
            nn.ReLU(),
            # output of previous layer is input of next layer
            nn.Conv2d(16, 32, 4, stride=3, padding=1),  # N, 32, 149, 149
            nn.ReLU(),
            # if there will be 1 pixel for each channel, then it will learn placement of the pixels
            # instead of learning the features of the fingerprint
            nn.Conv2d(32, 64, 5),  # N, 64 channels, 50/10 pixels per channel
        )

        # N, 64, 50/10, 50/10
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5),  # N, 32, 50, 50
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=3, padding=1),  # N, 16, 148, 148
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 6, stride=2, padding=1),  # N, 1, 300, 300
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# Training
outputs = []
for epoch in range(num_epochs):
    for img, _ in data_loader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch:{epoch+1}, Loss:{loss.item():.5f}")
    outputs.append((epoch, img, recon))

# Plotting the training images and their reconstructions
for k in range(0, num_epochs, int(num_epochs - 1)):
    plt.figure(figsize=(batch_size, 2))
    plt.title("Training Images vs Reconstructed Images (Epoch {})".format(k + 1))
    plt.axis("off")
    plt.gray()
    imgs = outputs[k][1].detach().cpu().numpy()
    recon = outputs[k][2].detach().cpu().numpy()
    for i, item in enumerate(imgs):
        # not more than 8
        if i >= 8:
            break
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 8:
            break
        plt.subplot(2, batch_size, batch_size + i + 1)
        plt.imshow(item[0])

    # plt.savefig(f"results/train_epoch{k}.png", dpi=600)
    plt.show(block=False)

# Testing
outputs = []
with torch.no_grad():
    for img, _ in test_data_loader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

print(f"Epoch:test, Loss:{loss.item():.5f}")
outputs.append((epoch, img, recon))

# Plotting the test images and their reconstructions
plt.figure(figsize=(test_dataset_size, 2))
plt.axis("off")
plt.title("Test Images vs Reconstructed Images test")
plt.gray()
imgs = outputs[0][1].detach().cpu().numpy()
recons = outputs[0][2].detach().cpu().numpy()

for i, item in enumerate(imgs):
    if i >= 8:
        break
    plt.subplot(2, test_dataset_size, i + 1)
    plt.imshow(item[0])

for i, item in enumerate(recons):
    if i >= 8:
        break
    plt.subplot(2, test_dataset_size, test_dataset_size + i + 1)  # 2 + i + 1
    plt.imshow(item[0])
    plt.show(block=False)

# plt.savefig(f"results/test.png", dpi=600)
plt.show()
