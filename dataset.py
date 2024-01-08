import glob
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image


class FVCDataset(Dataset):
    def __init__(self):
        self.imgs_path = "../datasets/FVC2002_DB1_B_split/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            # in our case numbers are class names, so they can be used as class_id
            class_id = int(class_path.split("/")[-1])
            for img_path in glob.glob(class_path + "/*.tif"):
                self.data.append([img_path, class_id])
        self.img_dim = (784, 784)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img = Image.open(img_path)
        img = img.resize(self.img_dim, Image.LANCZOS)
        img_tensor = ToTensor()(img)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


if __name__ == "__main__":
    dataset = FVCDataset()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # load images to dict
    fingerprints = {}
    for batch in data_loader:
        for img, label in zip(batch[0], batch[1]):
            img = img.permute(1, 2, 0)
            label = label.item()
            print(f"Image shape: {img.shape}, label: {label}")
            fingerprints.setdefault(label, []).append(img)

    width = max([len(fingerprints[key]) for key in fingerprints.keys()])
    height = fingerprints.__len__()
    # reduce size to make it more readable
    if height > 4:
        width = 4
        height = 4

    print(f"Width: {width}, height: {height}")

    finger_numbers = list(fingerprints.keys())
    fig = plt.figure(figsize=(width, height))

    for i in range(height):
        for j in range(width):
            fig.add_subplot(height, width, i * width + j + 1)
            plt.gray()
            plt.imshow(fingerprints[finger_numbers[i]][j])
            plt.axis("off")
            plt.title(f"Class: {finger_numbers[i]} - {j}")

    fig.tight_layout()
    plt.show()
