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
        # print(file_list)
        self.data = []
        for class_path in file_list:
            # in our case numbers are class names, so they can be used as class_id
            class_id = int(class_path.split("/")[-1])
            for img_path in glob.glob(class_path + "/*.tif"):
                self.data.append([img_path, class_id])
        # print(self.data)
        self.img_dim = (784, 784)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img = Image.open(img_path)
        img = img.resize(self.img_dim, Image.LANCZOS)
        # img = rasterio.open(img_path)
        # img = img.read(1)
        # plt.imshow(img, cmap="gray")
        # plt.show()
        img_tensor = ToTensor()(img)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
        # return img_tensor.float(), class_id.float()


if __name__ == "__main__":
    dataset = FVCDataset()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for imgs, labels in data_loader:
        print("Batch of images has shape: ", imgs.shape)
        print("Batch of labels has shape: ", labels.shape)

    images, labels = next(iter(data_loader))
    image = images[0]
    print(image.shape)
    image = image.permute(1, 2, 0)
    print(image.shape)
    plt.imshow(image.numpy())
    plt.gray()
    plt.show()
