import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image


class FVCDataset(Dataset):
    def __init__(
        self,
        root="FVC2002_dataset/",
        img_dim=(784, 784),
        single_class=False,
        fingerprint_database="1",
        pca_transform=False,
    ):
        self.imgs_path = root + fingerprint_database
        self.img_dim = img_dim
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []

        self.pca_transform = pca_transform

        if pca_transform:
            self.pca_components = 100
            print(
                f"PCA transform to {self.pca_components} principal components enabled"
            )

        if single_class:
            file_list = [random.choice(file_list)]

        for class_path in file_list:
            # in our case numbers are class names, so they can be used as class_id
            class_id = int(class_path.split("/")[-1])
            for img_path in glob.glob(class_path + "/*.tif"):
                self.data.append([img_path, class_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img = Image.open(img_path)
        img = img.resize(self.img_dim, Image.LANCZOS)
        if self.pca_transform:
            img = pca_transform(img, self.pca_components)

        img_tensor = ToTensor()(img)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


def pca_transform(img, k):
    import numpy as np
    from sklearn.decomposition import PCA, IncrementalPCA

    img_array = np.array(img)

    pca = PCA()
    pca.fit(img)
    ipca = IncrementalPCA(n_components=k)
    img_recon = ipca.inverse_transform(ipca.fit_transform(img_array))
    img_recon = img_recon.astype(img_recon.dtype)
    # Apply min-max scaling to bring the values in the [0, 1] range
    min_val = np.min(img_recon)
    max_val = np.max(img_recon)
    img_recon = (img_recon - min_val) / (max_val - min_val)
    img_recon = Image.fromarray(img_recon)
    return img_recon


def get_data_loaders(
    img_dim=(784, 784),
    fingerprint_database="1",
    single_class=False,
    validation_split=0.25,
    shuffle_dataset=True,
    batch_size=2,
    pca_transform=False,
    train_class=None,
    test_class=None,
):
    dataset = FVCDataset(
        single_class=single_class,
        img_dim=img_dim,
        fingerprint_database=fingerprint_database,
        pca_transform=pca_transform,
    )
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    classes = list(set([dataset[i][1].item() for i in range(dataset_size)]))
    print(f"Dataset size: {dataset_size}")
    print(f"Dataset classes: {classes}")
    img_size = dataset[0][0].size()
    print(f"Image size: {img_size}")

    # in case of single class whole dataset contains only one class
    if single_class:
        if shuffle_dataset:
            random.shuffle(indices)
        split = int(validation_split * dataset_size)
        train_indices, test_indices = indices[split:], indices[:split]

    else:
        if not train_class:
            train_class = random.choice(classes)
            if train_class not in classes:
                raise Exception(
                    f"Train class {train_class} not in dataset classes, available classes: {classes}"
                )
        if not test_class:
            test_class = random.choice(classes)
            if test_class not in classes:
                raise Exception(
                    f"Test class {test_class} not in dataset classes, available classes: {classes}"
                )

        for i in range(dataset_size):
            print(f"indice:{i}: {dataset[i][1].item()}")

        print(f"train class: {train_class}, test class: {test_class}")
        train_indices = [
            i for i in range(dataset_size) if dataset[i][1].item() == train_class
        ]
        if train_class == test_class:
            indices = train_indices
            split = int(validation_split * len(indices))
            if shuffle_dataset:
                random.shuffle(indices)
            train_indices = indices[split:]
            test_indices = indices[:split]
        else:
            test_indices = [
                i for i in range(dataset_size) if dataset[i][1].item() == test_class
            ]
            test_split = int(validation_split * len(test_indices))
            train_split = int(validation_split * len(train_indices))
            if shuffle_dataset:
                random.shuffle(train_indices)
                random.shuffle(test_indices)
            train_indices = train_indices[test_split:]
            test_indices = test_indices[:test_split]

    train_dataset_size = len(train_indices)
    test_dataset_size = len(test_indices)
    print(f"Train idices: {train_indices}, test indices: {test_indices}")
    print(
        f"Train dataset size: {train_dataset_size}, test dataset size: {test_dataset_size}"
    )
    print(f"Used dataset size: {train_dataset_size + test_dataset_size}")

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader, train_dataset_size, test_dataset_size


if __name__ == "__main__":
    train_loader, test_loader, _, _ = get_data_loaders(
        single_class=False, test_class=101, shuffle_dataset=True
    )

    for loader in [train_loader, test_loader]:
        # load images to dict
        fingerprints = {}
        for batch in loader:
            for img, label in zip(batch[0], batch[1]):
                img = img.permute(1, 2, 0)
                label = label.item()
                fingerprints.setdefault(label, []).append(img)

        width = max([len(fingerprints[key]) for key in fingerprints.keys()])
        height = fingerprints.__len__()
        # reduce size to make it more readable
        if height > 4:
            width = 8
            height = 4

        finger_numbers = list(fingerprints.keys())
        fig = plt.figure(figsize=(width, height))

        for i in range(height):
            for j in range(width):
                fig.add_subplot(height, width, i * width + j + 1)
                plt.gray()
                plt.imshow(fingerprints[finger_numbers[i]][j])
                plt.axis("off")
                plt.title(f"Class: {finger_numbers[i]} - {j + 1}")

        fig.tight_layout()
        suptitle = "Train" if loader == train_loader else "Test"
        fig.suptitle(f"{suptitle} dataset")
        plt.show(block=False)

plt.show()
