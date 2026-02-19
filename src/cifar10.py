import cv2
import torch
import numpy as np
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize
import os
import pickle
from torchvision.transforms import ToTensor
class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(root, "test_batch")]
        self.transform = transform
        self.images = []
        self.labels = []
        self.categories = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        # self.transform = transform
        # print(len(self.images))
        # print(len(self.labels))
                # print(len(data[b'data']))
                # print(len(data[b'labels']))
                # # print(data.keys())
                # print('----------------------')
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item]
        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, mode='RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label



if __name__ == '__main__':
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    dataset = MyDataset(root='./cifar10/cifar-10-batches-py', train=True, transform=transform)
    image, label = dataset.__getitem__(234)
    print(image.shape)
    print(label)
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    cv2.imshow("CIFAR10 Image", cv2.resize(image, (320, 320)))
    cv2.waitKey(0)
# dataset = CIFAR10("./cifar10", train=True, download=True)
# image, label = dataset.__getitem__(2000)
# image.show()
# print(label)

# image = cv2.imread('anh.jpg') # BGR
# print(image.shape)
# image[:, :, 0] = 0
# image[:, :, 2] = 0
# cv2.imshow('Image', image)
# image = torch.from_numpy(image)
# print(image.shape)
# print(image.ndim)
# cv2.waitKey(0)