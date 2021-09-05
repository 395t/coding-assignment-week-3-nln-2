from PIL import Image

import csv
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SLT10Dataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        image_to_tensor = transforms.ToTensor()
        
        self.transforms = transform
        self.imgs_labels = list()
        # # Open the labels file and read it line by line into a list
        # labels = list()
        # with open(dataset_path + '/labels.csv', mode ='r') as f:
        #     labs = csv.reader(f)
        #     for l in labs:
        #         labels.append(l)
        # # Remove the column labels from the labels
        # labels = labels[1:]
        # # Create tuples of the label numbers and the images
        # for l in labels:
        #     filename = l[0]
        #     label_num = LABEL_MAPPING[l[1]]
        #     with Image.open(dataset_path + '/' + filename, 'r') as img:
        #         img_tensor = image_to_tensor(img)
        #         self.imgs_labels.append((img_tensor, label_num))

        for i in os.listdir(dataset_path):
            images = os.listdir(dataset_path + '/' + i)
            label_num = int(i) - 1
            for image in images:
                with Image.open(dataset_path + '/' + i + '/' + image, 'r') as img:
                    self.imgs_labels.append((image_to_tensor(img), label_num))

    def __len__(self):
        return len(self.imgs_labels)

    def __getitem__(self, idx):
        if self.transforms != None:
            return (self.transforms(self.imgs_labels[idx][0]), self.imgs_labels[idx][1])
        return self.imgs_labels[idx]


def load_data(dataset_path, transform, num_workers=2, batch_size=128):
    dataset = SLT10Dataset(dataset_path, transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()