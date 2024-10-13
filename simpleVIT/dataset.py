import torch
from torch.utils.data import Dataset

from PIL import Image
import os

animal_list = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar',
    'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow',
    'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly',
    'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat','goldfish'
]
# 创建字典 {animal_name: index}
animal_dict = {animal: index for index, animal in enumerate(animal_list)}


class AnimalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None,num_classes=30):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)[:num_classes]
        self.num_classes=num_classes
        self.class_to_idx = animal_dict
        self.images = self.get_images()
        self.transform = transform
        print(f"root_dir:{root_dir} is initialized!")

    def get_images(self):
        images = []
        for class_name in os.listdir(self.root_dir)[:self.num_classes]:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):  # 确保是文件
                        images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)#[3,224,224]
        return image, label



