from VIT import VIT
import torch
from dataset import AnimalImageDataset
from Train import Trainer
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader


num_classes = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VIT(imgsize=224, patchsize=16, embed_dim=768, num_classes=num_classes, layer_num=12, num_heads=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

summary(model, input_size=(3, 224, 224),batch_size=2)

# 数据集根目录和转换
root_dir = "D:/datasets/ANIMALS/animals/animals"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# 创建自定义数据集实例

train_dataset = AnimalImageDataset(root_dir, transform, num_classes)
trainer = Trainer(model, optimizer, criterion, device)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
trainer.train(train_loader, epochs=10)
