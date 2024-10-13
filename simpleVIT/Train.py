from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    def accuracy(self,output, target):
        return (output.argmax(dim=1) == target).float().mean()

    def train(self, train_loader, epochs):
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = torch.tensor(target).to( self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
                self.train_loss.append(loss.item())
                self.train_acc.append(self.accuracy(output, target)) # type: ignore
            self.scheduler.step(epoch)

        
            
        
        
        