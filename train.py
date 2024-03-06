from model import AlexNetModel
import numpy as np
import torch

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os

torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
# define model parameters
NUM_EPOCHS = 20 
BATCH_SIZE = 16
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 10  # 10 classes for CIFAR 10 dataset
# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models/'  # model checkpoints


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


# CIFAR10 dataset 
train_loader, valid_loader = get_train_valid_loader(data_dir = './data',                                      
                                                    batch_size = BATCH_SIZE ,
                                                    augment = False,                             		     
                                                    random_seed = 1)

test_loader = get_test_loader(data_dir = './data',
                              batch_size = BATCH_SIZE)

# setting the seed
seed = torch.initial_seed()
print(f"Seed: {seed}")




tbwriter = SummaryWriter(log_dir=LOG_DIR)

#setting the model
alexnet = AlexNetModel(num_classes=NUM_CLASSES).to(device)
alexnet = nn.parallel.DataParallel(alexnet)

#setting the optimiser
optimiser = torch.optim.SGD(params=alexnet.parameters(), lr=LR_INIT, momentum=MOMENTUM, weight_decay=LR_DECAY)

#setting the loss function
criterion = nn.CrossEntropyLoss()

total_steps = 1
for epoch in range(NUM_EPOCHS):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'alexnet_{epoch+1}.pth')
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path)
        alexnet.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        seed = state['seed']
        total_steps = state['total_steps']
    else:
        print(f"Checkpoint {checkpoint_path} not found")
        print("Training at {} epoch".format(epoch+1))
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = alexnet(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            if (total_steps) % 3 == 0:
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    accuracy = torch.sum(predicted == labels).item() / labels.size(0)
                    print (f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
                    tbwriter.add_scalar('Loss/train', loss.item(), total_steps)
                    tbwriter.add_scalar('Accuracy/train', accuracy, total_steps)
                
    total_steps += 1
        
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'alexnet_{epoch+1}.pth')
        
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimiser': optimiser.state_dict(),
        'model': alexnet.state_dict(),
        'seed': seed
    }
    torch.save(state, checkpoint_path)
    
def calculate_accuracy(test_loader):
    """
    Calculate the accuracy of the model on the test set
    
    Args:
        test_loader: DataLoader object
    
    Returns:
        accuracy: float
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_20.pth')
            state = torch.load(checkpoint_path)
            alexnet.load_state_dict(state['model'])
            outputs = alexnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        accuracy = 100 * correct / total
        
    return accuracy

accuracy = calculate_accuracy(test_loader)

print('Accuracy of the network on the {} test images: {} %'.format(10000, accuracy)) 


def calculate_topk_error(outputs, targets, topk=(1, 5)):
    """Calculates top-k error rates

    Args:
        outputs (torch.Tensor): Model prediction outputs (logits).
        targets (torch.Tensor): Ground truth labels.
        topk (tuple, optional): Values of k for which to calculate errors. 
                                Defaults to (1, 5).

    Returns:
        list: Top-k error rates.
    """

    with torch.no_grad():
        maxk = max(topk)  # Get the maximum 'k'

        batch_size = targets.size(0)

        # Find indices of top-k predictions
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # Transpose

        # Expand target to compare with top-k predictions 
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        error_rates = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            error_rates.append(100.0 - (correct_k.mul_(100.0 / batch_size)))
        return error_rates

for data, target in test_loader:
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_20.pth')
    state = torch.load(checkpoint_path)
    alexnet.load_state_dict(state['model'])
    output = alexnet(data)
    top1_error, top5_error = calculate_topk_error(output, target)

    print(f"Top-1 Error: {top1_error.item():.2f}%")
    print(f"Top-5 Error: {top5_error.item():.2f}%")