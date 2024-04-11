import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from dotenv import load_dotenv
import random
from model import AlexNetModel as model
import time
import json
from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter
from utils import load_pretrained_state_dict, load_resume_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter, accuracy
load_dotenv()

def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR100(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
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


def build_model(num_classes,model_arch,model_ema_decay,device) -> [nn.Module, nn.Module]:
    vgg_model = model(num_classes=num_classes)
    vgg_model = vgg_model.to(device)
    
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_vgg_model = AveragedModel(vgg_model, device=device, avg_fn=ema_avg)
    return vgg_model, ema_vgg_model


def loss_def(loss_label_smoothing, device) -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=loss_label_smoothing)
    criterion = criterion.to(device)
    
    return criterion

def optimizer_def(model, learning_rate, momentum, weight_decay) -> torch.optim.SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    return optimizer

def scheduler_def(optimizer, t_0:int, t_mult, eta_min) -> torch.optim.lr_scheduler.MultiStepLR:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t_0, t_mult, eta_min)
    return scheduler

def train(train_loader, vgg_model, ema_vgg_model, criterion, optimizer, device, scaler, epoch, writer, train_print_frequency):
    batches = len(train_loader)
    
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    accuracy_train = AverageMeter("Accuracy", ":6.2f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")
    
    vgg_model.train()
    
    # Get the initialization training time
    end = time.time()
    
    total = 0
    correct = 0
    
    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)
        
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = vgg_model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        ema_vgg_model.update_parameters(vgg_model)
        
        top1, top5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        acc1.update(top1[0], batch_size)
        acc5.update(top5[0], batch_size)
        
        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_index % train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index)
            progress.display(batch_index)
        
        batch_index += 1
    
    return correct / total * 100, losses.avg, acc1.avg.item(), acc5.avg.item()
        
def validate(valid_loader, ema_vgg_model, criterion, device, epoch, writer):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = ema_vgg_model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top1, top5 = accuracy(outputs, labels, topk=(1, 5)) 
            writer.add_scalar("Valid/Loss", loss.item(), epoch)
    return correct / total * 100, top1[0].item(), top5[0].item(), loss.item()
        

def main():
    
    seed = int(os.getenv('seed'))

    # Model configure
    model_arch_name = os.getenv('model_arch_name')
    model_num_classes = int(os.getenv('model_num_classes'))

    # Experiment name, easy to save weights and log files
    exp_name = os.getenv('exp_name')

    # Dataset address
    train_image_dir = os.getenv('train_image_dir')
    test_image_dir = os.getenv('test_image_dir')

    batch_size = int(os.getenv('batch_size'))

    # The address to load the pretrained model
    pretrained_model_weights_path = os.getenv('pretrained_model_weights_path')

    # Total num epochs
    epochs = int(os.getenv('epochs'))

    # Loss parameters
    loss_label_smoothing = float(os.getenv('loss_label_smoothing'))
    
    resume_model_weights_path = os.getenv('resume_model_weights_path')
    # Optimizer parameter
    model_lr = float(os.getenv('model_lr'))
    model_momentum = float(os.getenv('model_momentum'))
    model_weight_decay = float(os.getenv('model_weight_decay'))
    model_ema_decay = float(os.getenv('model_ema_decay'))

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = int(os.getenv('lr_scheduler_T_0'))
    lr_scheduler_T_mult = int(os.getenv('lr_scheduler_T_mult'))
    lr_scheduler_eta_min = float(os.getenv('lr_scheduler_eta_min'))
    
    train_print_frequency = int(os.getenv('train_print_frequency'))
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    
    # CIFAR100 dataset 
    train_loader, valid_loader = data_loader(data_dir=train_image_dir,
                                         batch_size=batch_size)
    
    vgg_model, ema_vgg_model = build_model(num_classes=model_num_classes,model_arch=model_arch_name,model_ema_decay=model_ema_decay,device=device)
    
    criterion = loss_def(loss_label_smoothing, device)
    
    optimizer = optimizer_def(vgg_model, model_lr, model_momentum, model_weight_decay)
    
    scheduler = scheduler_def(optimizer, lr_scheduler_T_0, lr_scheduler_T_mult, lr_scheduler_eta_min)
    
    if os.listdir(resume_model_weights_path) is not None:
        max_epoch = 0
        for files in os.listdir(resume_model_weights_path):
            try:
                index = int(files.split("_")[1].split(".")[0])
                if index > max_epoch:
                    max_epoch = index
            except:
                pass
    
        pretrained_model_weights_path = resume_model_weights_path + f'epoch_{max_epoch}.pth.tar'
        print(max_epoch)
        vgg_model, ema_vgg_model, start_epoch, optimizer, scheduler = load_resume_state_dict(vgg_model, pretrained_model_weights_path, ema_vgg_model, optimizer, scheduler)
        print(f"Loaded `{resume_model_weights_path}` resume model weights successfully.")
        
    else:
        print("Resume model weights not found. Starting from scratch.")
        
        
    results_dir = os.path.join("results", "model",exp_name)
    make_directory(results_dir)
    
        
    for epoch in range(start_epoch, epochs):
        # Create training process log file
        writer_train = SummaryWriter(os.path.join("results", "logs", exp_name, "train", f"epoch_{epoch}"))
        train_acc, loss_train, train_acc1, train_acc5 = train(train_loader, vgg_model, ema_vgg_model, criterion, optimizer, device, scaler, epoch, writer_train, train_print_frequency)
        valid_writer = SummaryWriter(os.path.join("results", "logs", exp_name, "valid", f"epoch_{epoch}"))
        valid_acc, valid_acc1, valid_acc5, loss_valid = validate(valid_loader, ema_vgg_model, criterion, device, epoch, valid_writer)
        
        #save all the variables as a JSON file
        json_path = "./results/model/"+exp_name+"/results.json"
        with open(json_path, "a+") as f:
            if epoch == 0:
                json.dump([{"epoch": epoch + 1,
                           "train_loss": loss_train,
                           "valid_loss": loss_valid,
                           "valid_accuracy": valid_acc,
                           "acc1_train": train_acc1,
                           "acc5_train": train_acc5,
                           "acc1_valid": valid_acc1,
                           "acc5_valid": valid_acc5}], f)
            else:
                details_list = json.load(f)
                details_list.append({"epoch": epoch + 1,
                       "train_loss": loss_train,
                       "valid_loss": loss_valid,
                       "valid_accuracy": valid_acc,
                       "acc1_train": train_acc1,
                       "acc5_train": train_acc5,
                       "acc1_valid": valid_acc1,
                       "acc5_valid": valid_acc5})
                json.dump(details_list, f)
        
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": vgg_model.state_dict(),
                         "ema_state_dict": ema_vgg_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "train_loss": loss_train,
                         "valid_loss": loss_valid,
                         "valid_accuracy": valid_acc,
                         "acc1_train": train_acc1,
                         "acc5_train": train_acc5,
                         "acc1_valid": valid_acc1,
                         "acc5_valid": valid_acc5},
                        f"epoch_{epoch + 1}.pth.tar",
                        results_dir)
        
if __name__ == '__main__':
    main()
    
