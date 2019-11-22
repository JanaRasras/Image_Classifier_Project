import argparse
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def main():
    results = parse_input_args()
    print('data_directory     = {!r}'.format(results.data_directory))
    print('resume     = {!r}'.format(results.resume))
    print('arch     = {!r}'.format(results.arch))
    print('learning_rate     = {!r}'.format(results.learning_rate))
    print('hidden_units     = {!r}'.format(results.hidden_units))
    print('epochs     = {!r}'.format(results.epochs))
    print('gpu     = {!r}'.format(results.gpu))
    data_directory=results.data_directory
    resume=results.resume
    arch=results.arch
    learning_rate=results.learning_rate
    hidden_units=results.hidden_units
    epochs=results.epochs
    gpu=results.gpu
    
    if gpu==False:
        device='cpu'
    else:
        device='cuda'
    if results.resume:
        model=load_checkpoint(results.resume,arch)
    else:
        model=model_def(arch,hidden_units)
    steps = 0
    print_every = 40
    running_loss = 0
    criterion = nn.NLLLoss()
    learning_rate = learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    data_loaders = create_dataloaders(data_directory)
    train(model, data_loaders, optimizer, criterion, epochs, steps, print_every, running_loss, device)
    data_sets = create_datasets(data_directory)
    model.class_to_idx = data_sets['train_data'].class_to_idx
    save_checkpoint(model,arch,model.classifier[0].in_features,102,500,0.5,epochs,"checkpoint.pth",model.class_to_idx)
    
def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', action='store',dest='data_directory',help='data_directory')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--arch', action='store', dest='arch', help='arch',default="vgg16")
    parser.add_argument('--learning_rate', action='store',dest='learning_rate',help='learning_rate',type=float,default=0.01)
    parser.add_argument('--hidden_units', action='store',dest='hidden_units', help='hidden_units',type=int,default=512)
    parser.add_argument('--epochs', action='store',dest='epochs',help='epochs',type=int,default=20)
    parser.add_argument("--gpu",   default=False, action="store_true", help='Bool type gpu')
    results = parser.parse_args()
    return results

def create_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    new_transforms = {'training_transforms': train_transforms,
                                      'validation_transforms': valid_transforms,
                                      'testing_transforms': test_transforms}
    return new_transforms

def create_datasets(data_dir):
    
    new_transforms = create_transforms()
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
                        
    train_data = datasets.ImageFolder(train_dir, transform = new_transforms['training_transforms'])
    valid_data = datasets.ImageFolder(valid_dir, transform = new_transforms['validation_transforms'])
    test_data = datasets.ImageFolder(test_dir, transform = new_transforms['testing_transforms'])
                        
    data_sets = {'train_data': train_data,'valid_data': valid_data,'test_data': test_data}
    return data_sets
def create_dataloaders(data_dir):
                        data_sets = create_datasets(data_dir)
                        trainloader = torch.utils.data.DataLoader(data_sets['train_data'], batch_size=64, shuffle=True)
                        validloader = torch.utils.data.DataLoader(data_sets['valid_data'], batch_size=32)
                        testloader = torch.utils.data.DataLoader(data_sets['test_data'], batch_size=32)
                        
                        data_loaders = {'trainloader': trainloader,
                                       'validloader': validloader,
                                       'testloader': testloader}
                        return data_loaders

def model_def(mod_type,hidden_units):
    if(mod_type=="vgg16"):
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[0].in_features
    else:
        model = models.alexnet(pretrained=True)
        num_ftrs = model.classifier[1].in_features
    
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([

                              ('fc1', nn.Linear(num_ftrs, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))           
                               ]))

    model.classifier = classifier
    return model

def load_checkpoint(filepath,mod_type):
    checkpoint=torch.load(filepath, map_location=lambda storage, loc: storage)

    if(mod_type=="vgg16"):
        model = models.vgg16(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.optim_state = checkpoint['optim_state']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    return model

def validation(model, loader, criterion,device):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def train(model, data_loaders, optimizer, criterion, epochs, steps, print_every, running_loss, device):
    
    model.to(device)
    running_loss = running_loss
    for e in range(epochs):

        model.train()
        for ii, (inputs, labels) in enumerate(data_loaders['trainloader']):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, data_loaders['validloader'], criterion,device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(data_loaders['validloader'])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(data_loaders['validloader'])))

                running_loss = 0
                model.train()
    print('\nModel has trained')

def save_checkpoint(model,arch,input_size,output_size,hidden_layer,dropout,epochs,chkpointname,class_to_idx):
    model.class_to_idx =  class_to_idx
    checkpoint = {'model' : model,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layer': hidden_layer,
                  'dropout' : dropout,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  #'optim_state': optimizer.state_dict(),
                  'idx_to_class': {v: k for k, v in model.class_to_idx.items()},
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier}
    torch.save(checkpoint, chkpointname)
    print("Checkpoint saved")
if __name__== "__main__":
    main()