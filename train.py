from torch import nn, optim
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from helpers import train_input_args, save_checkpoint

def train(model, args, dataloader, optimizer, criterion, device): 
    steps = 0
    running_loss = 0
    print_every = 10
    start = time.time()
    
    for epoch in range(args.epochs):
        for inputs, labels in dataloader[0]:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            
            
            optimizer.zero_grad()
            #feed forward
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            #backpropogation
            loss.backward()
            optimizer.step() 

            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader[1]:
                        inputs, labels = inputs.to(device), labels.to(device)

                        #feed forward and calculate loss
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloader[1]):.3f}.. "
                      f"Accuracy: {accuracy/len(dataloader[1]):.3f}")
                
                running_loss = 0
                model.train()
    end_time = time.time()
    print(f"Device = {args.gpu}")
    print(f"Training time: {(end_time - start)//60}mins {(end_time - start)%60:.2f}seconds")
    

def main():
    in_args = train_input_args()
    train_dir = in_args.data_dir + '/train'
    valid_dir = in_args.data_dir + '/valid' 
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    dataloader = [trainloader, validloader]
    
    if in_args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier = nn.Sequential(OrderedDict([("fc1",nn.Linear(1024, in_args.hidden_units)),
                                     ("ReLU1",nn.ReLU()),
                                     ("dropout",nn.Dropout(0.5)),
                                     ("fc2",nn.Linear(in_args.hidden_units, 102)),
                                     ("output",nn.LogSoftmax(dim=1))]))
    elif in_args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(OrderedDict([("fc1",nn.Linear(25088, in_args.hidden_units)),
                                     ("ReLU1",nn.ReLU()),
                                     ("dropout",nn.Dropout(0.5)),
                                     ("fc2",nn.Linear(in_args.hidden_units, 102)),
                                     ("output",nn.LogSoftmax(dim=1))]))
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_args.learning_rate)
    device = torch.device('cuda' if in_args.gpu == 'gpu' else 'cpu')
    train(model, in_args, dataloader, optimizer, criterion, device)
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(in_args.save_dir, optimizer, in_args, classifier, model)
    
if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    