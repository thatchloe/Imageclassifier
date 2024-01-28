import argparse
import torch
import json


def train_input_args():
    
    parser = argparse.ArgumentParser(description = 'training')
    parser.add_argument('--data_dir', type = str, action = 'store', default = 'flowers', 
                    help = 'path to the folder of images') 
    parser.add_argument('--arch', type = str, default = 'densenet121', choices = ['vgg16', 'densenet121'],
                    help = 'The CNN model architecture to use')
    parser.add_argument('--save_dir', type = str, action = 'store', default = 'checkpoint.pth',
                    help = 'saving directory')
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'learning_rate')
    parser.add_argument('--hidden_units', type = int, default = 512)
    parser.add_argument('--epochs', type = int, default = 3)
    parser.add_argument('--gpu', type = str, action = 'store', default = 'gpu', 
                    help = 'choose a device to train on')
    return parser.parse_args()
       
    
def predict_input_args():
    
    parser = argparse.ArgumentParser(description = 'predicting')
    parser.add_argument('--image_dir', type = str, action = 'store', default = 'flowers/test/1/image_06752.jpg',
                       help = 'image directory')
    parser.add_argument('--checkpoint', type = str, action = 'store', default = 'checkpoint.pth',
                       help = 'loading model directory')
    parser.add_argument('--top_k', type = int, default = 5,
                       help = 'top K highest class probabilities')
    parser.add_argument('--category_names', type = str, action = 'store', default = 'cat_to_name.json')
    parser.add_argument('--gpu', type = str, action = 'store', default = 'gpu')
    
    return parser.parse_args()


def save_checkpoint(save_dir, optimizer, args, classifier, model):
    
    checkpoint = {'model': model,
                  'arch': args.arch,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, save_dir)
    
def load_checkpoint(file_path):
    
    checkpoint = torch.load(file_path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def category_name(file_name):
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name




