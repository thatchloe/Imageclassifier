from PIL import Image
import json
import torch
from torch.autograd import Variable
import numpy as np
import time
from helpers import predict_input_args, load_checkpoint, category_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    wid = 256
    hei = 256
    if image.size[0] > image.size[1]:
        wid = image.size[0]
    else:
        hei = image.size[1]
    
    image.thumbnail((wid,hei))
    cropped1 = (256 - 224)/2
    cropped2 = (256 + 224)/2
    image = image.crop((cropped1, cropped1, cropped2, cropped2))
    image = np.array(image)
    image = image/255
    mean = np.array([0.485, 0.456, 0.406])
    stdv = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/stdv
    image = image.transpose(2, 0, 1)
    
    return image



def predict(image, args, model):
    device = torch.device('cuda' if args.gpu == 'gpu' else 'cpu')
    
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)
    image = Variable(image)
    
    C_I = model.class_to_idx
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model.forward(image)
    
    #get top K largest class probabilities
    ps = torch.exp(output)
    top_p, top_class = ps.topk(args.top_k, dim=1)
    top_p = np.array(top_p)[0]  
    I_C = {i: c for (c, i) in C_I.items()}
    top_classes = np.array([I_C[idx] for idx in np.array(top_class)[0]])    
    
    return (top_p, top_classes)
    
    
def main():
    in_args = predict_input_args()
    image = Image.open(in_args.image_dir)
    image = process_image(image)
    model = load_checkpoint(in_args.checkpoint)
    
    cat_to_name = category_name(in_args.category_names)
    
    probs, classes = predict(image, in_args, model)
    top_labels = [cat_to_name[i] for i in classes]
    label_prob = {l: p for (l, p) in list(zip(top_labels, probs))}
    
    for name, prob in label_prob.items():
        print(f'image name: {name} , probability: {prob}')
    print(f'Device: {in_args.gpu}')   
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    