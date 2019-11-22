import pandas as pd
import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import glob, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
#user def functions 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    img_tensor = preprocess(Image.open(image))
    
    return img_tensor
def predict(image_path, model, device,topk=5,):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    # Pre-process image to numpy array
    model.to(device)
    #np_image = process_image(image_path)
    processed_image = process_image(image_path)
    #processed_image = torch.from_numpy(processed_image).float()
    processed_image = processed_image.unsqueeze_(0)
    with torch.no_grad():
        #outputs = model(processed_image)
        outputs = model.forward(processed_image.to(device))
             
    ps = torch.exp(outputs)
    probs, classes = torch.topk(ps, topk)
    return(probs, classes)
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    if title is not None:
        plt.title(title)
    ax.imshow(image)
    
    return ax
#end user def functions
def parse_input_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', action='store',
                        dest='path_checkpoint',default="/home/workspace/checkpoint20.pth",
                        help='path_checkpoint')
    parser.add_argument('--path_image', action='store',
                        dest='path_image',default="/home/workspace/aipnd-project/flowers/train/10/image_07086.jpg",
                        help='path_image')
    parser.add_argument('--category_names', action='store',
                        dest='category_names',
                        help='category_names',default="/home/workspace/aipnd-project/cat_to_name.json")
    parser.add_argument('--top_k', action='store',
                        dest='top_k',
                        help='top_k',type=int,default=3)
    parser.add_argument("--gpu",   default=False, action="store_true", help='Bool type gpu')
    results = parser.parse_args()
    return results
def main(): 
    print('Starting Prediction...')
    results = parse_input_args()
    print('path_checkpoint     = {!r}'.format(results.path_checkpoint))
    print('path_image     = {!r}'.format(results.path_image))
    print('category_names     = {!r}'.format(results.category_names))
    print('top_k     = {!r}'.format(results.top_k))
    print('gpu     = {!r}'.format(results.gpu))



    path_checkpoint=results.path_checkpoint
    path_image=results.path_image
    category_names=results.category_names
    top_k=results.top_k

    gpu=results.gpu
    if gpu==False:
        print("cpu is used")
        device='cpu'
    else:
        print("gpu is used")
        device='cuda'
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(path_checkpoint)
    #ima=path_image
    #imagen=process_image(image=ima)
    prob, classes = predict(path_image, model,device,top_k,)
    #print(prob, classes)
    inv_dict = {values : keys for keys, values in model.class_to_idx.items()}
    classes_list = [inv_dict[idx] for idx in classes.cpu().numpy()[0]]
    probs_list = prob.cpu().numpy()[0].tolist()
    cat_list = [cat_to_name[lbl] for lbl in classes_list]
    print(probs_list, cat_list)
    print("Flower is ",cat_list[0])
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
if __name__== "__main__":
    main()
