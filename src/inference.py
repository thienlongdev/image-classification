import torch
import torch.nn as nn
import torch.optim
from cifar10 import MyDataset
from torchvision.transforms import ToTensor, Compose, Resize
from models import SimpleCNN
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

def get_args():
    parser = ArgumentParser(description='CNN inference')
    parser.add_argument('--image-path', '-p', type=str, default=None)
    parser.add_argument('--image-size', '-i', type=int, default=224, help='image size')
    parser.add_argument('--checkpoint', '-c', type=str, default='trained_model/best_cnn.pt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SimpleCNN(num_classes=10).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
    else:
        print('No checkpoint found.')
        exit(0)

    model.eval()
    categories = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.0
    image = image[None,:,:,:] # 1 X 3 X 224 X 224
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(image)
        print(output)
        probs = softmax(output)
        print(probs)

    max_i = torch.argmax(probs)
    predicted_class = categories[max_i]
    print(f'The test image is about {predicted_class} with confident score of {probs[0, max_i]}')
    cv2.imshow(f'{predicted_class}:{probs[0, max_i]*100:.2f}%', ori_image)
    cv2.waitKey(0)