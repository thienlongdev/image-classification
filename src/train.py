import torch
import torch.nn as nn
import torch.optim
from cifar10 import MyDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from models import SimpleCNN
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
def get_args():
    parser = ArgumentParser(description='CNN training')
    parser.add_argument('--root', '-r', type=str, default='./cifar10/cifar-10-batches-py', help='Root of the dataset')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--image-size', '-i', type=int, default=224, help='image size')
    parser.add_argument('--logging', '-l', type=str, default='tensorboard')
    parser.add_argument('--trained_models', '-m', type=str, default='trained_model')
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_matrix', figure, epoch)

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])
    train_dataset = MyDataset(root=args.root, train=True, transform = transform)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        num_workers = 16,
        shuffle = True,
        drop_last = True
    )

    test_dataset = MyDataset(root=args.root, train = False, transform = transform)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        num_workers = 16,
        shuffle = False,
        drop_last = False
    )

    # if os.path.isdir(args.logging):
    #     shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.makedirs(args.trained_models)
    writer = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_acc = 0

    num_i = len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour='cyan')
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description(f'epoch {epoch + 1}/{args.epochs}, Iteration {i + 1}/{num_i}, Loss {loss_value:.3f}')
            writer.add_scalar('Train/Loss', loss_value, epoch * num_i + i)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for i, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predictions = model(images)
                all_predictions.extend(torch.argmax(predictions.cpu(), dim=1))
                loss_value = criterion(predictions, labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [predictions.item() for predictions in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=test_dataset.categories, epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Epoch {epoch + 1}. Accuracy {accuracy_score(all_labels, all_predictions)}")
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        # torch.save(model.state_dict(), f"{args.trained_models}/last_cnn.pt")
        checkpoint = {
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, f'{args.trained_models}/last_cnn.pt')
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint = {
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, f'{args.trained_models}/best_cnn.pt')
            best_acc = accuracy
        # print(classification_report(all_labels, all_predictions))
