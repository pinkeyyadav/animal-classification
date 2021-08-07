import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
import torch
import torchvision.models as MODEL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


CNN_2D = MODEL.vgg16(pretrained=True, progress=True).eval()



def DATA_LOADER():
    train_folder = "F:\\new\\DATASET\\train\\"
    test_folder = "F:\\new\\DATASET\\test\\"
    BATCH_SIZE = 10
    Transforming = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.4, 0.5], std=[0.22, 0.24, 0.22])])

    train_data = torchvision.datasets.ImageFolder(root=train_folder, transform=Transforming)
    test_data = torchvision.datasets.ImageFolder(root=test_folder, transform=Transforming)
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return train_loader, test_loader