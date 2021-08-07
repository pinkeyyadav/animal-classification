import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as MODELS
import torch.utils.data as data
import cv2
import torch.nn as nn


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
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader


train_loader, test_loader = DATA_LOADER()
dataiter = iter(test_loader)
images, labels = dataiter.next()
model = MODELS.vgg16(pretrained=True)
#model.classifier[6] = nn.Linear(4096, out_features=4)
model.load_state_dict(torch.load('F:\\new\\animal_classification-V_1.pth'))
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))

#out = model(inputimg)
#print(out.shape)
#print(model)