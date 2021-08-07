import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch
import torchvision.models as MODELS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def DATA_LOADER():
    train_folder = "F:\\new\\DATASET\\train\\"
    test_folder = "F:\\new\\DATASET\\train\\"
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


net = MODELS.vgg16(pretrained=True, progress=True).eval()
for params in net.parameters():
    params.require_grad = False
net.classifier[6] = nn.Linear(4096, out_features=4)
#net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
train_loader, test_loader = DATA_LOADER()
#net.to(device='cuda')


def main():
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
 #           inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    print('Finished Training')
    PATH = './transfer_learning_v_1.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()