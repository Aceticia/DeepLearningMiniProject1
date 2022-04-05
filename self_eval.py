import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

# DO NOT import unnecessary libraries in `project1_model.py`
# Just import libraries enough to define the model
# Otherwise if we donot have the library at our side, the script will fail

def load_model(device):
    model = None
    try:
        from project1_model import project1_model
        model = project1_model().to(device)
    except:
        print("FAIL TO LOAD MODEL")
    return model

def test(model, testloader, criterion, device):
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / len(testloader.dataset)
    loss = test_loss / len(testloader)
    return acc, loss

def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    validset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform_test)
    validloader = torch.utils.data.DataLoader(
      validset, batch_size=1000)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        model = load_model(device)
        model.load_state_dict(torch.load('./project1_model.pt', map_location=device), strict=False)
        criterion = nn.CrossEntropyLoss()
        
        model.eval()
        v_acc, _ = test(model, validloader, criterion, device)
        print('Valid Accuracy: {:.1f}'.format(v_acc))

    except:
        print("FAIL")

if __name__ == '__main__':
    main()

