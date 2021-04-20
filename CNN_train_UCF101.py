import torch
from torchvision import datasets,models,transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                normalize
            ]),


        'test':
            transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                normalize
            ])

        }
train_set = datasets.ImageFolder('./data2/train', data_transforms['train'])
train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                pin_memory = True)
test_set = datasets.ImageFolder('./data2/test', data_transforms['test'])
test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                pin_memory = True)

trainclasses = len(train_set)
testclasses = len(test_set)
#print(classes)

def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(pretrained = True)


    for param in model.parameters():
        param.requires_grad = False
    #print(device)
    model.fc = nn.Sequential(
        nn.Linear(2048,1024),
        nn.Dropout(0.55),
        nn.LeakyReLU(inplace = True),
        nn.Linear(1024,101))
    model = model.to(device)

    return model








if __name__ == '__main__':
    #torch.manual_seed(10)
    num_epochs = 60
    model = get_model()
    #print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(),lr=0.0010)
    scheduler = StepLR(optimizer,step_size=100,gamma=0.9)

    ##Training-----------------------------------------------
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('--'*10)
        running_loss = 0.0
        running_corrects = 0
        running_acc = 0.

        for inputs,labels in train_loader:
            model.train(True)
            inputs,labels = inputs.cuda(),labels.cuda()
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_corrects = (preds == labels.data).sum()
            running_acc += running_corrects.item()

            optimizer.zero_grad()


        print('Train Loss:{:.6f},Acc:{:.6f}'.format(running_loss/trainclasses,
              running_acc/trainclasses))

        #state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #filepath = './changesgdstate/' + 'Epoch{:d}'.format(epoch+1) + 'model.pth'
        #print(filepath)
        #torch.save(state, filepath)
        #print('Model of Epoch {:d} has been saved'.format(epoch+1))

        #torch.save(state,'./data')

    ##Evaluation----------------------------------------
        with torch.no_grad():
            model.eval()
            test_loss = 0.
            test_acc = 0.
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(),labels.cuda()
                outputs= model(inputs)
                loss = criterion(outputs,labels)
                test_loss += loss.item()
                preds = torch.argmax(outputs,dim=1)
                num_acc = (preds==labels.data).sum()
                test_acc += num_acc.item()

            print('Test Loss:{:.6f},Acc:{:.6f}'.format((test_loss/testclasses),(test_acc/testclasses)))