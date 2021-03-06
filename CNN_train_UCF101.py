import torch
from torchvision import datasets,models,transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import copy
import os
from torch.autograd import Variable


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
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
train_set = datasets.ImageFolder('./data/train', data_transforms['train'])
train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=32,
                shuffle=True,
                num_workers=0)
test_set = datasets.ImageFolder('./data/test', data_transforms['test'])
test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=32,
                shuffle=True,
                num_workers=0)



classes = len(train_set)
#print(classes)

def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(pretrained = True)


    for param in model.parameters():
        param.requires_grad = False
    #print(device)
    model.fc = nn.Sequential(
        nn.Linear(2048,1024),
        nn.ReLU(inplace = True),
        nn.Linear(1024,101))
    model = model.to(device)



    return model







if __name__ == '__main__':
    torch.manual_seed(10)
    num_epochs = 30
    model = get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

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
            outputs,_ = model(inputs)

            loss = criterion(outputs, labels)



            _, preds = torch.max(outputs, 1)
            #print(inputs.size(0))
            #print(loss.item())
            running_loss += loss.item()
            running_corrects = (preds == labels.data).sum()
            running_acc += running_corrects.item()

            #print("loss:")
            #print(loss.item())
            #print("running_loss:")
            #print(running_loss)
            #print("running_corrects:")
            #print(running_corrects)
            #print("running_acc:")
            #print(running_acc)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Loss:{:.6f},Acc:{:.6f}'.format(running_loss/classes,
              running_acc/classes))

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        filepath = './state/' + 'Epoch{:d}'.format(epoch) + 'model.pth'
        print(filepath)
        torch.save(state, filepath)
        print('Model of Epoch {:d} has been saved'.format(epoch))

        #torch.save(state,'./data')

    ##Evaluation----------------------------------------
        model.eval()
        test_loss = 0.
        test_acc = 0.
        
        for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs,_ = model(inputs)
                loss = criterion(outputs,labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                num_acc = (preds==labels).sum()
                test_acc += num_acc.item()

        print('Test Loss:{:.6f},Acc:{:.6f}'.format((test_loss/classes),(test_acc/classes)))
