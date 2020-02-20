from UCFdata import load_train_data,load_test_data
import torch
from torchvision import datasets,models,transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import copy
result = []

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

image_datasets = {
        'train':
            datasets.ImageFolder('./data/train', data_transforms['train']),
        'test':
            datasets.ImageFolder('./data/test', data_transforms['test'])
        }

dataloaders = {
        'train':
            torch.utils.data.DataLoader(
                image_datasets['train'],
                batch_size=32,
                shuffle=True,
                num_workers=0),
        'test':
            torch.utils.data.DataLoader(
                image_datasets['test'],
                batch_size=32,
                shuffle=True,
                num_workers=0),
        }




def get_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(pretrained = True)


    for param in model.parameters():
        param.requires_grad = False
    print(device)
    model.fc = nn.Sequential(
        nn.Linear(2048,1024),
        nn.ReLU(inplace = True),
        nn.Linear(1024,101))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    return model,criterion,optimizer




def train_model(model,criterion,optimizer,num_epochs,lr=1e-3):

    #train_loader = load_data().get('train')
    #test_loader = load_data().get('test')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('--'*10)

        for phase in ['train','test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()


            running_loss = 0.0
            running_corrects = 0

            for inputs,labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs,_ = model(inputs)
                #print(outputs)


                #print(labels)
                loss = criterion(outputs,labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _,preds = torch.max(outputs,1)
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss /len(image_datasets[phase])
            epoch_acc = running_corrects.double()/len(image_datasets[phase])


            print('{} loss:{:.4f},acc:{:.4f}'.format(phase,
                                                     epoch_loss,
                                                     epoch_acc))


            optimizer.step()



    return model
'''''
def plot():
    data = np.loadtxt('result.csv',delimiter=',')
    plt.figure()
    plt.plot(range(1,len(data[:,0] + 1),
                   data[:,0],color = 'blue',label = 'train'))
    plt.legend()
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Accuracy',fontsize=14)
    plt.title('Training Accuracy',fontsize=20)
    plt.show()
'''''

if __name__ == '__main__':
    torch.manual_seed(10)
    model,criterion,optimizer = get_model()
    train_model(model,criterion,optimizer,30,1e-3)