import json
import time
from lib.lib_flows import *
from lib.lib_models import *
from lib.lib_parameters import *
from lib.lib_processes import *
import poutyne as pt
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import torch.optim as optim

dataset_train = CelebA("data/", split='train', download=True, transform=transformations.normalize_with_celebA)
dataset_test = CelebA("data/", train=False, download=True, transform=transformations.normalize_with_celebA)

## Définition du Data
dataset_train = MNIST("data/", train=True, download=False, transform=transformations.normalize_with_MNIST)
dataset_test = MNIST("data/", train=False, download=False, transform=transformations.normalize_with_MNIST)


## Creation d'un simple classificateur
class Classificateur(nn.Module):

    def __init__(self, tailleDimensionLatente, outFeatures):

        super(Classificateur, self).__init__()
        
        self.fc1 = nn.Linear(
            in_features=tailleDimensionLatente, out_features=tailleDimensionLatente, bias=False
        )
        self.fc2 = nn.Linear(
            in_features=tailleDimensionLatente, out_features=tailleDimensionLatente, bias=False
        )

        self.output = nn.Linear(
            in_features=tailleDimensionLatente, out_features=outFeatures, bias=False
        )

    def forward(self, input):

        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.log_softmax(x,dim=-1) 
        x = x.view((x.shape[0],x.shape[2]))
        return x


## AutoEncoderClassique


AutoEncoderPath = "results/experimentClassic/models/ClassicAutoencoder_Adam_MSE_lr=0.001.pth"
Autoencoder = classicAutoencoder.Autoencoder(784,1.1)
Autoencoder.load_state_dict(torch.load(AutoEncoderPath, map_location=torch.device('cpu')))


for i in Autoencoder.parameters():
    i.requires_grad = False


dataset_train_encoded = list()
for i in range(len(dataset_train)):   
    x = dataset_train[i][0]
    x = x.view(x.shape[0], -1)
    x = Autoencoder.encoder(x)
    x = F.relu(x)
    dataset_train_encoded.append((x,dataset_train[i][1]))

dataset_test_encoded = list()
for i in range(len(dataset_test)):   
    x = dataset_test[i][0]
    x = x.view(x.shape[0], -1)
    x = Autoencoder.encoder(x)
    x = F.relu(x)
    dataset_test_encoded.append((x,dataset_test[i][1] ))

dataset_train_encoded, dataset_valid_encoded = torch.utils.data.random_split(dataset_train_encoded, [55000, 5000])
train_loader = DataLoader(dataset_train_encoded, batch_size=64,shuffle=True)
valid_loader = DataLoader(dataset_valid_encoded, batch_size=64,shuffle=True)
##Modèle

callbacks = [
              pt.ReduceLROnPlateau(monitor='val_acc',verbose=True, mode='min', factor=0.3, patience=5, threshold=0.001),
              # On va ajouter 25 epoch pour être certain qu'il n'arrête pas lorsque la croissance est très faible.
              pt.EarlyStopping(monitor='acc', mode='max', min_delta=(0.005), patience=10, verbose=True),
              ]
modele = Classificateur(862,10)
optimizer = optim.Adam(modele.parameters())
model = pt.Model(modele, optimizer, 'nllloss', batch_metrics=['accuracy'])
history = model.fit_generator(train_loader,valid_loader, epochs=100, callbacks=callbacks)


##Irmae

irmaPath = "results/experimentIrma/models/IrmaAutoencoder_Adam_MSE_lr=0.001.pth"
irma = irmaAutoencoder.Autoencoder(784,1.1,4)
irma.load_state_dict(torch.load(irmaPath, map_location=torch.device('cpu')))
for i in irma.parameters():
    i.requires_grad = False

dataset_train_encoded = list()
for i in range(len(dataset_train)):   
    x = dataset_train[i][0]
    x = x.view(x.shape[0], -1)
    x = irma.encoder(x)
    x = F.relu(x)
    for layer in irma.linear_layers:
        x = layer(x)
    dataset_train_encoded.append((x,dataset_train[i][1]))

dataset_test_encoded = list()
for i in range(len(dataset_test)):   
    x = dataset_test[i][0]
    x = x.view(x.shape[0], -1)
    x = irma.encoder(x)
    x = F.relu(x)
    for layer in irma.linear_layers:
        x = layer(x)
    dataset_test_encoded.append((x,dataset_test[i][1] ))

dataset_train_encoded, dataset_valid_encoded = torch.utils.data.random_split(dataset_train_encoded, [55000, 5000])
train_loader = DataLoader(dataset_train_encoded, batch_size=64,shuffle=True)
valid_loader = DataLoader(dataset_valid_encoded, batch_size=64,shuffle=True)

modele = Classificateur(862,10)
optimizer = optim.Adam(modele.parameters())
model = pt.Model(modele, optimizer, 'nllloss', batch_metrics=['accuracy'])
history = model.fit_generator(train_loader,valid_loader, epochs=100, callbacks=callbacks)



##Contractive AutoEncoder

contractiveAutoEncoderPath = "results/experimentContractive/models/ContractiveAutoencoder_Adam_Contractive_lr=0.001.pth"
contractiveAutoEncoder = contractiveAutoencoder.Autoencoder(784,1.1)
contractiveAutoEncoder.load_state_dict(torch.load(contractiveAutoEncoderPath, map_location=torch.device('cpu')))

for i in contractiveAutoEncoder.parameters():
    i.requires_grad = False

dataset_train_encoded = list()
for i in range(len(dataset_train)):   
    x = dataset_train[i][0]
    x = x.view(x.shape[0], -1)
    x = contractiveAutoEncoder.encoder(x)
    x = F.relu(x)
    dataset_train_encoded.append((x,dataset_train[i][1] ))

dataset_test_encoded = list()
for i in range(len(dataset_test)):   
    x = dataset_test[i][0]
    x = x.view(x.shape[0], -1)
    x = contractiveAutoEncoder.encoder(x)
    x = F.relu(x)
    dataset_test_encoded.append((x,dataset_test[i][1] ))


dataset_train_encoded, dataset_valid_encoded = torch.utils.data.random_split(dataset_train_encoded, [55000, 5000])
train_loader = DataLoader(dataset_train_encoded, batch_size=64,shuffle=True)
valid_loader = DataLoader(dataset_valid_encoded, batch_size=64,shuffle=True)

modele = Classificateur(862,10)
optimizer = optim.Adam(modele.parameters())
model = pt.Model(modele, optimizer, 'nllloss', batch_metrics=['accuracy'])
history = model.fit_generator(train_loader,valid_loader, epochs=100, callbacks=callbacks)




