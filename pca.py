from lib.lib_models import *
from lib.lib_flows import *
from lib.lib_processes import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torchvision.datasets import MNIST
import torchvision
import matplotlib.pyplot as plt
from sklearn import *
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
 
## Dataset Definition
dataset_train = MNIST("data\\", train=True, download=False, transform=transformations.normalize_with_MNIST)
dataset_test = MNIST("data\\", train=False, download=False, transform=transformations.normalize_with_MNIST)



## AutoEncoderClassique
AutoEncoderPath = "results/experimentClassic/models/ClassicAutoencoder_Adam_MSE_lr=0.001.pth"
Autoencoder = classicAutoencoder.Autoencoder(784,1.1)
Autoencoder.load_state_dict(torch.load(AutoEncoderPath, map_location=torch.device('cpu')))

## AutoEncoderIrma
irmaPath = "results/experimentIrma/models/IrmaAutoencoder_Adam_MSE_lr=0.001.pth"
irma = irmaAutoencoder.Autoencoder(784,1.1,4)
irma.load_state_dict(torch.load(irmaPath, map_location=torch.device('cpu')))


## contractiveAutoEncoder
contractiveAutoEncoderPath = "results/experimentContractive/models/ContractiveAutoencoder_Adam_Contractive_lr=0.001.pth"
contractiveAutoEncoder = contractiveAutoencoder.Autoencoder(784,1.1)
contractiveAutoEncoder.load_state_dict(torch.load(contractiveAutoEncoderPath, map_location=torch.device('cpu')))



color = list()
hiddenRepresentationContractiveAutoEncoder = list()
hiddenRepresentationIrma = list()
hiddenRepresentationAE = list()


## Latent Representation
for i in range(1000):
    x = dataset_train[i][0]
    label = dataset_train[i][1]
    color.append(label)
    x = x.view(x.shape[0], -1)

    x_autoEncoder = Autoencoder.encoder(x)
    x_autoEncoder = F.relu(x_autoEncoder)
    x_autoEncoder = np.array(x_autoEncoder.detach()[0])
    

    x_irma = irma.encoder(x)
    x_irma = F.relu(x_irma)
    for layer in irma.linear_layers:
        x_irma = layer(x_irma)

    x_irma = np.array(x_irma.detach()[0])

    x_ContractiveAutoEncoder = contractiveAutoEncoder.encoder(x)
    x_ContractiveAutoEncoder = F.relu(x_ContractiveAutoEncoder)
    x_ContractiveAutoEncoder = np.array(x_ContractiveAutoEncoder.detach()[0])

    hiddenRepresentationAE.append(x_autoEncoder)
    hiddenRepresentationIrma.append(x_irma)
    hiddenRepresentationContractiveAutoEncoder.append(x_ContractiveAutoEncoder)



##Fitting PCA

pcaAutoEncoder = decomposition.PCA(n_components=2)
pcaAutoEncoder.fit(hiddenRepresentationAE)
XAutoEncoder = pcaAutoEncoder.transform(hiddenRepresentationAE)

pcaIrmae = decomposition.PCA(n_components=2)
pcaIrmae.fit(hiddenRepresentationIrma)
XIrmae = pcaIrmae.transform(hiddenRepresentationIrma)

pcaContractiveAE = decomposition.PCA(n_components=2)
pcaContractiveAE.fit(hiddenRepresentationContractiveAutoEncoder)
XContractiveAE = pcaContractiveAE.transform(hiddenRepresentationContractiveAutoEncoder)


##Ploting Results
_, axes = plt.subplots(1, 3)
axes[0].scatter(XAutoEncoder[:, 0], XAutoEncoder[:, 1], c=color)
axes[0].title.set_text("AutoEncoder")
axes[0].legend()
axes[1].scatter(XIrmae[:, 0], XIrmae[:, 1], c=color)
axes[1].title.set_text("Irmae")
axes[2].scatter(XContractiveAE[:, 0], XContractiveAE[:, 1], c=color)
axes[2].title.set_text("Contractive AutoEncoder")
plt.show()