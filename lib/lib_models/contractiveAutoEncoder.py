import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CAE(nn.Module):
	def __init__(self):
		super(CAE, self).__init__()

		self.fc1 = nn.Linear(784, 900, bias = False) # Encoder
		self.fc2 = nn.Linear(900, 784, bias = False) # Decoder

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def encoder(self, x):
		h1 = self.relu(self.fc1(x.view(-1, 784)))
		return h1

	def decoder(self,z):
		h2 = self.sigmoid(self.fc2(z))
		return h2

	def forward(self, x):
            h1 = self.encoder(x)
            h2 = self.decoder(h1)
            return h1, h2

        # Writing data in a grid to check the quality and progress
	def samples_write(self, x, epoch):
		_, samples = self.forward(x)
		#pdb.set_trace()
		samples = samples.data.cpu().numpy()[:16]
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)
		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
		if not os.path.exists('out/'):
			os.makedirs('out/')
		plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
		#self.c += 1
		plt.close(fig)

