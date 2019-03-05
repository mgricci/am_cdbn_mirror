from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import ipdb

class Net(nn.Module):
    def __init__(self, img_side):
        super(Net, self).__init__()
	self.img_side = img_side
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def cplx_max_pool2d(self, z_norm, z_angle, k, stride):
        unfolded_z_norm   = z_norm.unfold(2, k, stride).unfold(3,k,stride)
	shp 	          = unfolded_z_norm.shape
	unfolded_z_norm   = unfolded_z_norm.reshape(shp[0], shp[1], shp[2], shp[3], -1)
	m, am             = torch.max(unfolded_z_norm,-1, keepdim=True)
	z_angle_unfolded  = z_angle.unfold(2,k,stride).unfold(3,k,stride).reshape(shp[0], shp[1], shp[2], shp[3], -1)
	am_angle          = torch.gather(z_angle_unfolded,-1, am)
	return (m*torch.cos(am_angle)).squeeze(-1), (m*torch.sin(am_angle)).squeeze(-1)
	

    def forward_cplx(self, z):
	z_real = z[:,0,:,:].unsqueeze(1)
	z_imag = z[:,1,:,:].unsqueeze(1)
	# Layer 1 
	batch_size = z_real.shape[0]
	z_real = self.conv1(z_real)
	z_imag = self.conv1(z_imag)  - self.conv1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(z_real.shape[0],1,z_real.shape[-2], z_real.shape[-1])
	z_norm = torch.sqrt((z_real)**2 + z_imag**2)
        z_norm = torch.sigmoid(z_norm)
	z_angle = torch.atan2(z_imag, z_real)
        z_real, z_imag = self.cplx_max_pool2d(z_norm, z_angle, 2, 2)

	#Layer 2
	z_real = self.conv2(z_real)
	z_imag = self.conv2(z_imag) - self.conv2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(z_real.shape[0],1,z_real.shape[-2], z_real.shape[-1])
	z_norm = torch.sqrt((z_real)**2 + z_imag**2)
        z_norm = torch.sigmoid(z_norm)
	z_angle = torch.atan2(z_imag, z_real)
        z_real, z_imag = self.cplx_max_pool2d(z_norm, z_angle, 2, 2)

	# Layer 3
	z_real = z_real.view(-1,4*4*64)
	z_imag = z_imag.view(-1,4*4*64)
	z_real = self.fc1(z_real)
	z_imag = self.fc1(z_imag) - self.fc1.bias.unsqueeze(0)
	z_norm = torch.sigmoid(torch.sqrt((z_real)**2 + z_imag**2))
	z_angle = torch.atan2(z_imag, z_real)
	z_real, z_imag = z_norm*torch.cos(z_angle), z_norm*torch.sin(z_angle)

	# Layer 4
	z_real, z_imag = self.fc2(z_real), self.fc2(z_imag) - self.fc2.bias.unsqueeze(0)
	z_norm = torch.sqrt(z_real**2 + z_imag**2)
	z_norm = F.softmax(z_norm, dim=1)
	z_angle = torch.atan2(z_imag, z_real)
	
	return torch.cat((z_norm*torch.cos(z_angle).view(batch_size,-1), z_norm*torch.sin(z_angle).view(batch_size,-1)),dim=1)

    def run_single_cplx(self, z):
        z_real = torch.tensor(z[:self.img_side**2].reshape(1,1,self.img_side, self.img_side)).cuda()
        z_imag = torch.tensor(z[self.img_side**2:2*self.img_side**2].reshape(1,1,self.img_side, self.img_side)).cuda()
        return self.forward_cplx(z_real, z_imag)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
	target_onehot = torch.zeros(target.shape[0],10).cuda()
	target_onehot[range(target.shape[0]), target] = 1
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target_onehot)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
	    target_onehot = torch.zeros(target.shape[0], 10).cuda()
	    target_onehot[range(target.shape[0]), target] = 1
            output = model(data)
            test_loss += F.mse_loss(output, target_onehot)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def run():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(28).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"/home/jk/matt/mnist_cnn.pt")
        
if __name__ == '__main__':
    run()
