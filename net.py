from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable
from data_loader import ImageFolder
from tensorboardX import SummaryWriter
# SUM or MEAN Margin Loss

def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=33): 
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=2) # Input 224 Output 112
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) # Input 112 Output 56
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3,stride=2, padding=1) # Output 28
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1) # Output 26
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1) # Output 24
        # self.pool = nn.MaxPool2d(2, 1)  
        # self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # Output 22
        self.primaryCaps = PrimaryCapsLayer(256, 30, 8, kernel_size=9, stride=1)  # outputs 44
        self.num_primaryCaps = 30 * 17 * 17
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)
        self.Linear = nn.Linear(16, n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        # x = self.pool(x)
        # x = self.conv6(x)
        # x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        # print("X :", x.shape)



        y = self.Linear(x)
        probs = y.pow(2).sum(dim=2).sqrt()
        # probs = x.pow(2).sum(dim=2).sqrt()
        # print("Y :", y.shape)
        # x = F.softmax(x, dim=0)
        # print(y)
        # print(x)
        # print(x)
        # print(x)
        # print(x1)
        # print(x2)
        
        # print("Probs ", probs.shape)
        # print(probs)
        # print("x :", x,"Probs :",probs)
        return x, probs

class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=33):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 50176)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        target = target.long()
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=False):
        targets = targets.long()
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    writer = SummaryWriter()
    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=12, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=True)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


    train_data = ImageFolder("/media/jiashu/Data/Peal_Train/1/", transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers = 1)

    test_data = ImageFolder("/media/jiashu/Data/Peal_Test/2/", transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers = 1)

    model = CapsNet(args.routing_iterations)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, 33)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = MarginLoss(0.9, 0.1, 0.5)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    def accuracy(output, target, topk=(1,)):

        target = target.long()
        maxk = max(topk)

        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        result = res[0]
        return result

    def train(epoch):
        model.train()
        losslist = []
        correctlist = []

        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 50176))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
                losslist.append(loss)
                # print("Margin :", margin_loss, "Re :", reconstruction_loss)
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
                losslist.append(loss)
            loss.backward()
            optimizer.step()
            correct = accuracy(probs, target) 
            correctlist.append(correct)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data, correct ))
        correct = 0.
        loss = 0.
        i = 0
        for singleresult in correctlist:
            correct = correct + singleresult
            i = i + 1
        for singleloss in losslist:
            loss = loss + singleloss       
        correct = correct/i
        loss = loss/i
        writer.add_scalar('data/loss', loss, epoch)
        writer.add_scalar('data/acc', correct, epoch)

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        correctlist = []
        avgcorrect = 0.
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                if args.with_reconstruction:
                    output, probs = model(data, target)
                    # print(output.size)
                    reconstruction_loss = F.mse_loss(output, data.view(-1, 50176)).data
                    test_loss += loss_fn(probs, target).data
                    test_loss += reconstruction_alpha * reconstruction_loss
                else:
                    output, probs = model(data)
                    test_loss += loss_fn(probs, target).data

                pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                
                correct = accuracy(probs, target) 
                correctlist.append(correct)

        i = 0
        correct = 0
        for singleresult in correctlist:
            correct = correct + singleresult
            i = i + 1
        avgcorrect = correct/i
        avgnum = avgcorrect * len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, avgnum, len(test_loader.dataset),
            avgcorrect))

        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', avgcorrect, epoch)

        return test_loss


    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss)
        torch.save(model.state_dict(),
                   '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
                                                                             args.with_reconstruction))
    writer.export_scalars_to_json("./test.json")
    writer.close()
