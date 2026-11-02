import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MLP(nn.Module):
    def __init__(self, num_classes=10, hidden=(100, 100, 100)):
        super().__init__()
        layers, width = [], 3 * 28 * 28
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, num_classes)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pad = planes // 4 if (stride != 1 or in_planes != planes) else 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad)) if self.pad else x
        return F.relu(out + shortcut)


class ResNet20(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        in_planes, stages = 16, []
        for planes, stride in ((16, 1), (32, 2), (64, 2)):
            blocks = []
            for s in (stride, 1, 1):
                blocks.append(BasicBlock(in_planes, planes, s))
                in_planes = planes
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3 = stages
        self.linear = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)


class Net(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


class MLPBody(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.feature = mlp.feature

    def forward(self, x):
        return self.feature(x.reshape(x.size(0), -1) / 255)


class ResNet20Body(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.conv1, self.bn1 = net.conv1, net.bn1
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.avg_pool2d(out, out.size()[3])
        return out.view(out.size(0), -1)


def resnet18_net(num_classes):
    net = torchvision.models.resnet18(weights=None)
    head = nn.Linear(net.fc.in_features, num_classes)
    net.fc = nn.Identity()
    return Net(net, head)


def build_model(benchmark, num_classes):
    if benchmark.startswith("cmnist"):
        mlp = MLP(num_classes=num_classes)
        return Net(MLPBody(mlp), mlp.classifier)
    if benchmark.startswith("cifar10c"):
        net = ResNet20(num_classes)
        return Net(ResNet20Body(net), net.linear)
    if benchmark.startswith("bffhq"):
        return resnet18_net(num_classes)
    raise ValueError(benchmark)


def ldd_encoder(benchmark):
    if benchmark.startswith("cmnist"):
        return MLPBody(MLP(hidden=(100, 100, 16))), 16
    net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Identity()
    return net, 512


class LDDNet(nn.Module):
    def __init__(self, benchmark, num_classes):
        super().__init__()
        self.body_i, dim = ldd_encoder(benchmark)
        self.body_b, _ = ldd_encoder(benchmark)
        self.head = nn.Linear(2 * dim, num_classes)
        self.head_b = nn.Linear(2 * dim, num_classes)

    def encode(self, x):
        return self.body_i(x), self.body_b(x)

    def features(self, x):
        return torch.cat(self.encode(x), 1)

    def forward(self, x):
        return self.head(self.features(x))


def network_for(method, benchmark, num_classes):
    if method == "ldd":
        return LDDNet(benchmark, num_classes)
    return build_model(benchmark, num_classes)
