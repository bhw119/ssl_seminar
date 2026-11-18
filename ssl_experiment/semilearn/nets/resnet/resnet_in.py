import torch
import torch.nn as nn
import torchvision

class ResNet50IN(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet50(weights=weights)
        self.conv1, self.bn1, self.relu, self.maxpool = m.conv1, m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.avgpool = m.avgpool
        self.num_features = m.fc.in_features
        self.classifier = nn.Linear(self.num_features, num_classes)
        self.fusion_hook_module = self.layer4

    def extract(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.classifier(x)
        feat = self.extract(x)
        if only_feat:
            return feat
        return {'logits': self.classifier(feat), 'feat': feat}

    def group_matcher(self, coarse=False, prefix=''):
        return dict(stem=r'^{}conv1|^{}bn1|^{}maxpool'.format(prefix, prefix, prefix),
                    blocks=r'^{}layer(\d+)'.format(prefix) if coarse else r'^{}layer(\d+)\.(\d+)'.format(prefix))

    def no_weight_decay(self):
        return []

def resnet50_in(pretrained=True, pretrained_path=None, **kwargs):
    return ResNet50IN(pretrained=pretrained, **kwargs)
