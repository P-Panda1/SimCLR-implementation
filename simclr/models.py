"""
Model architectures for SimCLR: encoder and projection head.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from .config import Config


class ProjectionHead(nn.Module):
    """
    MLP projection head that maps encoder representations to projection space.
    Default: 512 -> 512 -> 128 (projection_dim)
    """

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=None):
        super(ProjectionHead, self).__init__()
        if output_dim is None:
            output_dim = Config.PROJECTION_DIM
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class SimCLRModel(nn.Module):
    """
    Complete SimCLR model: ResNet-18 encoder + MLP projection head.
    ResNet-18 is initialized from scratch (no pretrained weights).
    """

    def __init__(self, projection_dim=None):
        super(SimCLRModel, self).__init__()
        if projection_dim is None:
            projection_dim = Config.PROJECTION_DIM

        # ResNet-18 encoder (initialized from scratch)
        encoder = resnet18(pretrained=False)
        # Modify first conv layer for CIFAR-10 (32x32 images)
        encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.maxpool = nn.Identity()  # Remove maxpool for small images
        # Remove final fully connected layer (we only need features)
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        # Get encoder output dimension (512 for ResNet-18)
        self.encoder_dim = 512
        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=self.encoder_dim,
            output_dim=projection_dim
        )

    def forward(self, x, return_features=False):
        """
        Forward pass: encode and project.
        If return_features=True, also return encoder features (for linear probe).
        """
        # Encode
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        # Project
        projection = self.projection_head(features)
        if return_features:
            return projection, features
        return projection


class LinearProbe(nn.Module):
    """
    Linear classifier for evaluating learned representations.
    Freezes encoder and trains only a linear layer.
    """

    def __init__(self, encoder, num_classes=10):
        super(LinearProbe, self).__init__()
        self.encoder = encoder
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Learnable linear classifier
        # 512 = ResNet-18 feature dim
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Do not wrap the encoder forward in torch.no_grad() here.
        # The encoder weights are frozen by setting requires_grad=False on its
        # parameters in __init__, which prevents parameter updates but still
        # allows gradients to be computed with respect to the input. This
        # enables explainability methods (e.g., Integrated Gradients) that
        # require gradients w.r.t. the input image.
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
