import torch
import torch.nn as nn
import torch.nn.functional as F

class SummarizingModel(nn.Module):


    def __init__(self, input_image_width_and_height, num_channels=128):
        super(SummarizingModel, self).__init__()

        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        self.group_norm1 = nn.GroupNorm(128, 128, affine=True)
        self.group_norm2 = nn.GroupNorm(128, 128, affine=True)
        self.group_norm3 = nn.GroupNorm(128, 128, affine=True)

        self.pool = nn.AvgPool2d(2, 2)
        self.fc = None

        self.input_image_width_and_height = input_image_width_and_height

        self.in_features_fc = self._get_conv_output_len()

    def embedding(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.group_norm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.group_norm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.group_norm3(x)
        x = F.relu(x)
        x = self.pool(x)

        embedding = x.view(batch_size, -1)
        return embedding

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.group_norm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.group_norm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.group_norm3(x)
        x = F.relu(x)
        x = self.pool(x)

        embedding = x.view(batch_size, -1)
        output = self.fc(embedding)
        return embedding, output


    def _get_conv_output_len(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, *self.input_image_width_and_height)
            x = self.conv1(x)
            x = self.group_norm1(x)
            x = F.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.group_norm2(x)
            x = F.relu(x)
            x = self.pool(x)

            x = self.conv3(x)
            x = self.group_norm3(x)
            x = F.relu(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)

    def add_classes(self, num_classes):

        if self.fc is None:
            self.fc = nn.Linear(self.in_features_fc, num_classes)
        else:
            new_fc = nn.Linear(self.fc.in_features, self.fc.out_features + num_classes)

            new_fc.weight.data[:-num_classes, :] = self.fc.weight.data
            new_fc.bias.data[:-num_classes] = self.fc.bias.data

            self.fc = new_fc