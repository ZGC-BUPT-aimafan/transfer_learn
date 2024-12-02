import torch
import torch.nn as nn


class DFNet(nn.Module):
    def __init__(self, input_shape, classes):
        super(DFNet, self).__init__()
        filter_num = ["None", 32, 64, 128, 256]
        kernel_size = ["None", 8, 8, 8, 8]
        conv_stride_size = ["None", 1, 1, 1, 1]
        pool_stride_size = ["None", 4, 4, 4, 4]
        pool_size = ["None", 8, 8, 8, 8]

        padding_size = 4

        self.block1_conv1 = nn.Conv1d(
            in_channels=input_shape[0],
            out_channels=filter_num[1],
            kernel_size=kernel_size[1],
            stride=conv_stride_size[1],
            padding=padding_size,
        )
        self.batch_norm = nn.BatchNorm1d(filter_num[1])
        self.block1_adv_act1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(
            in_channels=filter_num[1],
            out_channels=filter_num[1],
            kernel_size=kernel_size[1],
            stride=conv_stride_size[1],
            padding=padding_size,
        )
        self.batch_norm1 = nn.BatchNorm1d(filter_num[1])
        self.elu1 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(
            kernel_size=pool_size[1], stride=pool_stride_size[1], padding=padding_size
        )
        self.dropout1 = nn.Dropout(0.1)

        self.block2_conv1 = nn.Conv1d(
            in_channels=filter_num[1],
            out_channels=filter_num[2],
            kernel_size=kernel_size[2],
            stride=conv_stride_size[2],
            padding=padding_size,
        )
        self.batch_norm2 = nn.BatchNorm1d(filter_num[2])
        self.relu1 = nn.ReLU()

        self.block2_conv2 = nn.Conv1d(
            in_channels=filter_num[2],
            out_channels=filter_num[2],
            kernel_size=kernel_size[2],
            stride=conv_stride_size[2],
            padding=padding_size,
        )
        self.batch_norm3 = nn.BatchNorm1d(filter_num[2])
        self.relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(
            kernel_size=pool_size[2], stride=pool_stride_size[2], padding=padding_size
        )
        self.dropout2 = nn.Dropout(0.1)

        self.block3_conv1 = nn.Conv1d(
            in_channels=filter_num[2],
            out_channels=filter_num[3],
            kernel_size=kernel_size[3],
            stride=conv_stride_size[3],
            padding=padding_size,
        )
        self.batch_norm4 = nn.BatchNorm1d(filter_num[3])
        self.relu3 = nn.ReLU()

        self.block3_conv2 = nn.Conv1d(
            in_channels=filter_num[3],
            out_channels=filter_num[3],
            kernel_size=kernel_size[3],
            stride=conv_stride_size[3],
            padding=padding_size,
        )
        self.batch_norm5 = nn.BatchNorm1d(filter_num[3])
        self.relu4 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(
            kernel_size=pool_size[3], stride=pool_stride_size[3], padding=padding_size
        )
        self.dropout3 = nn.Dropout(0.1)

        self.block4_conv1 = nn.Conv1d(
            in_channels=filter_num[3],
            out_channels=filter_num[4],
            kernel_size=kernel_size[4],
            stride=conv_stride_size[4],
            padding=padding_size,
        )
        self.batch_norm6 = nn.BatchNorm1d(filter_num[4])
        self.relu5 = nn.ReLU()

        self.block4_conv2 = nn.Conv1d(
            in_channels=filter_num[4],
            out_channels=filter_num[4],
            kernel_size=kernel_size[4],
            stride=conv_stride_size[4],
            padding=padding_size,
        )
        self.batch_norm7 = nn.BatchNorm1d(filter_num[4])
        self.relu6 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(
            kernel_size=pool_size[4], stride=pool_stride_size[4], padding=padding_size
        )
        self.dropout4 = nn.Dropout(0.1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filter_num[4], 512)
        self.batch_norm8 = nn.BatchNorm1d(512)
        self.relu7 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(512, 512)
        self.batch_norm9 = nn.BatchNorm1d(512)
        self.relu8 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1_conv1(x)
        x = self.batch_norm(x)
        x = self.block1_adv_act1(x)
        x = self.block1_conv2(x)
        x = self.batch_norm1(x)
        x = self.elu1(x)
        x = self.block1_pool(x)
        x = self.dropout1(x)

        x = self.block2_conv1(x)
        x = self.batch_norm2(x)
        x = self.relu1(x)

        x = self.block2_conv2(x)
        x = self.batch_norm3(x)
        x = self.relu2(x)
        x = self.block2_pool(x)
        x = self.dropout2(x)

        x = self.block3_conv1(x)
        x = self.batch_norm4(x)
        x = self.relu3(x)

        x = self.block3_conv2(x)
        x = self.batch_norm5(x)
        x = self.relu4(x)
        x = self.block3_pool(x)
        x = self.dropout3(x)

        x = self.block4_conv1(x)
        x = self.batch_norm6(x)
        x = self.relu5(x)

        x = self.block4_conv2(x)
        x = self.batch_norm7(x)
        x = self.relu6(x)
        x = self.block4_pool(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batch_norm8(x)
        x = self.relu7(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        x = self.batch_norm9(x)
        x = self.relu8(x)
        x = self.dropout6(x)

        x = self.fc3(x)
        # x = self.softmax(x)
        return x
