import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from transfer_learn.df.model_nodef import DFNet
from transfer_learn.myutils.logger import logger


def train(
    X_train,
    X_test,
    X_valid,
    y_train,
    y_test,
    y_valid,
    modelpath,
    num_epochs,
):
    random.seed(0)
    # 检测是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LENGTH = X_train.shape[1]
    print(f"LENGTH is {LENGTH}")
    # 优化器
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08
    decay = 0.0

    NB_CLASSES = 15
    print("NB_CLASSES IS ", NB_CLASSES)

    # Convert data as float32 type
    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")
    X_test = X_test.astype("float32")

    # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
    X_train = X_train[:, :, np.newaxis]
    X_valid = X_valid[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    print(X_train.shape[0], "train samples")
    print(X_valid.shape[0], "validation samples")
    print(X_test.shape[0], "test samples")

    # Convert class vectors to categorical classes matrices

    y_train = torch.tensor(y_train)
    y_valid = torch.tensor(y_valid)
    y_test = torch.tensor(y_test)
    y_train = F.one_hot(y_train, num_classes=NB_CLASSES).float()
    y_valid = F.one_hot(y_valid, num_classes=NB_CLASSES).float()
    y_test = F.one_hot(y_test, num_classes=NB_CLASSES).float()

    INPUT_SHAPE = (LENGTH, 1)

    model = DFNet(input_shape=INPUT_SHAPE, classes=NB_CLASSES).to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=decay,
    )

    # 转换为 Torch 的 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = y_train.clone().detach().to(device)

    # 创建 TensorDataset 对象
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # 定义批次大小
    batch_size = 128

    # 创建训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 转换为 Torch 的 Tensor
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid_tensor = y_valid.clone().detach().to(device)

    # 创建 TensorDataset 对象
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

    # 定义批次大小
    batch_size = 32

    # 创建训练数据加载器
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        total_samples = 0
        # correct_predictions = torch.zeros(32)
        correct_predictions = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            tensor_labels = torch.argmax(labels, dim=1)

            correct_predictions += (predicted == tensor_labels).sum().item()

        train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples

        # 在验证集上评估模型
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_correct = 0
            total_valid_samples = 0
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                tensor_labels = torch.argmax(labels, dim=1)
                valid_correct += (predicted == tensor_labels).sum().item()
                total_valid_samples += labels.size(0)

            valid_loss /= total_valid_samples
            valid_accuracy = valid_correct / total_valid_samples

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss} - Train Accuracy: {train_accuracy:.4f} - Valid Loss: {valid_loss:.4f} - Valid Accuracy: {valid_accuracy:.4f}"
        )

    logger.info(modelpath)
    torch.save(model.state_dict(), modelpath)

    # 加载模型时需要重新指定设备
    # model.load_state_dict(torch.load(modelpath, map_location=device))

    return valid_accuracy


def verification(model_path, X_valid, y_valid, num_classes=15):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFNet(input_shape=(X_valid.shape[1], 1), classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置模型为评估模式
    # 定义批次大小
    batch_size = 32
    # 转换为 float32 类型并增加额外的维度（如果需要）
    X_valid = X_valid.astype("float32")
    X_valid = X_valid[:, :, np.newaxis]  # 适应模型的输入形状

    # 转换标签为 one-hot 编码
    y_valid = torch.tensor(y_valid)
    y_valid = F.one_hot(y_valid, num_classes=num_classes).float()

    # 将数据转换为 Tensor，并移动到正确的设备
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid_tensor = y_valid.clone().detach().to(device)

    # 创建 DataLoader
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0

    # 在验证集上进行评估
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            tensor_labels = torch.argmax(labels, dim=1)
            correct_predictions += (predicted == tensor_labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    logger.info(
        f"Validation Loss: {avg_loss:.4f} - Validation Accuracy: {accuracy:.4f}"
    )

    return accuracy
