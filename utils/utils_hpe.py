import copy
import os

import numpy as np
import torch
from torch.autograd import Variable


def optimize_scaling_factor(
    cam_cs_hat,
    img_cs_hat,
    epochs=200,
    learningRate=0.00005,
    stop_tolerance=0.000001,
    gpus="0, 1",
):

    # cam_cs_hat, img_cs_hat: (17, 3)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
    class linearRegression(torch.nn.Module):
        def __init__(self, inputSize, outputSize):
            super(linearRegression, self).__init__()
            self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

        def forward(self, x):
            out = self.linear(x)
            return out

    x_train = copy.deepcopy(
        cam_cs_hat.reshape(-1, 1).astype(np.float32)
    )  # 모든 점을 batch로 취급
    y_train = copy.deepcopy(
        img_cs_hat.reshape(-1, 1).astype(np.float32)
    )  # 모든 점을 batch로 취급

    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'

    model = linearRegression(inputDim, outputDim)
    # model.linear.weight.data[0] = 0.25

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    losses = []
    weights = []
    tol_cnt = 0
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # loss.requires_grad = True
        losses.append(loss.item())
        # print('epoch {}, loss {}'.format(epoch, loss.item()))

        # if loss is not decreasing, stop training
        if epoch > 1:
            if abs(losses[-1] - losses[-2]) < stop_tolerance:
                tol_cnt += 1
                if tol_cnt > 5:
                    break
            else:
                tol_cnt = 0

        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        # weights.append(model.linear.weight.data.item())

        # print('epoch {}, loss {}'.format(epoch, loss.item()))
    return model.linear.weight.data.item(), losses
