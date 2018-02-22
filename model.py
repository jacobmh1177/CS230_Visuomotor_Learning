"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_convolved_image_width(input_width, filter_width, stride, padding):
    return int(np.floor(1 + (input_width + 2 * padding - filter_width) / stride))


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels

        # # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # # stride, padding). We also include batch normalisation layers that help stabilise training.
        # # For more details on how to use these layers, check out the documentation.
        # self.conv1_s = nn.Conv2d(4, self.num_channels, 3, stride=1, padding=1)
        # self.bn1_s = nn.BatchNorm2d(self.num_channels)
        # self.conv1_o = nn.Conv2d(4, self.num_channels, 3, stride=1, padding=1)
        # self.bn1_o = nn.BatchNorm2d(self.num_channels)
        # self.conv2_s = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        # self.bn2_s = nn.BatchNorm2d(self.num_channels * 2)
        # self.conv2_o = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        # self.bn2_o = nn.BatchNorm2d(self.num_channels * 2)
        # self.conv3_s = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        # self.bn3_s = nn.BatchNorm2d(self.num_channels * 4)
        # self.conv3_o = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        # self.bn3_o = nn.BatchNorm2d(self.num_channels * 4)
        #
        # # 2 fully connected layers to transform the output of the convolution layers to the final output
        # self.fc1 = nn.Linear(2 * 8 * 8 * self.num_channels * 4, self.num_channels * 4)
        # self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4)
        # self.fc2 = nn.Linear(self.num_channels * 4, 6)
        # self.dropout_rate = params.dropout_rate


        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(8, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(37*37*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)
        self.dropout_rate = params.dropout_rate
    
    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        # # Split input into scene and object
        # s = data[:, :4, :, :]
        # o = data[:, 4:, :, :]
        # #                                                  -> batch_size x 3 x 64 x 64
        # # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        # s = self.bn1_s(self.conv1_s(s))  # batch_size x num_channels x 64 x 64
        # output_width = compute_convolved_image_width(299, 3, 1, 1)
        # s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 32 x 32
        # output_width = compute_convolved_image_width(output_width, 2, 2, 0)
        # s = self.bn2_s(self.conv2_s(s))  # batch_size x num_channels*2 x 32 x 32
        # output_width = compute_convolved_image_width(output_width, 3, 1, 1)
        # s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 16 x 16
        # output_width = compute_convolved_image_width(output_width, 2, 2, 0)
        # s = self.bn3_s(self.conv3_s(s))  # batch_size x num_channels*4 x 16 x 16
        # output_width = compute_convolved_image_width(output_width, 3, 1, 1)
        # s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 8 x 8
        # output_width = compute_convolved_image_width(output_width, 2, 2, 0)
        #
        # # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        # o = self.bn1_o(self.conv1_o(o))  # batch_size x num_channels x 64 x 64
        # o = F.relu(F.max_pool2d(o, 2))  # batch_size x num_channels x 32 x 32
        # o = self.bn2_o(self.conv2_o(o))  # batch_size x num_channels*2 x 32 x 32
        # o = F.relu(F.max_pool2d(o, 2))  # batch_size x num_channels*2 x 16 x 16
        # o = self.bn3_o(self.conv3_o(o))  # batch_size x num_channels*4 x 16 x 16
        # o = F.relu(F.max_pool2d(o, 2))  # batch_size x num_channels*4 x 8 x 8
        #
        # # flatten the output for each image
        # s = s.view(-1, output_width * output_width * self.num_channels * 4)  # batch_size x 8*8*num_channels*4
        # o = o.view(-1, output_width * output_width * self.num_channels * 4)
        # t = np.concatenate((s, o), axis=1)
        # # apply 2 fully connected layers with dropout
        # t = F.dropout(F.relu(self.fcbn1(self.fc1(t))),
        #               p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        # t = self.fc2(t)  # batch_size x 6
        #
        # # apply log softmax on each image's output (this is recommended over applying softmax
        # # since it is numerically more stable)
        s = torch.squeeze(s)
        #print("Input shpae = {}".format(s.shape))
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 64 x 64
        output_width = compute_convolved_image_width(299, 3, 1, 1)
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 32 x 32
        output_width = compute_convolved_image_width(output_width, 2, 2, 0)
        s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 32 x 32
        output_width = compute_convolved_image_width(output_width, 3, 1, 1)
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 16 x 16
        output_width = compute_convolved_image_width(output_width, 2, 2, 0)
        s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 16 x 16
        output_width = compute_convolved_image_width(output_width, 3, 1, 1)
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 8 x 8
        output_width = compute_convolved_image_width(output_width, 2, 2, 0)
        #print ("Output_width = {}".format(output_width))

        # flatten the output for each image
        s = s.view(-1, output_width * output_width * self.num_channels * 4)  # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        s = self.fc2(s)  # batch_size x 6
        return s


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    # num_examples = outputs.size()[0]
    # return -torch.sum(outputs[range(num_examples), labels]) / num_examples
    return torch.sqrt(torch.mean((outputs - labels).pow(2)))


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def pos_error(outputs, labels):
    labels = np.squeeze(labels)
    # print("Evaluation- Output shape: {}".format(outputs.shape))
    # print("Evaluation- Labels shape: {}".format(labels.shape))
    position_output = outputs[:3]
    position_label = labels[:3]
    return np.sqrt(np.mean(np.power((position_output-position_label), 2)))

def pose_error(outputs, labels):
    labels = np.squeeze(labels)
    pose_output = outputs[3:]
    pose_label = labels[3:]
    return np.sqrt(np.mean(np.power((pose_output-pose_label), 2)))
# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'position error': pos_error,
    'pose error': pose_error,
    # could add more metrics such as accuracy for each token type
}
