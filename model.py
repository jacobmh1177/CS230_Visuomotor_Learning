"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable



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
        self.pre_learned_output_dim = params.pre_learned_output_dim

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4 * self.pre_learned_output_dim, 1000)
        self.fcbn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fcbn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 6)
        self.dropout_rate = params.dropout_rate

        self.pre_trained_model = models.resnet18(pretrained=True)#models.inception_v3(pretrained=True)
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False
        num_ftrs = self.pre_trained_model.fc.in_features
        self.pre_trained_model.fc = nn.Linear(num_ftrs, self.pre_learned_output_dim)

    def apply_transfer_learning(self, s):
        return self.pre_trained_model(s).data


    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """

        # Create ResNet encodings for each input
        scene_rgb_encoding = self.apply_transfer_learning(s[:, :3, :, :])
        scene_d_encoding = self.apply_transfer_learning(s[:, 3:6, :, :])
        obj_rgb_encoding = self.apply_transfer_learning(s[:, 6:9, :, :])
        obj_d_encoding = self.apply_transfer_learning(s[:, 9:, :, :])

        # Concatenate encodings
        #s = s.view(-1, 4 * self.pre_learned_output_dim)
        s = Variable(torch.cat((scene_rgb_encoding, scene_d_encoding, obj_rgb_encoding, obj_d_encoding), -1))
        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn2(self.fc2(F.relu(self.fcbn1(self.fc1(s)))))),
                      p=self.dropout_rate, training=self.training)
        # s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
        #               p=self.dropout_rate, training=self.training)
        s = self.fc3(s)
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
    #return torch.sqrt(torch.mean((outputs - labels).pow(2)))
    zero = torch.Tensor([0])
    zero = Variable(zero)
    pos_threshold = 5
    pose_threshold = 15
    pos_loss = torch.mean(torch.abs(outputs[:3] - labels[:3])) - pos_threshold
    pose_loss = torch.mean(torch.abs(outputs[3:] - labels[3:])) - pose_threshold
    total_loss = pos_loss + pose_loss
    return torch.max(zero, total_loss)

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
    return np.sqrt(np.mean(np.power((position_output - position_label), 2)))


def pose_error(outputs, labels):
    labels = np.squeeze(labels)
    pose_output = outputs[3:]
    pose_label = labels[3:]
    return np.sqrt(np.mean(np.power((pose_output - pose_label), 2)))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'position error': pos_error,
    'pose error': pose_error,
    # could add more metrics such as accuracy for each token type
}
