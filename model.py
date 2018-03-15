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
        self.model_architecture = params.model_architecture

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

        model_architecture = self.model_architecture.upper()
        if model_architecture == 'A':
            self.build_a()
        elif model_architecture == 'B':
            self.build_b()
        elif model_architecture == 'C':
            self.build_c()
        elif model_architecture == 'D':
            self.build_d()

    def build_a(self):
        self.fc_1_scene = nn.Linear(2 * self.pre_learned_output_dim, 500)
        self.fcbn_1_scene = nn.BatchNorm1d(500)
        self.fc_1_obj = nn.Linear(2 * self.pre_learned_output_dim, 500)
        self.fcbn_1_obj = nn.BatchNorm1d(500)
        self.fc_2 = nn.Linear(1000, 128)
        self.fcbn_2 = nn.BatchNorm1d(128)
        self.fc_3= nn.Linear(128, 6)
    def forward_a(self, scene_encoding, obj_encoding):
        layer_1_scene = F.dropout(F.relu(self.fcbn_1_scene(self.fc_1_scene(scene_encoding))), p=self.dropout_rate, training=self.training)
        layer_1_obj = F.dropout(F.relu(self.fcbn_1_obj(self.fc_1_obj(obj_encoding))), p=self.dropout_rate, training=self.training)
        layer_2_input = torch.cat((layer_1_scene, layer_1_obj), -1)
        out = F.dropout(F.relu(F.relu(self.fcbn_2(self.fc_2(layer_2_input)))), p=self.dropout_rate, training=self.training)
        return self.fc_3(out)

    def build_b(self):
        self.fc_1 = nn.Linear(4 * self.pre_learned_output_dim, 1000)
        self.fcbn_1 = nn.BatchNorm1d(1000)
        self.fc_2 = nn.Linear(1000, 500)
        self.fcbn_2 = nn.BatchNorm1d(500)
        self.fc_3 = nn.Linear(500, 128)
        self.fcbn_3 = nn.BatchNorm1d(128)
        self.fc_4 = nn.Linear(128, 6)
    def forward_b(self, scene_encoding, obj_encoding):
        stacked_input = torch.cat((scene_encoding, obj_encoding), -1)
        out = F.dropout(F.relu(self.fcbn_3(self.fc_3(F.relu(self.fcbn_2(self.fc_2(F.relu(self.fcbn_1(self.fc_1(stacked_input))))))))), p=self.dropout_rate, training=self.training)
        out = self.fc_4(out)
        return out

    def build_c(self):
        self.fc_1_scene = nn.Linear(2 * self.pre_learned_output_dim, 500)
        self.fcbn_1_scene = nn.BatchNorm1d(500)
        self.fc_1_obj = nn.Linear(2 * self.pre_learned_output_dim, 500)
        self.fcbn_1_obj = nn.BatchNorm1d(500)
        self.fc_2_scene = nn.Linear(500, 500)
        self.fcbn_2_scene = nn.BatchNorm1d(500)
        self.fc_3 = nn.Linear(1000, 128)
        self.fcbn_3 = nn.BatchNorm1d(128)
        self.fc_4 = nn.Linear(128, 6)
    def forward_c(self, scene_encoding, obj_encoding):
        layer_1_scene = F.dropout(F.relu(self.fcbn_1_scene(self.fc_1_scene(scene_encoding))), p=self.dropout_rate, training=self.training)
        layer_1_obj = F.dropout(F.relu(self.fcbn_1_obj(self.fc_1_obj(obj_encoding))), p=self.dropout_rate, training=self.training)
        layer_2_scene = F.dropout(F.relu(self.fcbn_2_scene(self.fc_2_scene(layer_1_scene))), p=self.dropout_rate, training=self.training)
        layer_3_input = torch.cat((layer_2_scene, layer_1_obj), -1)
        out = F.dropout(F.relu(self.fcbn_3(self.fc_3(layer_3_input))), p=self.dropout_rate, training=self.training)
        return self.fc_4(out)

    def build_d(self):
        self.fc_1_scene = nn.Linear(2 * self.pre_learned_output_dim, 500)
        self.fcbn_1_scene = nn.BatchNorm1d(500)
        self.fc_1_obj = nn.Linear(2 * self.pre_learned_output_dim, 500)
        self.fcbn_1_obj = nn.BatchNorm1d(500)
        self.fc_2 = nn.Linear(1000, 500)
        self.fcbn_2 = nn.BatchNorm1d(500)
        self.fc_3 = nn.Linear(500, 128)
        self.fcbn_3 = nn.BatchNorm1d(128)
        self.fc_4 = nn.Linear(128, 6)
    def forward_d(self, scene_encoding, obj_encoding):
        layer_1_scene = F.dropout(F.relu(self.fcbn_1_scene(self.fc_1_scene(scene_encoding))), p=self.dropout_rate, training=self.training)
        layer_1_obj = F.dropout(F.relu(self.fcbn_1_obj(self.fc_1_obj(obj_encoding))), p=self.dropout_rate, training=self.training)
        layer_2_input = torch.cat((layer_1_scene, layer_1_obj), -1)
        out = F.dropout(F.relu(self.fcbn_3(self.fc_3(F.relu(self.fcbn_2(self.fc_2(layer_2_input)))))), p=self.dropout_rate, training=self.training)
        return self.fc_4(out)

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

        scene_rgb_encoding = self.apply_transfer_learning(s[:, :3, :, :])
        scene_d_encoding = self.apply_transfer_learning(s[:, 3:6, :, :])
        obj_rgb_encoding = self.apply_transfer_learning(s[:, 6:9, :, :])
        obj_d_encoding = self.apply_transfer_learning(s[:, 9:, :, :])

        # Concatenate encodings
        scene_encoding = Variable(torch.cat((scene_rgb_encoding, scene_d_encoding), -1))
        obj_encoding = Variable(torch.cat((obj_rgb_encoding, obj_d_encoding), -1))
        model_architecture = self.model_architecture.upper()
        if model_architecture == 'A':
            return self.forward_a(scene_encoding, obj_encoding)
        elif model_architecture == 'B':
            return self.forward_b(scene_encoding, obj_encoding)
        elif model_architecture == 'C':
            return self.forward_c(scene_encoding, obj_encoding)
        elif model_architecture == 'D':
            return self.forward_d(scene_encoding, obj_encoding)
        else:
            # Create ResNet encodings for each input
            scene_rgb_encoding = self.apply_transfer_learning(s[:, :3, :, :])
            scene_d_encoding = self.apply_transfer_learning(s[:, 3:6, :, :])
            obj_rgb_encoding = self.apply_transfer_learning(s[:, 6:9, :, :])
            obj_d_encoding = self.apply_transfer_learning(s[:, 9:, :, :])

            # Concatenate encodings
            s = Variable(torch.cat((scene_rgb_encoding, scene_d_encoding, obj_rgb_encoding, obj_d_encoding), -1))
            # apply 2 fully connected layers with dropout
            s = F.dropout(F.relu(self.fcbn2(self.fc2(F.relu(self.fcbn1(self.fc1(s)))))),
                          p=self.dropout_rate, training=self.training)
            s = self.fc3(s)
            return s


def loss_fn(outputs, labels):
    position_loss_coefficient = 100 #placeholder
    position_loss = torch.mean(torch.sqrt((outputs[:, :3] - labels[:, :3]).pow(2)))
    position_loss = position_loss * position_loss_coefficient

    pose_loss = torch.mean(torch.sqrt((outputs[:, 3:] - labels[:, 3:]).pow(2)))
    return position_loss + pose_loss

def old_loss_fn(outputs, labels):
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
    zero = Variable(torch.FloatTensor([0]))
    num_examples = outputs.size()[0]
    pos_threshold = 5
    pose_threshold = 15
    pos_loss = torch.sum((torch.clamp(torch.sqrt(torch.sum((outputs[:, :3] - labels[:, :3]).pow(2), dim=1)) - pos_threshold, min=0)).type(torch.FloatTensor), dim=-1)
    pose_loss = torch.sum((torch.clamp(torch.sqrt(torch.sum((outputs[:, 3:] - labels[:, 3:]).pow(2), dim=1)) - pose_threshold, min=0)).type(torch.FloatTensor), dim=-1)
    total_loss = torch.sum(pos_loss + pose_loss) / num_examples
    return total_loss

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


def position_accuracy(outputs, labels):
    num_examples = len(labels)
    labels = np.squeeze(labels)
    pos_threshold = .05
    pos_loss = np.sum(
        np.clip((np.sqrt(np.sum(np.power((outputs[:, :3] - labels[:, :3]), 2), axis=-1)) - pos_threshold), a_min=0, a_max=1), axis=-1)
    return 1.0 - (pos_loss / float(num_examples))


def pose_accuracy(outputs, labels):
    num_examples = len(labels)
    labels = np.squeeze(labels)
    pose_threshold = 15
    pose_loss = np.sum(
        np.clip((np.sqrt(np.sum(np.power((outputs[:, 3:] - labels[:, 3:]), 2), axis=-1)) - pose_threshold), a_min=0, a_max=1), axis=-1)
    return 1.0 - (pose_loss / float(num_examples))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'position accuracy': position_accuracy,
    'pose accuracy': pose_accuracy,
    # could add more metrics such as accuracy for each token type
}
