"""Train the model"""
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from tqdm import tqdm

import utils
import model as net
import data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, pre_trained_model, optimizer, loss_fn, dataloader, metrics, params, epoch):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            train_batch = torch.squeeze(train_batch)
            labels_batch = torch.squeeze(labels_batch)
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            train_batch = Variable(utils.apply_pre_trained_model(train_batch, pre_trained_model), requires_grad=True)
            # compute model output and loss
            #print("Forward propagating")
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            #print("Done forward propagating")

            # clear previous gradients, compute gradients of all variables wrt loss
            #print("Backward propagating")
            optimizer.zero_grad()
            loss.backward()
            #print("Done backward propagating")

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            train_batch = None
            labels_batch = None

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    csv_string = ",".join("{:05.3f}".format(v) for v in metrics_mean.values())
    if epoch == 0:
        with open(params.viz_file, 'w') as f:
            f.write(",".join("{}".format(k) for k in metrics_mean.keys()))
            f.write("\n")
    with open(params.viz_file, 'a') as f:
        f.write(csv_string)
        f.write("\n")
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, pre_trained_model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, pre_trained_model, optimizer, loss_fn, train_dataloader, metrics, params, epoch)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, pre_trained_model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['position error']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    #params.cuda = False

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        print('Using CUDA.')
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    pre_trained_model = models.resnet18(pretrained=True)
    if torch.cuda.is_available():
        pre_trained_model = torch.nn.DataParallel(pre_trained_model).cuda()
    for param in pre_trained_model.parameters():
        param.requires_grad = False
    train_and_evaluate(model, pre_trained_model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
