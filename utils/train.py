import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import time

from utils.estimates import estimates_function_runtime


def __parse_dictionary(dictionary):
    """
    make dictionary to string as following format
    'key: value\n'
    """
    result = ''
    for key, value in dictionary.items():
        result += '%s: %s\n' % (str(key), str(value))
    return result


class ModelTrainer:

    def __init__(self,
                 model,
                 data_loader,
                 hyperparameters,
                 settings,
                 miscs=None):

        self.model = model
        self.data_loader = data_loader

        # hyperparameters
        self.params = hyperparameters
        self.params.setdefault('batch_size',
                               self.data_loader['test'].batch_size)
        self.params.setdefault('learning_rate', 1e-3)
        self.params.setdefault('num_epochs', 5)
        self.params.setdefault('weight_decay', 0)
        self.params.setdefault(
            'optimizer',
            optim.Adam(
                self.model.parameters(),
                lr=self.params['learning_rate'],
                weight_decay=self.params['weight_decay']))
        self.params.setdefault('loss_function', nn.CrossEntropyLoss())

        # other settings
        self.settings = settings
        self.settings.setdefault('print_every', 100)
        self.settings.setdefault('verbose', True)
        self.settings.setdefault('save_log', False)
        self.settings.setdefault('save_model', 0)
        self.settings.setdefault('start_epoch', 1)
        self.settings.setdefault('name', self.model.__class__.__name__)
        self.settings.setdefault(
            'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # something else to save in log
        # must be dictionary or None
        self.miscs = miscs

        # other base settings
        self.folder_name = None
        if self.settings['save_log']:
            current_time = time.ctime()
            current_time = current_time.split(' ')
            seconds = re.sub(r':', '', current_time[3])
            self.folder_name = os.path.join(
                './log', '%s-%s-%s-%s' % (current_time[4], current_time[1],
                                          current_time[2], seconds))

            if not os.path.exists(self.folder_name):
                os.mkdir(self.folder_name)

            self.__save_params(os.path.join(self.folder_name, 'params.txt'))

    def train(self):

        last_iter = 0
        optimizer = self.params['optimizer']
        loss_function = self.params['loss_function']

        verbose = self.settings['verbose']
        log = self.settings['save_log']
        save = self.settings['save_model']
        start_epoch = self.settings['start_epoch']

        total_train_time = 0
        best_acc = 0

        train_file = None
        if log:
            train_file = open(
                os.path.join(self.folder_name, 'train_log.txt'), 'w')
            test_file = open(
                os.path.join(self.folder_name, 'test_log.txt'), 'w')
        for epoch in range(start_epoch,
                           start_epoch + self.params['num_epochs']):
            elapsed_train_time, (train_loss, train_acc,
                                 last_iter) = self._train_single_epoch(
                                     optimizer, loss_function, last_iter,
                                     train_file)
            elapsed_test_time, (test_loss,
                                test_acc) = self._test_single_epoch()

            total_train_time += elapsed_train_time

            if test_acc > best_acc:
                best_acc = test_acc

            if verbose:
                print('Train epoch: %d [%f sec], loss = %f, accuracy = %f' %
                      (epoch, elapsed_train_time, train_loss, train_acc))
                print('Test epoch: %d [%f sec], loss = %f, accuracy = %f' %
                      (epoch, elapsed_test_time, test_loss, test_acc))

            if test_file is not None:
                test_file.write('%d %f %f\n' % (epoch, test_loss, test_acc))

            if save != 0 and epoch % save == 0:
                if not os.path.exists('./saved_models'):
                    os.mkdir('./saved_models')

                state = {
                    'state_dict': self.model.state_dict(),
                    'acc': test_acc,
                    'optimizer': optimizer.state_dict()
                }

                torch.save(
                    state,
                    os.path.join(
                        './saved_models',
                        '%s_%03d.pth.tar' % (self.settings['name'], epoch)))

        if log:
            train_file.write('%f\n' % total_train_time)
            train_file.close()
            test_file.write('%f\n' % self.best_acc)
            test_file.close()

    def test(self):
        elapsed_test_time, (test_loss, test_acc) = self._test_single_epoch()
        print('Test Result [%f sec elapsed]: loss = %f, accuracy = %f' %
              (elapsed_test_time, test_loss, test_acc))

    @estimates_function_runtime
    def _train_single_epoch(self, optimizer, loss_function, last_iter,
                            log_file):
        """
        Train single epoch
        this is for supervised learning problem
        """
        # change model status to train
        self.model.train()

        epoch_loss = 0
        epoch_correct = 0
        epoch_size = len(self.data_loader['train'])

        verbose = self.settings['verbose']
        print_every = self.settings['print_every']
        device = self.settings['device']

        for i, (images, labels) in enumerate(self.data_loader['train'],
                                             last_iter):
            images = images.to(device)
            labels = labels.to(device)

            real_batch_size = images.shape[0]

            outputs = self.model(images)

            _, predicts = torch.max(outputs, 1)
            corrects = torch.sum(predicts == labels)
            accuracy = corrects.double() / real_batch_size

            optimizer.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += (loss.item() * real_batch_size)
            epoch_correct += corrects.item()

            if verbose and i & print_every == 0:
                print('Iter: %d, loss = %f, accuracy = %f' % (i, loss,
                                                              accuracy))

            if log_file is not None:
                log_file.write('%d %f %f\n' % (i, loss, accuracy))

        return epoch_loss / epoch_size, epoch_correct / epoch_size, i

    @estimates_function_runtime
    def _test_single_epoch(self):
        # eval for one epoch
        self.model.eval()
        epoch_loss = 0
        epoch_correct = 0
        epoch_size = len(self.data_loader['test'])

        device = self.settings['device']
        for images, labels in self.data_loader['test']:
            images = images.to(device)
            labels = labels.to(device)

            real_batch_size = images.shape[0]

            outputs = self.model(images)
            _, predicts = torch.max(outputs, 1)
            corrects = torch.sum(predicts == labels)

            loss = self.loss_function(outputs, labels)

            epoch_loss += (loss.item() * real_batch_size)
            epoch_correct += corrects.item()

        return epoch_loss / epoch_size, epoch_correct / epoch_size

    def __save_params(self, filepath):
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        param_file = open(filepath, 'w')

        param_file.write('Hyperparameters\n')
        param_file.write(__parse_dictionary(self.params))
        param_file.write('Other Settings\n')
        param_file.write(__parse_dictionary(self.settings))
        if self.miscs is not None:
            param_file.write('Miscellaneous\n')
            param_file.write(__parse_dictionary(self.miscs))
