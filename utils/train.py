import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time

from utils.estimates import estimates_function_runtime


class ModelTrainer:

    def __init__(self,
                 model,
                 data_loader,
                 hyperparameters,
                 settings,
                 miscs=None):
        # other settings
        self.settings = settings
        self.settings.setdefault('print_every', 100)
        self.settings.setdefault('verbose', True)
        self.settings.setdefault('save_log', False)
        self.settings.setdefault('save_model', 0)
        self.settings.setdefault('start_epoch', 1)
        self.settings.setdefault('name', model.__class__.__name__)
        self.settings.setdefault(
            'device',
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.prepare_model(model)
        self.data_loader = data_loader

        # hyperparameters
        self.params = hyperparameters
        self.params.setdefault('batch_size',
                               self.data_loader['test'].batch_size)
        self.params.setdefault('learning_rate', 1e-3)
        self.params.setdefault('num_epochs', 5)
        self.params.setdefault('weight_decay', 0)
        """
        # optimizers could be many
        self.params.setdefault(
            'optimizer',
            optim.Adam(
                self.model.parameters(),
                lr=self.params['learning_rate'],
                weight_decay=self.params['weight_decay']))

        # loss_function could be many
        self.params.setdefault('loss_function', nn.CrossEntropyLoss())
        """

        # something else to save in log
        # must be dictionary or None
        self.miscs = miscs

        # other settings
        self.folder_name = None
        self.files = None

    def train(self, test_also=True):

        optimizer = self.params['optimizer']

        verbose = self.settings['verbose']
        save_log = self.settings['save_log']
        save = self.settings['save_model']
        start_epoch = self.settings['start_epoch']

        if save_log:
            if not os.path.exists('./log'):
                os.mkdir('./log')

            current_time = time.ctime()
            current_time = current_time.split(' ')
            self.folder_name = os.path.join(
                './log', '%s-%s-%s-%s' % (current_time[4], current_time[1],
                                          current_time[2], current_time[3]))

            if not os.path.exists(self.folder_name):
                os.mkdir(self.folder_name)

            self.__open_files()
            self.__save_params()

        total_train_time = 0
        best_acc = 0

        last_train_iter = -1
        last_test_iter = -1

        for epoch in range(start_epoch,
                           start_epoch + self.params['num_epochs']):
            elapsed_train_time, (
                train_loss, train_acc,
                last_train_iter) = self._train_or_test_single_epoch(
                    last_train_iter, 'train')
            """
            elapsed_test_time, (
                test_loss, test_acc,
                last_test_iter) = self._train_or_test_single_epoch(
                    last_test_iter, 'test')
            """

            elapsed_test_time = 0
            test_loss = 0.
            test_acc = 0.

            if test_acc > best_acc:
                best_acc = test_acc

            total_train_time += elapsed_train_time

            if verbose:
                print('Train epoch: %d [%f sec], loss = %f, accuracy = %f' %
                      (epoch, elapsed_train_time, train_loss, train_acc))
                print('Test epoch: %d [%f sec], loss = %f, accuracy = %f' %
                      (epoch, elapsed_test_time, test_loss, test_acc))

            if save_log:
                self.files['train_epoch'].write(
                    '%d %f %f %f\n' % (epoch, elapsed_train_time, train_loss,
                                       train_acc))
                self.files['test_epoch'].write(
                    '%d %f %f %f\n' % (epoch, elapsed_test_time, test_loss,
                                       test_acc))

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

        if save_log:
            self.files['summary'].write(
                'total_train_time = %f\n' % total_train_time)
            self.files['summary'].write('best accuracy = %f\n' % best_acc)
            self.__close_files()

    def test(self):
        elapsed_test_time, (test_loss, test_acc,
                            _) = self._train_or_test_single_epoch(
                                -1, 'test')
        if self.settings['verbose']:
            print('Test Result [%f sec elapsed]: loss = %f, accuracy = %f' %
                  (elapsed_test_time, test_loss, test_acc))

    def prepare_model(self, models):
        """
        if not override,
        treate as single model
        """
        if self.settings['device'].type == 'cuda':
            self.model = nn.DataParallel(models)
            cudnn.benchmark = True
        else:
            self.model = models

    def update_optimizer(self, outputs, labels):
        """
        Must override!
        """
        raise NotImplementedError
        pass

    def calculate_loss(self, outputs, labels):
        """
        Must override!
        """
        raise NotImplementedError
        pass

    def calculate_predicted_labels(self, outputs):
        """
        if not overrided, just use the simplest thing
        """
        _, predicted = torch.max(outputs, 1)
        return predicted

    @estimates_function_runtime
    def _train_or_test_single_epoch(self, last_iter, mode):
        if mode not in ('train', 'test'):
            raise ValueError('Invalid Mode')

        is_train = (mode == 'train')
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0
        epoch_correct = 0
        epoch_size = 0

        verbose = self.settings['verbose']
        print_every = self.settings['print_every']
        device = self.settings['device']

        for i, (images, labels) in enumerate(self.data_loader[mode],
                                             last_iter + 1):
            images = images.to(device)
            labels = labels.to(device)

            real_batch_size = images.shape[0]

            outputs = self.model(images)

            predicts = self.calculate_predicted_labels(outputs)
            corrects = torch.sum(predicts == labels)
            accuracy = corrects.double() / real_batch_size

            if is_train:
                loss = self.update_optimizer(outputs, labels)
            else:
                loss = self.calculate_loss(outputs, labels)

            epoch_loss += (loss.item() * real_batch_size)
            epoch_correct += corrects.item()
            epoch_size += real_batch_size

            if is_train and verbose and i % print_every == 0:
                print('Iter: %d, loss = %f, accuracy = %f' % (i, loss,
                                                              accuracy))

            if self.files is not None:
                self.files['%s_iter' % mode].write(
                    '%d %f %f\n' % (i, loss, accuracy))

        return epoch_loss / epoch_size, epoch_correct / epoch_size, i

    def __open_files(self):
        self.files = {
            'summary':
                open(os.path.join(self.folder_name, 'summary.txt'), 'w'),
            'train_iter':
                open(os.path.join(self.folder_name, 'train_iter.txt'), 'w'),
            'train_epoch':
                open(os.path.join(self.folder_name, 'train_epoch.txt'), 'w'),
            'test_iter':
                open(os.path.join(self.folder_name, 'test_iter.txt'), 'w'),
            'test_epoch':
                open(os.path.join(self.folder_name, 'test_epoch.txt'), 'w'),
        }

    def __close_files(self):
        for f in self.files.values():
            f.close()

    def __parse_dictionary(self, dictionary):
        """
        make dictionary to string as following format
        'key: value\n'
        """
        result = ''
        for key, value in dictionary.items():
            result += '%s: %s\n' % (str(key), str(value))
        return result

    def __save_params(self):
        param_file = self.files['summary']

        param_file.write('Hyperparameters\n')
        param_file.write(self.__parse_dictionary(self.params))
        param_file.write('Other Settings\n')
        param_file.write(self.__parse_dictionary(self.settings))
        if self.miscs is not None:
            param_file.write('Miscellaneous\n')
            param_file.write(self.__parse_dictionary(self.miscs))
