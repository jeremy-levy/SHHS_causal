import sys

from torch.utils.tensorboard import SummaryWriter

from utils.utils_func import save_dict

sys.path.append('/home/jeremy.levy/Jeremy/copd_osa')

import pandas as pd
import os
from category_encoders.cat_boost import CatBoostEncoder
from icecream import ic
import torch
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import optuna
import random

from SHHS_Causal.causal_dataloader import CausalDataset
from SHHS_Causal.tabular_model_DL import TabularModel
from SHHS_Causal.constants import x_columns, y_column, x_columns_continuous, x_columns_categorical, t_baseline
import utils.graphics as graph
from sklearn.metrics import classification_report


class Main:
    def __init__(self, batch_size, num_epochs, short_sample, device):
        self.assure_reproducibility(32)

        self.epochs_without_improvement = 0
        self.params = {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 0.01,

            'embedding_size_categorical': 16,
            'embedding_size_continuous': 16,

        }

        self.writer = SummaryWriter()
        self.short_sample = short_sample
        self.saved_path = '/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/saved_models'

        if device == 'gpu':
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device} device")

        self.representation_loss = self.get_representation_loss()
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.regularization_weight = 0.0005

        self.ticks_fontsize, self.fontsize, self.letter_fontsize = 15, 15, 15

    @staticmethod
    def assure_reproducibility(seed):
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_representation_loss(self):
        def no_loss(representation_x, t):
            return -1
        return no_loss

    def get_database(self):
        # TODO: Add preprocessing data

        all_data = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                            'data_dimensionality_reduction.csv'))

        if self.short_sample is True:
            all_data = all_data.sample(200)

        t = all_data[t_baseline].values
        x_continuous = all_data[x_columns_continuous].values
        x_categorical = all_data[x_columns_categorical].values
        y = all_data[y_column].values

        # x_categorical = OneHotEncoder(sparse=False).fit_transform(x_categorical)
        x_categorical = CatBoostEncoder().fit_transform(x_categorical, y).values

        idx_train, idx_val, _, _ = train_test_split(np.arange(t.shape[0]), np.arange(t.shape[0]), train_size=0.8,
                                                    random_state=32, stratify=y)

        train_dataset = CausalDataset(t[idx_train], x_continuous[idx_train], x_categorical[idx_train], y[idx_train])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['batch_size'],
                                                       shuffle=True)

        val_dataset = CausalDataset(t[idx_val], x_continuous[idx_val], x_categorical[idx_val], y[idx_val])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=True)

        self.params['x_shape_categorical'] = x_categorical.shape[1]
        self.params['x_shape_continuous'] = x_continuous.shape[1]

        return train_dataloader, val_dataloader

    def plot_gradients(self, model, epoch):
        all_grads = []
        for param in model.parameters():
            all_grads += list(param.grad.view(-1).cpu().numpy())

        fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8))

        axes[0][0].hist(all_grads, bins=50)
        plt.yscale('log')

        graph.complete_figure(fig, axes, put_legend=[[False]],
                              xticks_fontsize=self.ticks_fontsize, yticks_fontsize=self.ticks_fontsize,
                              xlabel_fontsize=self.fontsize, ylabel_fontsize=self.fontsize, tight_layout=True,
                              savefig=True, main_title='grads_model_' + str(epoch),
                              legend_fontsize=self.fontsize)

    def regularization_loss(self, model):
        l2_reg = torch.tensor(0., device=self.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        reg_loss = self.regularization_weight * l2_reg

        return reg_loss

    def run_epoch(self, data_loader, optimizer, model, train_flag, epoch):
        if train_flag is True:
            add_str = 'Train'
            model.train()
        else:
            add_str = 'Val'
            model.eval()

        representation_loss_epoch, classification_loss_epoch, regularization_loss_epoch = [], [], []
        for i, data in enumerate(tqdm(data_loader)):
            if train_flag is True:
                optimizer.zero_grad()

            x_continuous, x_categorical, y, t = data

            x_continuous = x_continuous.to(self.device)
            x_categorical = x_categorical.to(self.device)
            y = torch.squeeze(y.to(self.device))
            t = t.to(self.device)

            representation_x, predicted_outcome = model(x_continuous, x_categorical, t)

            # representation_loss = self.representation_loss(representation_x, t)
            classification_loss = self.classification_loss(predicted_outcome, y)
            regularization_loss = self.regularization_loss(model)

            if train_flag is True:
                # representation_loss.backward(retain_graph=True)
                classification_loss.backward(retain_graph=True)
                regularization_loss.backward(retain_graph=True)
                # self.plot_gradients(model, epoch=epoch)
                optimizer.step()

            # representation_loss_epoch.append(representation_loss.data.item())
            classification_loss_epoch.append(classification_loss.data.item())
            regularization_loss_epoch.append(regularization_loss.data.item())

        representation_loss_epoch = np.mean(representation_loss_epoch)
        classification_loss_epoch = np.mean(classification_loss_epoch)
        regularization_loss_epoch = np.mean(regularization_loss_epoch)

        self.writer.add_scalar("Classification/" + add_str, classification_loss_epoch, epoch)
        self.writer.add_scalar("Regularization/" + add_str, regularization_loss_epoch, epoch)

        log = " (" + add_str + ") Representation: {:.4f}  Classification: {:.4f}  | ".format(
            representation_loss_epoch, classification_loss_epoch)

        return classification_loss_epoch, log

    def train_model(self, train_dataloader, val_dataloader, optimizer, model):
        best_val_loss = np.inf
        for epoch in range(1, self.params['num_epochs'] + 1):
            epoch_time = time.time()

            try:
                _, train_log = self.run_epoch(train_dataloader, optimizer, model, train_flag=True, epoch=epoch)
            except KeyboardInterrupt:
                break

            with torch.no_grad():
                val_loss, val_log = self.run_epoch(val_dataloader, optimizer, model, train_flag=False, epoch=epoch)

            best_val_loss, stop_training = self.save_model(best_val_loss, val_loss, model, epoch, early_stopping=15)

            epoch_time = time.time() - epoch_time
            log = 'Epoch {:.0f} |'.format(epoch) + train_log + val_log
            log += "Epoch Time: {:.2f} secs | Best loss is {:.4f}".format(epoch_time, best_val_loss)
            print(log)

            if stop_training is True:
                print('Early stopping')
                break

        return best_val_loss

    def save_model(self, best_test_loss, eval_loss, model, epoch, early_stopping=15):
        if best_test_loss > eval_loss:
            torch.save(model.state_dict(), os.path.join(self.saved_path, 'model_' + str(eval_loss) + '.pth'))
            best_test_loss = eval_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= early_stopping:
            stop_training = True
        else:
            stop_training = False

        return best_test_loss, stop_training

    def get_model(self):
        print(self.params)
        return TabularModel(self.params).to(self.device)

    def run(self):
        train_dataloader, val_dataloader = self.get_database()

        model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'])

        best_val_loss = self.train_model(train_dataloader, val_dataloader, optimizer, model)
        self.writer.close()

        model.load_state_dict(torch.load(os.path.join(self.saved_path, 'model_' + str(best_val_loss) + '.pth')))
        dict_res = self.inference(val_dataloader, model)

        print(dict_res)

    def inference(self, val_dataloader, model):
        y_pred, y_test = [], []
        for i, data in enumerate(tqdm(val_dataloader)):
            x_continuous, x_categorical, y, t = data

            x_continuous = x_continuous.to(self.device)
            x_categorical = x_categorical.to(self.device)
            t = t.to(self.device)

            _, predicted_outcome = model(x_continuous, x_categorical, t)

            y_pred.append(predicted_outcome)
            y_test.append(y)

        y_pred = torch.cat(y_pred, 0).detach().cpu().numpy().squeeze()
        y_test = torch.cat(y_test, 0).cpu().numpy().squeeze()

        fig, axes = graph.create_figure(subplots=(1, 1), figsize=(8, 8))
        axes[0][0].hist(y_pred, label=y_pred)
        axes[0][0].hist(y_test, label=y_test)
        graph.complete_figure(fig, axes, put_legend=[[True]], xticks_fontsize=self.ticks_fontsize,
                              yticks_fontsize=self.ticks_fontsize, xlabel_fontsize=self.fontsize,
                              ylabel_fontsize=self.fontsize, tight_layout=True, savefig=True,
                              main_title='output_causal_model', legend_fontsize=self.fontsize)

        y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.5] = 1

        dict_results = classification_report(y_test, y_pred)
        return dict_results

    def baseline_model(self):
        all_data = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                            'data_dimensionality_reduction.csv'))

        if self.short_sample is True:
            all_data = all_data.sample(200)

        x = all_data[t_baseline + x_columns_continuous].values
        x_categorical = all_data[x_columns_categorical].values
        x_categorical = OneHotEncoder(sparse=False).fit_transform(x_categorical)
        x = np.concatenate([x, x_categorical], axis=1)
        y = all_data[y_column].values

        x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=32, stratify=y)

        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)

        y_pred = xgb.predict(x_val)
        dict_results = classification_report(y_val, y_pred)

        print(dict_results)

    def optuna_search(self):
        train_dataloader, val_dataloader = self.get_database()

        def objective(trial):
            self.params = {
                'batch_size': 128,
                'num_epochs': 200,
                'learning_rate': 0.005,

                'embedding_size_categorical': trial.suggest_int('embedding_size_categorical', 2, 32, step=2),
                'embedding_size_continuous': trial.suggest_int('embedding_size_continuous', 2, 32, step=2),
                'combined_embedding': trial.suggest_int('combined_embedding', 2, 32, step=2),
                'dropout': trial.suggest_float('combined_embedding', 0.1, 0.8, step=0.1),
            }

            model = self.get_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'])
            best_val_loss = self.train_model(train_dataloader, val_dataloader, optimizer, model)

            self.params['val_loss'] = best_val_loss
            save_dict(self.params, file_name='/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/optuna_results//optuna.csv')
            return best_val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200)

        best_trial = study.best_trial
        print('Best loss: {}'.format(best_trial.value))
        print("Best hyperparameters: {}".format(best_trial.params))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(file=os.path.join('optuna_plot_param_importances.png'), format='png')

        fig = optuna.visualization.plot_slice(study)
        fig.write_image(file=os.path.join('plot_slice.png'), format='png')


if __name__ == "__main__":
    main = Main(batch_size=64, num_epochs=200, short_sample=False, device='gpu')

    main.run()
    # main.baseline_model()
