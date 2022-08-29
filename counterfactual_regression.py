import sys

from pobm.prep import median_spo2

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
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

from SHHS_Causal.causal_dataloader import CausalDataset
from SHHS_Causal.tabular_model_DL import TabularModel
from SHHS_Causal.constants import x_columns, y_column, x_columns_continuous, x_columns_categorical, t_baseline, \
    t_autoencoder, t_pca
import utils.graphics as graph
from sklearn.metrics import classification_report
from utils.utils_func import save_dict
from SHHS_Causal.TabNet import TabNet
from SHHS_Causal.mutual_information import MutualInformation


class Main:
    def __init__(self, batch_size, num_epochs, short_sample, device):
        self.assure_reproducibility(32)

        self.epochs_without_improvement = 0
        self.params = {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 0.01,

            # TabularDL parameters
            'embedding_size_categorical': 2,
            'embedding_size_continuous': 6,
            'combined_embedding': 26,
            'dropout': 0.1,

            # TabNet parameters
            'model': 'TabularDL',  # TabularDL / TabNet

            # weight for each loss
            'representation_weight': 0.01,
            'regularization_weight': 0.0005,
            'tabnet_weight': 0.1,
        }

        self.writer = SummaryWriter()
        self.short_sample = short_sample
        self.saved_path = '/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/saved_models'

        if device == 'gpu':
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f"Using {self.device} device")

        self.representation_loss = MutualInformation()
        self.classification_loss = nn.BCEWithLogitsLoss()

        self.ticks_fontsize, self.fontsize, self.letter_fontsize = 15, 15, 15

    @staticmethod
    def assure_reproducibility(seed):
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_database(self):
        all_data = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                            'data_dimensionality_reduction.csv'))

        if self.short_sample is True:
            all_data = all_data.sample(200)

        t = all_data[t_baseline].values
        x_continuous = all_data[x_columns_continuous].values
        x_categorical = all_data[x_columns_categorical].values
        y = all_data[y_column].values

        x_categorical = CatBoostEncoder().fit_transform(x_categorical, y).values

        idx_train, idx_val, _, _ = train_test_split(np.arange(t.shape[0]), np.arange(t.shape[0]), train_size=0.8,
                                                    random_state=32, stratify=y)

        train_dataset = CausalDataset(t[idx_train], x_continuous[idx_train], x_categorical[idx_train], y[idx_train])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['batch_size'],
                                                       shuffle=True)

        val_dataset = CausalDataset(t[idx_val], x_continuous[idx_val], x_categorical[idx_val], y[idx_val])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=True)

        x_shape_categorical = x_categorical.shape[1]
        x_shape_continuous = x_continuous.shape[1]

        self.params['x_shape_categorical'] = x_categorical.shape[1]
        self.params['x_shape_continuous'] = x_continuous.shape[1]

        return train_dataloader, val_dataloader, x_shape_categorical, x_shape_continuous

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
        reg_loss = self.params['regularization_weight'] * l2_reg

        return reg_loss

    def run_epoch(self, data_loader, optimizer, model, train_flag, epoch):
        if train_flag is True:
            add_str = 'Train'
            model.train()
        else:
            add_str = 'Val'
            model.eval()

        representation_loss_epoch, classification_loss_epoch, regularization_loss_epoch, tabnet_loss_epoch = [], [], [], []
        for i, data in enumerate(tqdm(data_loader)):
            if train_flag is True:
                optimizer.zero_grad()

            x_continuous, x_categorical, y, t = data

            x_continuous = x_continuous.to(self.device)
            x_categorical = x_categorical.to(self.device)
            y = torch.squeeze(y.to(self.device))
            t = t.to(self.device)

            representation_x, predicted_outcome, tabnet_loss = model(x_continuous, x_categorical, t)

            representation_loss = self.representation_loss.getMutualInformation(representation_x, t)
            representation_loss = representation_loss * self.params['representation_weight']
            classification_loss = self.classification_loss(predicted_outcome, y)
            regularization_loss = self.regularization_loss(model)
            tabnet_loss = self.params['tabnet_weight'] * tabnet_loss

            if train_flag is True:
                representation_loss.backward(retain_graph=True)
                classification_loss.backward(retain_graph=True)
                regularization_loss.backward(retain_graph=True)
                tabnet_loss.backward(retain_graph=True)

                # self.plot_gradients(model, epoch=epoch)
                optimizer.step()

            representation_loss_epoch.append(representation_loss.data.item())
            classification_loss_epoch.append(classification_loss.data.item())
            regularization_loss_epoch.append(regularization_loss.data.item())
            tabnet_loss_epoch.append(tabnet_loss.data.item())

        representation_loss_epoch = np.mean(representation_loss_epoch)
        classification_loss_epoch = np.mean(classification_loss_epoch)
        regularization_loss_epoch = np.mean(regularization_loss_epoch)
        tabnet_loss_epoch = np.mean(tabnet_loss_epoch)

        self.writer.add_scalar("Classification/" + add_str, classification_loss_epoch, epoch)
        self.writer.add_scalar("Regularization/" + add_str, regularization_loss_epoch, epoch)
        self.writer.add_scalar("Tabnet/" + add_str, tabnet_loss_epoch, epoch)
        self.writer.add_scalar("Representation/" + add_str, representation_loss_epoch, epoch)

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
        if self.params['model'] == 'TabularDL':
            return TabularModel(self.params).to(self.device)
        elif self.params['model'] == 'TabNet':
            input_dim = self.params['x_shape_categorical'] + self.params['x_shape_continuous']
            return TabNet(input_dim=input_dim, output_dim=1, params=self.params)

    def run(self):
        train_dataloader, val_dataloader, _, _ = self.get_database()

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

            if self.params['model'] == 'TabNet':
                _, predicted_outcome, _ = model(x_continuous, x_categorical, t)
            else:
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
        train_dataloader, val_dataloader, x_shape_categorical, x_shape_continuous = self.get_database()

        def objective(trial):
            self.params = {
                'batch_size': 128,
                'num_epochs': 200,
                'learning_rate': 0.005,
                'x_shape_categorical': x_shape_categorical,
                'x_shape_continuous': x_shape_continuous,

                'model': 'TabNet',  # TabularDL / TabNet

                # weight for each loss
                'representation_weight': 0.01,
                'regularization_weight': 0.0005,
                'tabnet_weight': 0.1,

                # 'embedding_size_categorical': trial.suggest_int('embedding_size_categorical', 2, 32, step=2),
                # 'embedding_size_continuous': trial.suggest_int('embedding_size_continuous', 2, 32, step=2),
                # 'combined_embedding': trial.suggest_int('combined_embedding', 2, 32, step=2),
                # 'dropout': trial.suggest_float('dropout', 0.1, 0.8, step=0.1),

                'n_d': trial.suggest_int('n_d', 4, 64, step=2),
                'n_a': trial.suggest_int('n_a', 4, 64, step=2),
                'n_steps': trial.suggest_int('n_steps', 3, 10, step=1),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0, step=0.1),
                'cat_emb_dim': trial.suggest_int('cat_emb_dim', 2, 8, step=1),
                'n_independent': trial.suggest_int('n_independent', 1, 4, step=1),
                'n_shared': trial.suggest_int('n_shared', 1, 4, step=1),
                'momentum': trial.suggest_float('momentum', 0.0, 1.0, step=0.1),
                'mask_type': trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
            }
            print(self.params)

            model = self.get_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'])
            best_val_loss = self.train_model(train_dataloader, val_dataloader, optimizer, model)

            self.params['val_loss'] = best_val_loss
            save_dict(self.params, file_name='/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/optuna_results/optuna.csv')
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

    def estimate_dose_response(self):
        _, val_dataloader, _, _ = self.get_database()

        model = self.get_model()
        model.load_state_dict(torch.load(os.path.join(self.saved_path, 'tabnet_representation.pth')))
        model.eval()

        for i, data in enumerate(tqdm(val_dataloader)):
            x_continuous, x_categorical, y, t = data
            _, predicted_outcome, _ = model(x_continuous, x_categorical, t)

            ic(x_continuous.shape, x_categorical.shape, y.shape, t.shape, predicted_outcome.shape)
            break

    def visualization_representation(self):
        train_dataloader, _, _, _ = self.get_database()

        model = self.get_model()
        model.load_state_dict(torch.load(os.path.join(self.saved_path, 'tabnet_representation.pth')))
        model.eval()

        all_x, all_representation, all_t = [], [], []
        for i, data in enumerate(tqdm(train_dataloader)):
            x_continuous, x_categorical, y, t = data
            x = torch.cat([x_continuous, x_categorical], 1)

            representation_x, _, _ = model(x_continuous, x_categorical, t)

            all_x.append(x)
            all_representation.append(representation_x)
            all_t.append(t)

            if i == 10:
                break

        all_x = torch.cat(all_x, 0).detach().cpu().numpy()
        all_representation = torch.cat(all_representation, 0).detach().cpu().numpy()
        all_t = torch.cat(all_t, 0).detach().cpu().numpy()
        ic(all_x.shape, all_representation.shape, all_t.shape)

        all_x_TSNE = TSNE(n_components=2).fit_transform(all_x)
        all_representation_TSNE = TSNE(n_components=2).fit_transform(all_representation)
        ic(all_x_TSNE.shape, all_representation_TSNE.shape)

        fig, axes = graph.create_figure(subplots=(1, 2), figsize=(16, 8), sharey=False, sharex=False)
        axes[0][0].scatter(all_x_TSNE[:, 0], all_x_TSNE[:, 1], c=all_t, alpha=0.5)
        axes[0][1].scatter(all_representation_TSNE[:, 0], all_representation_TSNE[:, 1], c=all_t, alpha=0.5)

        graph.complete_figure(fig, axes, put_legend=[[False, False]],
                              y_titles=[['comp-1', 'comp-2']], x_titles=[['comp-1', 'comp-2']],
                              xticks_fontsize=self.ticks_fontsize, yticks_fontsize=self.ticks_fontsize,
                              xlabel_fontsize=self.fontsize, ylabel_fontsize=self.fontsize, tight_layout=True,
                              savefig=True, main_title='tsne_visu-causal', legend_fontsize=self.fontsize)


mean_mu_1 = [0.39703614, 0.43614898, 0.61330451, 0.80149212, 0.90965769, 0.92514184,
             0.93451862, 0.93766468, 0.93901879, 0.93931323, 0.93928047, 0.93913933,
             0.93899402, 0.93882752, 0.93859794, 0.93841049, 0.93816825, 0.93805596,
             0.93785797, 0.9376851, 0.93757562, 0.93744244, 0.93733568, 0.93722556,
             0.9370918, 0.9370245, 0.93697501, 0.93693999, 0.93689639, 0.93686589,
             0.93683661, 0.93682865, 0.93680112, 0.93679688, 0.93679425, 0.93678178,
             0.93677662, 0.93676084, 0.93675025, 0.9367329, 0.93672868, 0.93672868,
             0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868,
             0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868,
             0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868,
             0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868,
             0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868, 0.93672868,
             0.93672868, 0.93672868, 0.93672868, 0.93672868]

mean_mu_2 = [0.87240681, 0.75005895, 0.74918235, 0.74799826, 0.74689678, 0.74570694,
             0.74470635, 0.74355848, 0.74241484, 0.7411973, 0.74005004, 0.73881221,
             0.7373695, 0.73584665, 0.73426683, 0.73305322, 0.73164859, 0.73013838,
             0.7288526, 0.72738473, 0.72583997, 0.72441115, 0.72355069, 0.72241449,
             0.72132302, 0.72017859, 0.71905648, 0.71820455, 0.71740508, 0.71664765,
             0.71597227, 0.71508971, 0.71432858, 0.71350389, 0.7127146, 0.71167034,
             0.71090694, 0.71018521, 0.7093944, 0.70870133, 0.70799136, 0.70694764,
             0.70584391, 0.70509793, 0.70435015, 0.70359163, 0.70277342, 0.70178555,
             0.70084666, 0.70022065, 0.6996784, 0.69902421, 0.69841789, 0.69764882,
             0.69685805, 0.69597969, 0.69503532, 0.69440995, 0.69367738, 0.69310483,
             0.69236728, 0.69185721, 0.69135114, 0.69092432, 0.6905091, 0.68996056,
             0.68955496, 0.68908185, 0.68845348, 0.68806765, 0.68759586, 0.68703205,
             0.68651427, 0.6859785, 0.68545041, 0.68483406, 0.6843349, 0.68373782,
             0.68303493, 0.68262766, 0.68234722, 0.68197737, 0.6816058, 0.68119311,
             0.68084468, 0.68049568, 0.68005359, 0.67964876, 0.67930886, 0.67896029,
             0.67863505, 0.67815115, 0.67771871, 0.67727682, 0.67688167, 0.67651599,
             0.67619054, 0.67583265, 0.67553586, 0.67532223, 0.67518392, 0.67502761,
             0.67491019, 0.67471907, 0.67464953, 0.67454872, 0.67455759, 0.67456722,
             0.6746192, 0.67472036, 0.6747901, 0.67481486, 0.67487543, 0.67484727,
             0.67474439, 0.67468161, 0.67461674, 0.67448265, 0.67433926, 0.67424491,
             0.67412707, 0.67398001, 0.67377741, 0.67357386, 0.67338184, 0.67324819,
             0.67303751, 0.67285774, 0.67263702, 0.67246, 0.6723204, 0.67211056,
             0.67197475, 0.67181861, 0.67163109, 0.67145172, 0.67115712, 0.6707556,
             0.67036694, 0.66998679, 0.66963226, 0.66935513, 0.66913628, 0.66891161,
             0.66873883, 0.66846373, 0.66815302, 0.66787392, 0.66756413, 0.66743442,
             0.66727163, 0.66709965, 0.66693703, 0.66689195, 0.66684025, 0.66675683,
             0.66676845, 0.66668882, 0.66666656, 0.66664636, 0.66664103, 0.66672444,
             0.66687341, 0.66695785, 0.66714554, 0.66730226, 0.66746438, 0.66766973,
             0.6678454, 0.66800895, 0.66811609, 0.66819771, 0.66828802, 0.66833188,
             0.66842765, 0.66851116, 0.6685271, 0.66854249, 0.66846215, 0.66837539,
             0.66825296, 0.66812815, 0.66795108, 0.66776286, 0.66764558, 0.66747544,
             0.66727077, 0.66707946, 0.66689342, 0.66668186, 0.66643154, 0.66617964,
             0.6659921, 0.66578268, 0.66557707, 0.66539518, 0.66514721, 0.66482889,
             0.66452191, 0.66420155, 0.66382758, 0.66356448, 0.66321085, 0.66287804,
             0.66252349, 0.66214585, 0.66192019, 0.66154345, 0.66123333, 0.66091364,
             0.66062946, 0.66027756, 0.6600617, 0.65985589, 0.65969163, 0.65955636,
             0.65931886, 0.65911444, 0.65895307, 0.65887301, 0.65879179, 0.65874432,
             0.658737, 0.65875632, 0.65887801, 0.65894063, 0.6590531, 0.65908839,
             0.65908194, 0.65904469, 0.65896083, 0.65893518, 0.65888713, 0.6588058,
             0.65869758, 0.6586056, 0.65848284, 0.65833186, 0.65814528, 0.657969,
             0.65778655, 0.65759572, 0.65745571, 0.65727922, 0.65713762, 0.65688023,
             0.65670365, 0.65653043, 0.65636644, 0.65617743, 0.65608667, 0.65604092,
             0.65600334, 0.65588285, 0.65583575, 0.65582012, 0.65579517, 0.65580321,
             0.6558191, 0.65575134, 0.65573821, 0.65572262, 0.65571872, 0.65570709,
             0.65569993, 0.6556888, 0.65567644, 0.65566174, 0.65565429, 0.65564978,
             0.65564978, 0.65564575, 0.65563352, 0.65563352, 0.65563305, 0.65563305,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837, 0.65563837,
             0.65563837, 0.65563837, ]

mean_mu_3 = [0.50372478, 0.64398462, 0.75232745, 0.86191378, 0.869855, 0.86617222,
             0.87690462, 0.87695259, 0.87683652, 0.87661002, 0.88635039, 0.88598947,
             0.8855827, 0.88507449, 0.88450862, 0.88375552, 0.88281152, 0.88157892,
             0.88030427, 0.88869843, 0.88709709, 0.88480092, 0.88231521, 0.88921998,
             0.88565409, 0.8805691, 0.88068782, 0.8867629, 0.88991182, 0.8812459,
             0.88139097, 0.88141005, 0.88140561, 0.8813993, 0.88139853, 0.88139537,
             0.88139244, 0.88138683, 0.88138683, 0.88138683, 0.88138242, 0.88138242,
             0.88138242, 0.88138087, 0.88138087, 0.88138087, 0.88138087, 0.88138087,
             0.88138087, 0.88138087]


def plot_estimate_response():
    base_path = '/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/save_np'

    def one_panel(filename, ax, min_T, max_T, mean_mu, smooth=False):
        x_coordinate = np.arange(start=min_T, stop=max_T + 0.1, step=0.1)
        if smooth is True:
            from scipy import signal
            mean_mu = signal.medfilt(mean_mu, 7)

        ax.plot(x_coordinate, mean_mu, c='dodgerblue')

    all_data = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                        'data_dimensionality_reduction.csv'))

    fig, axes = graph.create_figure(subplots=(1, 3), figsize=(24, 8))
    ticks_fontsize, fontsize, letter_fontsize = 15, 15, 15

    all_T = all_data[t_baseline].values
    min_T_baseline, max_T_baseline = np.min(all_T), np.max(all_T)
    one_panel(filename='mu_t_CA_relative_3_gps_False_rf.npy', ax=axes[0][0], min_T=min_T_baseline, max_T=max_T_baseline,
              mean_mu=mean_mu_1)

    all_T = all_data[t_autoencoder].values
    min_T_autoencoder, max_T_autoencoder = np.min(all_T), np.max(all_T)
    one_panel(filename='mu_t_autoencoder_treatment_gps_False_rf.npy', ax=axes[0][1], min_T=min_T_autoencoder,
              max_T=max_T_autoencoder, mean_mu=mean_mu_2, smooth=True)

    all_T = all_data[t_pca].values
    min_T_pca, max_T_pca = np.min(all_T), np.max(all_T)
    one_panel(filename='mu_t_pca_treatment_gps_False_rf.npy', ax=axes[0][2], min_T=min_T_pca, max_T=max_T_pca,
              mean_mu=mean_mu_3, smooth=True)

    x_pos, y_pos = -0.1, 1.01
    axes[0][0].text(x_pos, y_pos, "(a)", fontsize=letter_fontsize, transform=axes[0][0].transAxes)
    axes[0][1].text(x_pos, y_pos, "(b)", fontsize=letter_fontsize, transform=axes[0][1].transAxes)
    axes[0][2].text(x_pos, y_pos, "(c)", fontsize=letter_fontsize, transform=axes[0][2].transAxes)

    graph.complete_figure(fig, axes, put_legend=[[False, False, False]],
                          y_titles=[['Average dose function', '', '']],
                          x_titles=[['$CA_{3}$', 'Auto-encode', 'PCA']],
                          xticks_fontsize=ticks_fontsize, yticks_fontsize=ticks_fontsize,
                          xlabel_fontsize=fontsize, ylabel_fontsize=fontsize, tight_layout=True,
                          savefig=True, main_title='counterfactual_dose_response',
                          legend_fontsize=fontsize)


if __name__ == "__main__":
    # main = Main(batch_size=512, num_epochs=200, short_sample=True, device='cpu')

    # main.run()
    # main.baseline_model()
    # main.optuna_search()
    # main.visualization_representation()

    # main.estimate_dose_response()
    plot_estimate_response()
