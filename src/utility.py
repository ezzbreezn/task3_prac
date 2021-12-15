import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ensembles import RandomForestMSE, GradientBoostingMSE
import json



class Model:
    __model_types = {'RF': RandomForestMSE, 'GB': GradientBoostingMSE}

    __str_types = {'RF': 'Случайный лес', 'GB': 'Градиентный бустинг'}

    def __init__(self, name, model_type, form):
        self.name = name
        params = form.data
        content = [('Тип модели', self.__str_types[model_type])]
        content += [(form[param].label.text, params[param]) for param in params]
        content = content[:-1]
        self.description = pd.DataFrame(content, columns=['Параметр', 'Значение'])
        trees_parameters = params.pop('trees_parameters')
        params = {**params, **trees_parameters}
        self.model = self.__model_types[model_type](**params)
        
        add_content = []
        for key, val in trees_parameters.items():
            add_content.append((key, val))
        self.add_content = pd.DataFrame(add_content, columns=['Параметр', 'Значение'])
        if not add_content:
            self.param_flag = False
        else:
            self.param_flag = True

        self.train_loss = None
        self.val_loss = None
        self.times = None

    def fit(self, train_data, val_data=None):
        X_train = train_data.features
        y_train = train_data.target
        self.target_name = train_data.target_name
        if val_data is not None:
            self.train_loss, self.val_loss, self.times = self.model.fit(X_train, y_train, val_data.features, val_data.target, True)
            self.best_train_loss = self.train_loss[np.argmin(self.train_loss)]
            self.best_val_loss = self.val_loss[np.argmin(self.val_loss)]
            self.learning_time = self.times[-1]
        else:
            self.train_loss, self.times = self.model.fit(X_train, y_train, return_train_loss=True)
            self.best_train_loss = self.train_loss[np.argmin(self.train_loss)]
            self.learning_time = self.times[-1]

    @property
    def fitted(self):
        return self.train_loss is not None

    def predict(self, test_data):
        y_pred = self.model.predict(test_data.features)
        return pd.DataFrame(y_pred, index=test_data.data.index, columns=[self.target_name])

    def plot(self):
        plt.rc('font', size=22)
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rc('savefig', facecolor='white')
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_title('Функция ошибок в процессе обучения')
        ax.plot(np.arange(1, self.model.n_estimators + 1), self.train_loss, label='Обучающая выборка', c='r')
        if self.val_loss is not None:
            ax.plot(np.arange(1, self.model.n_estimators + 1), self.val_loss, label='Валидационная выборка', c='tab:orange')
        ax.set_xlabel('Число деревьев')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True)
        return fig


class Data:
    def __init__(self, name, data, target_name):
        self.name = name
        self.data = data
        self.target_name = target_name
        self.target_col = (target_name != '')

    @property
    def features(self):
        if self.target_col:
            return self.data.drop(columns=self.target_name).to_numpy()
        else:
            return self.data.to_numpy()

    @property
    def target(self):
        if self.target_col:
            return self.data[self.target_name].to_numpy()
        else:
            raise ValueError(f'Wrong target name')
