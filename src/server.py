import os
import numpy as np
import pandas as pd
import io

from flask import Flask, render_template, url_for, session, send_from_directory, request, redirect, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg

from forms import TypeForm, ParamsForm, DataForm, FitForm, PredictForm
from utility import Model, Data

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'
app.url_map.strict_slashes = False

models = {}
datasets = {}


@app.route('/index')
@app.route('/')
def get_index():
    return render_template('index.html', bg_class="bg")


@app.route('/models/', methods=['GET', 'POST'])
def model_start_page():
    form = TypeForm(meta={'csrf': False})
    if form.validate_on_submit():
        session['model_type'] = form.model_type.data
        return redirect(url_for('model_tuning', name=form.name.data))
    return render_template('model_type_set.html', form=form, models=models)


@app.route('/models/<name>/settings/', methods=['GET', 'POST'])
def model_tuning(name):
    model_type = session['model_type']
    form = ParamsForm(meta={'csrf': False})
    if form.validate_on_submit():
        if model_type == 'RF':
            del form.learning_rate
        models[name] = Model(name, model_type, form)
        return redirect(url_for('model_start_page'))
    args = {'form': form, 'name': name, 'model_type': model_type}
    return render_template('model_tuning.html', **args)


@app.route('/data/', methods=['GET', 'POST'])
def get_data():
    form = DataForm()
    if form.validate_on_submit():
        data = pd.read_csv(form.data_file.data, index_col=0)
        target_name = form.target_name.data
        if form.target_file.data is not None:
            target = pd.read_csv(form.target_file.data, index_col=0)
            if target_name == '':
                target_name = target.columns[0]
            data = pd.concat([data, target], axis=1)
        datasets[form.name.data] = Data(form.name.data, data, target_name)
        return redirect(url_for('get_data'))
    return render_template('data_load_page.html', form=form, datasets=datasets)


@app.route('/models/<name>/plot.png')
def plot(name):
    fig = models[name].plot()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/models/<name>/work/', methods=['GET', 'POST'])
def model_page(name):
    fit_form = FitForm(meta={'csrf': False})
    predict_form = PredictForm(meta={'csrf': False})
    fit_form.train_data.choices = [d for d in datasets if datasets[d].target_col]
    fit_form.val_data.choices = ['-'] + fit_form.train_data.choices
    predict_form.test_data.choices = [d for d in datasets]
    if fit_form.validate_on_submit():
        train_data = datasets[fit_form.train_data.data]
        val_data = datasets.get(fit_form.val_data.data)
        models[name].fit(train_data, val_data)
        return redirect(url_for('model_page', name=name))
    if predict_form.validate_on_submit():
        test_data = datasets[predict_form.test_data.data]
        preds = models[name].predict(test_data)
        filename = test_data.name + '_ans.csv'
        path = os.path.join(os.getcwd(), 'tmp/')
        if not os.path.exists(path):
            os.mkdir(path)
        preds.to_csv(os.path.join(path, filename))
        return send_from_directory(path, filename, as_attachment=True)
    args = {'model': models[name], 'fit_form': fit_form, 'predict_form': predict_form}
    return render_template('model_page.html', **args)
