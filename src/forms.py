import json
from wtforms import StringField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Optional
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired


def json_param_reader(data):
    try:
        params = json.loads(data)
    except Exception:
        params = {}
    return params


class TypeForm(FlaskForm):
    name = StringField('Имя модели', validators=[DataRequired()])
    model_type = SelectField('Тип модели', choices=[('RF', 'Случайный лес'), ('GB', 'Градиентный бустинг')])


class ParamsForm(FlaskForm):
    n_estimators = IntegerField('Количество деревьев')
    learning_rate = FloatField('Темп обучения')
    max_depth = IntegerField('Максимальная глубина деревьев', validators=[Optional()])
    feature_subsample_size = IntegerField('Число признаков для одного дерева', validators=[Optional()])
    trees_parameters = StringField('Дополнительные параметры для деревьев в формате JSON', validators=[Optional()], filters=[json_param_reader])


class DataForm(FlaskForm):
    name = StringField('Название датасета', validators=[DataRequired()])
    data_file = FileField('Файл', validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат, допускается только формат csv')])
    target_name = StringField('Столбец целевой переменной', validators=[Optional()])
    target_file = FileField('Файл с целевой переменной (опционально)', validators=[Optional(), FileAllowed(['csv'], 'Неверный формат, допускается только формат csv')])


class FitForm(FlaskForm):
    train_data = SelectField('Обучающая выборка', validators=[DataRequired()])
    val_data = SelectField('Валидационная выборка')


class PredictForm(FlaskForm):
    test_data = SelectField('Тестовая выборка', validators=[DataRequired()])
