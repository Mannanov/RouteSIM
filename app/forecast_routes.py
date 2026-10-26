"""Веб-сценарий почасового прогноза спроса на основном маршруте /forecast/."""

import json
import sqlite3
from datetime import date, datetime
from time import perf_counter

from flask import (
    Blueprint, abort, current_app, flash, redirect, render_template,
    request, send_file, session, url_for,
)
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import BooleanField, FloatField, IntegerField, SelectField, StringField, SubmitField
from wtforms.validators import InputRequired, Length, NumberRange, Regexp

from forecast_engine import forecast_service as service
import models
from serv_funcs import check_disk_size


forecast_bp = Blueprint('forecast', __name__, url_prefix='/forecast')
FILE_LABELS = {
    'matrix_stops': 'Матрица межостановочных корреспонденций',
    'ref_stops': 'Справочник остановок',
    'ref_tr_distr': 'Справочник транспортных районов',
    'los_districts': 'Уровень обслуживания (характеристики межрайонных связей)',
}


class ProjectForm(FlaskForm):
    """Поля кода и описания создаваемого проекта."""
    proj_name = StringField('Код проекта', validators=[InputRequired(), Length(min=1, max=9),
                            Regexp(r'^[\w-]+$', message='Используйте буквы, цифры, _ или -')])
    proj_descr = StringField('Описание', validators=[Length(max=150)])
    submit = SubmitField('Создать проект')


class UploadForm(FlaskForm):
    """Форма последовательной загрузки одного входного CSV."""
    file = FileField('CSV-файл', validators=[FileRequired(), FileAllowed(['csv'], 'Требуется CSV-файл')])
    submit = SubmitField('Загрузить')


class DeleteProjectForm(FlaskForm):
    """Подтверждение удаления проекта и его файлов."""
    submit = SubmitField('Удалить')


class CalculateForm(FlaskForm):
    """Пользовательские параметры расчёта; технические допуски остаются в конфигурации."""
    hours = SelectField('Часы', choices=[('day', 'День')] + [
        (h, f'{int(h[1:])}:00–{int(h[1:]) + 1}:00') for h in service.HOUR_COLUMNS
    ], default='day', validators=[InputRequired()])
    recalculation_mode = SelectField('Режим', choices=[('recalculate_all', 'Полный пересчет спроса'),
                                                       ('preserve_old', 'Сохранить исходный спрос для существующих районов')])
    new_connection_mode = SelectField('Связи новых районов', default='all',
        choices=[('all', 'Все связи'), ('analogues_only', 'Только подтвержденные аналогами')])
    include_intrazonal = BooleanField('Учитывать внутрирайонные поездки', default=True)
    los_neighbor_radius_km = SelectField('Радиус аналогов LOS, км', coerce=float, default=1.5,
                                        choices=[(value / 2, f'{value / 2:.1f}') for value in range(3, 8)],
                                        validators=[InputRequired()])
    access_radius_km = SelectField('Радиус доступности остановок, км', coerce=float, default=.7,
                                   choices=[(v / 10, f'{v / 10:.1f}') for v in range(3, 13)],
                                   validators=[InputRequired()])
    min_overlap_share = FloatField('Минимальная доля буфера в районе (0,15–0,30)', default=.15,
                                   validators=[InputRequired(), NumberRange(min=.15, max=.3)])
    validation_folds = IntegerField('Число фолдов проверки по районам', default=5,
                                    validators=[InputRequired(), NumberRange(min=3, max=10)])
    submit = SubmitField('Рассчитать')


@forecast_bp.record_once
def _register(state):
    """Подготавливает хранилище при подключении Blueprint к приложению."""
    service.init_project_storage()


@forecast_bp.before_request
def _require_login():
    """Ограничивает все маршруты модуля пользователями с авторизованной сессией."""
    if not session.get('loggedin') or not session.get('username'):
        return redirect(url_for('login'))


def _project_or_404(proj_name):
    """Возвращает проект текущего пользователя или скрывает его отсутствие ответом 404."""
    project = service.get_project(session['username'], proj_name)
    if project is None:
        abort(404)
    return project


def _account_allowed(additional_bytes=0):
    """Проверяет срок подписки и лимит диска с учётом размера предстоящей загрузки."""
    account = models.get_acc_by_username(session['username'])
    if account is None:
        abort(403)
    limits = models.get_user_limits(account['tarif_name'])
    if account['expire_date'] <= date.today():
        flash('Необходимо продлить подписку.')
        return False
    if check_disk_size(session['username']) + additional_bytes / (1024 * 1024) > limits['disk_limit']:
        flash('Достигнут лимит объёма хранения файлов.')
        return False
    return True


@forecast_bp.route('/', methods=['GET', 'POST'])
def projects():
    """Показывает проекты пользователя и обрабатывает создание нового проекта."""
    form = ProjectForm()
    if form.validate_on_submit() and _account_allowed():
        try:
            service.create_project(session['username'], form.proj_name.data, form.proj_descr.data or '')
            return redirect(url_for('forecast.upload', proj_name=form.proj_name.data))
        except (ValueError, OSError) as error:
            flash(str(error))
    return render_template('forecast_projects.html', title='Прогноз спроса', form=form,
                           rows=service.list_projects(session['username']), statuses=service.STATUS_LABELS)


@forecast_bp.route('/<proj_name>/upload', methods=['GET', 'POST'])
def upload(proj_name):
    """Загружает очередной CSV либо направляет пользователя на следующий этап."""
    project = _project_or_404(proj_name)
    if project['status_prj'] == 2:
        return redirect(url_for('forecast.projects'))
    ready = project['status_src'] == 4
    if request.args.get('file_type') not in (None, 'ref_stops'):
        abort(404)
    coordinate_step = ready and (request.args.get('file_type') == 'ref_stops'
        or not (service.project_paths(session['username'], proj_name)[0] / 'ref_stops.csv').is_file())
    file_type = 'ref_stops' if coordinate_step else (None if ready else service.FILE_TYPES[project['status_src']])
    if request.method == 'POST' and file_type is None:
        abort(400)
    form = UploadForm()
    if form.validate_on_submit() and _account_allowed(request.content_length or 0):
        try:
            service.save_upload(session['username'], proj_name, file_type, form.file.data.stream)
            return redirect(url_for('forecast.upload', proj_name=proj_name))
        except (ValueError, OSError) as error:
            flash(f'Файл не прошёл загрузку: {error}')
    return render_template('forecast_upload.html', title='Прогноз спроса', project=project,
                           form=form, file_label=FILE_LABELS.get(file_type),
                           ready=ready, coordinate_step=coordinate_step,
                           files=[FILE_LABELS[name] for name in service.FILE_TYPES])


@forecast_bp.route('/<proj_name>/calculate', methods=['GET', 'POST'])
def calculate(proj_name):
    """Восстанавливает параметры формы, запускает расчет и сохраняет результат либо отказ."""
    project = _project_or_404(proj_name)
    if project['status_src'] < 4:
        return redirect(url_for('forecast.upload', proj_name=proj_name))
    if project['status_prj'] == 2:
        return redirect(url_for('forecast.projects'))
    if not (service.project_paths(session['username'], proj_name)[0] / 'ref_stops.csv').is_file():
        flash('Для нового расчёта загрузите координаты остановок id_stop, wkt.')
        return redirect(url_for('forecast.upload', proj_name=proj_name, file_type='ref_stops'))
    form = CalculateForm()
    saved = {}
    # Сохранённые технические настройки дополняются видимыми полями формы при отправке.
    if project['config_json']:
        try:
            saved = service.validate_config(json.loads(project['config_json']))
        except (ValueError, TypeError):
            flash('Сохранённые параметры недоступны; выберите параметры нового запуска.')
        if 'hours' in saved and 1 < len(saved['hours']) < len(service.HOUR_COLUMNS):
            form.hours.choices.append(('saved', 'Сохранённые часы: ' + ', '.join(
                hour[1:] for hour in saved['hours']
            )))
        if not form.is_submitted():
            for name in ('recalculation_mode', 'new_connection_mode', 'include_intrazonal', 'los_neighbor_radius_km',
                         'access_radius_km', 'min_overlap_share', 'validation_folds'):
                if name in saved:
                    form[name].data = saved[name]
            if 'hours' in saved:
                form.hours.data = (
                    'day' if saved['hours'] == list(service.HOUR_COLUMNS)
                    else saved['hours'][0] if len(saved['hours']) == 1 else 'saved'
                )
            if form.los_neighbor_radius_km.data not in dict(form.los_neighbor_radius_km.choices):
                flash('Сохранённый радиус аналогов не соответствует шагу 0,5 км; выберите значение из списка.')
                form.los_neighbor_radius_km.data = form.los_neighbor_radius_km.default
    if form.validate_on_submit() and _account_allowed():
        run_id = None
        calc_started = perf_counter()
        calc_status = 'FAILED'
        current_app.logger.info('%s - forecast project %s calc started at %s',
                                session['username'], proj_name, datetime.now().isoformat(sep=' ', timespec='seconds'))
        try:
            hours = (
                list(service.HOUR_COLUMNS) if form.hours.data == 'day'
                else saved['hours'] if form.hours.data == 'saved' else [form.hours.data]
            )
            config = service.validate_config({**saved, **{name: form[name].data for name in (
                'recalculation_mode', 'new_connection_mode', 'include_intrazonal', 'los_neighbor_radius_km',
                'access_radius_km', 'min_overlap_share', 'validation_folds')}, 'hours': hours})
            # Новый run_id фиксируется до проверки входов, чтобы отказ не выдавал старый результат.
            run_id = service.start_project(session['username'], proj_name, config)
            current_app.logger.info('%s - forecast project %s run_id=%s start_project pass; hours=%s; mode=%s',
                                    session['username'], proj_name, run_id, config['hours'], config['recalculation_mode'])
            summary = service.calculate_project(session['username'], proj_name, run_id, config)
            service.finish_project(session['username'], proj_name, run_id, summary['outcome'])
            calc_status = summary['outcome']
            return redirect(url_for('forecast.result', proj_name=proj_name))
        except (ValueError, OSError) as error:
            message = str(error)
            diagnostics = getattr(error, 'diagnostics', None)
            current_app.logger.warning('%s - forecast project %s run_id=%s calculate failed: %s',
                                       session['username'], proj_name, run_id, message)
        except Exception:
            current_app.logger.exception('%s - forecast project %s run_id=%s calculate error',
                                         session['username'], proj_name, run_id)
            message, diagnostics = 'Внутренняя ошибка расчёта. Запуск можно повторить.', None
        finally:
            current_app.logger.info(
                '%s - forecast project %s run_id=%s calc finished at %s; status=%s; duration_seconds=%.3f',
                session['username'], proj_name, run_id, datetime.now().isoformat(sep=' ', timespec='seconds'),
                calc_status, perf_counter() - calc_started)
        # Отказ начатого запуска сохраняем в отчётах и БД; ошибка до запуска остаётся сообщением формы.
        if run_id is not None:
            try:
                service.save_failure(session['username'], proj_name, run_id, message, diagnostics, config)
            except OSError:
                current_app.logger.exception('Не удалось сохранить диагностику прогноза спроса')
            try:
                service.finish_project(session['username'], proj_name, run_id, 'FAILED', message)
            except (ValueError, sqlite3.Error):
                current_app.logger.exception('Не удалось завершить статус прогноза спроса')
            return redirect(url_for('forecast.result', proj_name=proj_name))
        flash(message)
    return render_template('forecast_calculate.html', title='Прогноз спроса', project=project, form=form)


@forecast_bp.route('/<proj_name>/result')
def result(proj_name):
    """Показывает компактную сводку и основные файлы текущего запуска проекта."""
    project = _project_or_404(proj_name)
    if project['status_prj'] == 2:
        return redirect(url_for('forecast.projects'))
    if not project['run_id']:
        return redirect(url_for('forecast.upload', proj_name=proj_name))
    try:
        summary = service.read_summary(session['username'], proj_name, project['run_id'])
    except (ValueError, OSError):
        # После прерывания запись запуска может существовать без завершённого manifest.json.
        summary = {'outcome': 'FAILED', 'artifacts': {}, 'error_message': project['error_message'] or
                   'Расчёт был прерван или его файлы недоступны. Выполните новый запуск.'}
    overview = service.result_overview(session['username'], proj_name, project['run_id'], summary)
    downloads = service.result_downloads(summary)
    return render_template('forecast_info.html', title='Прогноз спроса', project=project,
                           summary=summary, overview=overview, downloads=downloads,
                           presentation=service.result_presentation(summary, overview))


@forecast_bp.route('/<proj_name>/delete', methods=['GET', 'POST'])
def delete_project(proj_name):
    """Показывает подтверждение и удаляет проект текущего пользователя."""
    project = _project_or_404(proj_name)
    if project['status_prj'] == 2:
        flash('Дождитесь завершения расчета перед удалением проекта')
        return redirect(url_for('forecast.projects'))
    form = DeleteProjectForm()
    if form.validate_on_submit():
        try:
            service.delete_project(session['username'], proj_name)
            flash(f'Проект {proj_name} удален')
            return redirect(url_for('forecast.projects'))
        except (ValueError, OSError) as error:
            flash(str(error))
            return redirect(url_for('forecast.projects'))
    return render_template('remove_project.html', title=proj_name, form=form)


@forecast_bp.route('/<proj_name>/download/<artifact>')
def download(proj_name, artifact):
    """Выдаёт разрешенный отчёт владельцу завершённого или неуспешного запуска."""
    project = _project_or_404(proj_name)
    # Во время расчёта файлы могут быть неполными, поэтому скачивание запрещено.
    if project['status_prj'] not in (3, 4):
        abort(409)
    try:
        summary = service.read_summary(session['username'], proj_name, project['run_id'])
        path = service.artifact_path(session['username'], proj_name, project['run_id'], artifact, summary)
    except (ValueError, OSError):
        abort(404)
    return send_file(path, as_attachment=True, download_name=f'{artifact}_{proj_name}.csv', mimetype='text/csv')
