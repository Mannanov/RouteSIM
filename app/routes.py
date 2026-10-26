from flask import request, render_template, flash, redirect, url_for, session, current_app, send_from_directory, \
                    make_response, abort, send_file #, render_template_string
from werkzeug.utils import secure_filename, safe_join
from werkzeug.security import check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import (
    BooleanField,
    DateField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import DataRequired, Length, InputRequired, NumberRange
import os
from datetime import datetime, date
from datetime import date
from rsim import app
from serv_funcs import path_gen, allowed_file, check_source_data, get_df, delete_project, \
                        check_disk_size, check_geo_source, sl_json, check_geo_add_source, \
                        is_geo_technical_field
import auth
import core_los as los
import core_eff as eff
import core_geo as geo
import models
import folium
import json
import colorsys
import re
import secrets
from functools import wraps
from io import BytesIO


def _validate_geo_request(function):
    """Проверяет допустимость имени проекта в запросах GEO."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if 'loggedin' not in session:
            return function(*args, **kwargs)
        generic = function.__name__ in ('download', 'del_proj', 'del_layer')
        if generic and request.args.get('proj_type') != 'geo':
            return function(*args, **kwargs)
        data = request.get_json(silent=True) if request.is_json else None
        project = (kwargs.get('proj_name') or (args[0] if args else None)
                   or request.args.get('project') or request.args.get('proj_name')
                   or request.form.get('proj_name')
                   or (data.get('project') if isinstance(data, dict) else None))
        if not project:
            return function(*args, **kwargs)
        if not _geo_component(project):
            return make_response({'err_info': 'Некорректное имя проекта'}, 400)
        return function(*args, **kwargs)
    return wrapped


def _geo_component(value):
    return (isinstance(value, str) and bool(value.strip()) and value not in ('.', '..')
            and not any(char in value for char in '/\\:\x00'))


@app.errorhandler(404)
def pageNotFound(error):
    return render_template('page404.html', title="Страница не найдена")


@app.errorhandler(RequestEntityTooLarge)
def tooBigFile(error):
    return '<b>Превышен лимит размера файла<b>'


@app.route('/internal_error')
def internal_error():
    return render_template('page_error.html', title="Ошибка на сервере")


@app.route('/uploads/', methods=['GET', 'POST'])
@_validate_geo_request
def download():
    if 'loggedin' in session:
        log_username = session['username']
        file_info = request.args.to_dict()
        valid_types = ['los', 'eff', 'geo', 'demand', 'forecast']
        if file_info.get('proj_type') == 'forecast' and file_info.get('type_path') != 'src':
            project = models.forecast_get_one_project(log_username, file_info.get('proj_name'))
            if project is None:
                abort(404)
            if (project['status_prj'] != models.FORECAST_STATUS_COMPLETED
                    and file_info.get('f_prefix') != 'forecast_failure_diagnostics'):
                abort(409)
        if file_info['proj_type'] in valid_types:
            try: 
                path_src, path_out = path_gen(session['username'], file_info['proj_type'], file_info['proj_name'], False)
                if 'type_path' not in file_info:
                    uploads = os.path.join(current_app.root_path, path_out)
                else:
                    if file_info['type_path'] == 'src':
                        uploads = os.path.join(current_app.root_path, path_src)
                filename = file_info['f_prefix'] + '_' + file_info['proj_name'] + '.csv'
                app.logger.info(f'{log_username} - {file_info["proj_type"]} project upload {filename}')
                if file_info['proj_type'] == 'geo':
                    # Снимок читается целиком; Windows не держит
                    # исходный CSV открытым во время отправки и следующей замены.
                    target = safe_join(uploads, filename)
                    if target is None or not os.path.isfile(target):
                        abort(404)
                    with open(target, 'rb') as stream:
                        snapshot = BytesIO(stream.read())
                    return send_file(snapshot, as_attachment=True, download_name=filename,
                                     mimetype='text/csv')
                return send_from_directory(uploads, filename, as_attachment=True)
            except Exception as app_error:
                app.logger.info(f'{log_username} - upload error {file_info["proj_type"]} - {file_info["proj_name"]}:\n{app_error}')
                return redirect(url_for('internal_error'))


class RemoveProjForm(FlaskForm):
    submit = SubmitField("Удалить")

@app.route('/del_proj/', methods=['GET', 'POST'])
@_validate_geo_request
def del_proj():
    if 'loggedin' in session:
        form = RemoveProjForm()
        del_info = request.args.to_dict()
        log_username = session['username']
        log_proj_type = del_info['proj_type']
        proj_name = del_info['proj_name']
        if form.validate_on_submit():
            delete_project(session['username'], del_info['proj_type'], del_info['proj_name'])
            models.delete_project(session['username'], del_info['proj_type'], del_info['proj_name'])
            app.logger.info(f'{log_username} - {log_proj_type} project {proj_name} deleted')
            return redirect(url_for('landing'))
        return render_template('remove_project.html', title=del_info['proj_name'], form=form)
    else:
        return redirect(url_for('login'))


class RuleUsers(FlaskForm):
    username = StringField(
        filters=[lambda value: value.strip() if value else value],
        validators=[InputRequired(), Length(min=5, max=25)],
    )
    password = PasswordField()
    expire_date = DateField(validators=[InputRequired()], format='%Y-%m-%d')
    tarif_name = SelectField(validators=[InputRequired()])
    update_user = BooleanField(default=False)
    submit = SubmitField('Создать/Обновить')


@app.route('/letsrule/', methods=['POST', 'GET'])
def lets_rule_users():
    if 'loggedin' not in session or session.get('username') != 'cappower':
        return redirect(url_for('login'))

    users_req = models.get_users()
    usernames = {row['username'] for row in users_req}
    form = RuleUsers()
    form.tarif_name.choices = [
        row['tarif_name'] for row in models.get_ref_tarif()
    ]

    if form.validate_on_submit():
        username = form.username.data
        is_update = bool(form.update_user.data)

        if not is_update and not form.password.data:
            form.password.errors.append(
                'Для нового пользователя необходимо указать пароль.'
            )
        elif is_update and username not in usernames:
            flash('Пользователь отсутствует')
        elif not is_update and username in usernames:
            flash('Пользователь с таким логином уже существует')
        elif is_update:
            models.update_new_user(
                username,
                form.password.data,
                form.expire_date.data,
                form.tarif_name.data,
            )
            flash('Данные пользователя успешно обновлены')
            return redirect(url_for('lets_rule_users'))
        else:
            models.commit_new_user(
                username,
                form.password.data,
                form.expire_date.data,
                form.tarif_name.data,
            )
            flash('Пользователь успешно создан')
            return redirect(url_for('lets_rule_users'))

    return render_template(
        'letsrule.html',
        title='Управление пользователями',
        form=form,
        headers=['Пользователь', 'Дата продления', 'Тариф'],
        rows=users_req,
    )


@app.route('/', methods=['GET', 'POST'])
def landing():
    if 'loggedin' in session:
        return render_template('landing.html', title='RouteSIM')
    else:
        return redirect(url_for('login'))


class CreateProject(FlaskForm):
    proj_name = StringField(validators=[InputRequired(), Length(min=1, max=9)])
    proj_descr = StringField(validators=[Length(min=0, max=150)])
    submit = SubmitField('Продолжить')

@app.route('/los/', methods=['GET', 'POST'])
def projects():
    headers = ['Код проекта', 'Описание проекта', 'Дата создания', 'Статус проекта']
    if 'loggedin' in session:
        proj_data = models.los_get_projects(session['username'])
        if len(proj_data) == 0: 
            not_filled = 0 
        else: not_filled = 1
        proj_names = [name[0] for name in proj_data]
        form = CreateProject()
        if form.validate_on_submit():
            if form.proj_name.data in proj_names:
                flash('Проект с указанным кодом уже существует. Выберите другой код.')
            else:
                account = models.get_acc_by_username(session['username'])
                check_expire = account['expire_date'] > date.today()
                if check_expire == False: 
                    flash('Необходимо продлить подписку для создания новых проектов.')
                else:
                    cur_disk = check_disk_size(session['username'])
                    user_limits = models.get_user_limits(account['tarif_name'])
                    disk_limit = user_limits['disk_limit']
                    if cur_disk > disk_limit:
                        flash(f'Вы достигли лимита объема хранения файлов (не более: {disk_limit} Мб).')
                    else:
                        models.los_commit_new_project(session['username'], form.proj_name.data, form.proj_descr.data, 
                        date.today().year, date.today().month, date.today().day)
                        path_gen(session['username'], 'los', form.proj_name.data, True)
                        user_name_log = session['username']
                        app.logger.info(f'{user_name_log} - create los project {form.proj_name.data}')
                        return redirect(url_for('src_loader', proj_name=form.proj_name.data))

        return render_template('los_projects.html', username=session['username'], title="Уровень обслуживания", form=form, 
                                headers=headers, rows=proj_data, not_filled=not_filled)
    else:
        return redirect(url_for('login'))


class UploadForm(FlaskForm):
    file = FileField(validators=[FileRequired(['csv'])])
    flow_load_field = BooleanField("GL", default=False)
    submit = SubmitField("Загрузить")

@app.route('/src_loader/<proj_name>', methods=['GET', 'POST'])
def src_loader(proj_name):
    ''' Загрузка исходных файлов'''
    load_file_state = {
            0:['rnd', 'Загрузите описание маршрутов'],
            1:['matrix_dist_stops', 'Загрузите матрицу межостановочных расстояний'],
            2:['ref_stops', 'Загрузите справочник остановочных пунктов'],
            3:['flow', 'Загрузите корреспонденции на машрутах']
            }

    if 'loggedin' in session:
        log_username = session['username']
        prj_info = models.los_get_one_project(session['username'], proj_name)
        if prj_info['status_prj'] == 1:
            return redirect(url_for('geom_loader', proj_name=proj_name))
        if prj_info['status_prj'] == 2:
            return redirect(url_for('project_calc', proj_name=proj_name))
        elif prj_info['status_prj'] == models.LOS_STATUS_CALCULATING:
            return redirect(url_for('projects'))
        elif prj_info['status_prj'] == models.LOS_STATUS_COMPLETED:
            return redirect(url_for('project_info', proj_name=proj_name))
        elif prj_info['status_prj'] == models.LOS_STATUS_FAILED:
            flash('Предыдущий расчет завершился с ошибкой. Вы можете запустить его повторно или удалить проект.')
            return redirect(url_for('project_calc', proj_name=proj_name))

        form = UploadForm()
        upload_folder, _ = path_gen(session['username'], 'los', proj_name, False)
        load_status = prj_info['status_src']
        load_done = 0
        if prj_info['status_prj'] == 1:  load_done = 1
        if load_status < 4: flash(load_file_state[load_status][1])
        if form.validate_on_submit() and load_status < 4:
            filename_check = secure_filename(form.file.data.filename)
            if allowed_file(filename_check):
                account = models.get_acc_by_username(session['username'])
                user_limits = models.get_user_limits(account['tarif_name'])
                net_limit = user_limits[2]*10 # ЛИМИТ УВЕЛИЧИЛ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                prefix_name = load_file_state[prj_info['status_src']][0]
                filename = prefix_name + '_' + proj_name + '.csv'
                form.file.data.save(upload_folder + filename)
                load_result, msg = check_source_data(upload_folder, filename, prefix_name, net_limit)
                if load_result:
                    session['_flashes'].clear()
                    flash(msg)
                    load_status = prj_info['status_src'] + 1
                    models.los_commit_update_status_src(session['username'], proj_name, load_status)
                    if load_status == 3 and form.flow_load_field.data == False:
                        load_done = 1
                        models.los_commit_update_status_prj(session['username'], proj_name, 1)
                        app.logger.info(f'{log_username} - los project {proj_name} src loaded')
                    if load_status == 4 and form.flow_load_field.data == True:
                        load_done = 1
                        models.los_commit_update_status_prj(session['username'], proj_name, 1)
                        app.logger.info(f'{log_username} - los project {proj_name} src loaded')
                else:
                    os.remove(upload_folder + filename)
                    flash(msg)
            else:
                flash('Требуется файл в формате csv (кодировка UTF-8)')

        return render_template('los_src_loader.html', form=form, title=proj_name, load_done=load_done)

    else:
        return redirect(url_for('login'))


@app.route('/geom_loader/<proj_name>', methods=['GET', 'POST'])
def geom_loader(proj_name):
    ''' Загрузка геометрии'''
    load_file_state = {
            0:['geom_stops', 'Загрузите справочник пространственных данных остановочных пунктов'], 
            1:['geom_clusters', 'Загрузите справочник пространственных данных кластеров ОП'],
            2:['geom_tr_distr', 'Загрузите справочник пространственных данных транспортных районов']
            }

    if 'loggedin' in session:
        log_username = session['username']
        prj_info = models.los_get_one_project(session['username'], proj_name)
        form = UploadForm()
        upload_folder, _ = path_gen(session['username'], 'los', proj_name, False)
        load_status = prj_info['status_geo']
        if load_status < 3: flash(load_file_state[load_status][1])
        load_done = 0
        if prj_info['status_prj'] == 2:  load_done = 1
        if form.validate_on_submit() and load_status < 3:
            if allowed_file(form.file.data.filename):
                # filename = secure_filename(form.file.data.filename)
                prefix_name = load_file_state[load_status][0]
                filename = prefix_name + '_' + proj_name + '.csv'
                form.file.data.save(upload_folder + filename)
                account = models.get_acc_by_username(session['username'])
                user_limits = models.get_user_limits(account['tarif_name'])
                net_limit = user_limits[2]*10
                load_result, msg = check_source_data(upload_folder, filename, prefix_name, net_limit)
                if load_result:
                    session['_flashes'].clear()
                    flash(msg)
                    load_status = prj_info['status_geo'] + 1
                    models.los_commit_update_status_geo(session['username'], proj_name, load_status)
   
                    if load_status == 3:
                        models.los_commit_update_status_prj(session['username'], proj_name, 2)
                        load_done = 1
                        app.logger.info(f'{log_username} - los project {proj_name} src geom loaded')
                else:
                    os.remove(upload_folder + filename)
                    flash(msg)
            else:
                flash('Требуется файл в формате csv (кодировка UTF-8)')

        return render_template('los_geom_loader.html', form=form, title=proj_name, load_done=load_done)

    else:
        return redirect(url_for('login'))


class RuncalcForm(FlaskForm):
    walk_dist_trash =  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist_trash_field = SelectField('DTF', choices=walk_dist_trash)
    max_dist_walk =  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] # нормативная дистанция
    max_dist_walk_field = SelectField('MDF', choices=max_dist_walk)
    ped_speed = [4.0, 4.25, 4.5, 4.75, 5.0]
    ped_speed_field = SelectField('PSF', choices =ped_speed)
    best_tresh =  [0.65, 0.7, 0.75, 0.80, 0.85, 0.90]
    best_tresh_field = SelectField('BSF', choices=best_tresh)
    interval_tresh = [15, 20, 25, 30]
    interval_tresh_field = SelectField('ITF', choices=interval_tresh)
    deep_limit = [0, 1, 2]
    deep_limit_field = SelectField('DLF', choices=deep_limit)
    change_penalty = [1, 2, 3, 4, 5]
    change_penalty_field = SelectField('CHPF', choices=change_penalty, default=3)
    calc_periods = ['h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14','h15','h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23']
    calc_period_field = SelectField('CPF', choices=calc_periods, default='h8')
    rnd_type_field = SelectField("RTF", choices=['Да', 'Нет'])
    use_reducer = SelectField(choices=['Линейная', 'Экспоненциальная'], default="Линейная")
    reducer_grade = [1, 2, 3]
    reducer_grade_field = SelectField(choices=reducer_grade, default=1)
    submit = SubmitField(label="Рассчитать", id='calc_butt')

@app.route('/project_calc/<proj_name>', methods=['GET', 'POST'])
def project_calc(proj_name):
    ''' Расчет показателей'''
    if 'loggedin' in session:
        log_username = session['username']
        form = RuncalcForm(dist_trash_field=0.5, max_dist_walk_field=0.8, ped_speed_field=4.5, best_tresh_field=0.7, deep_limit_field=1, rnd_type_field='Да', interval_tresh_field=25)
        models.recover_stale_calculations(session['username'])
        prj_info = models.los_get_one_project(session['username'], proj_name)
        if proj_name[0] != '_':
            ver = '_' + proj_name
        else:
            ver = proj_name

        if prj_info['status_src'] == 4:
            flow_data = True
        else:
            flow_data = False

        # для того, чтоб избежать нового расчета при перезагрузке страницы
        if prj_info['status_prj'] == models.LOS_STATUS_CALCULATING:
            return redirect(url_for('projects'))
        if prj_info['status_prj'] == models.LOS_STATUS_COMPLETED:
            return redirect(url_for('project_info', proj_name=proj_name))

        calc_done = 0
        if form.validate_on_submit():
            check_calc_run = models.check_calculate_process(session['username'])
            if check_calc_run == 1:
                flash('Одновременный запуск расчетов по нескольким проектам невозможен, просьба дождаться завершения вычислений по другому проекту.')
            else:
                calc_started_at = datetime.now()
                calc_status = 'failed'
                app.logger.info(
                    '%s - los project %s calc started at %s',
                    log_username,
                    proj_name,
                    calc_started_at.isoformat(sep=' ', timespec='seconds'),
                )
                try:            
                    models.mark_calculation_started(session['username'], 'los', proj_name)
                    los.rnd_combi(ver)
                    # app.logger.info(f'{log_username} - los project {proj_name} rnd_combi pass')
                    los.get_vectors(ver, interval_tresh=int(form.interval_tresh_field.data), thresh_best_links=float(form.best_tresh_field.data), use_reducer=form.use_reducer.data, reducer_grade=int(form.reducer_grade_field.data))
                    app.logger.info(f'{log_username} - los project {proj_name} get_vectors pass')
                    los.add_walk_links_to_vec(ver, dist_trash=float(form.dist_trash_field.data), ped_speed=float(form.ped_speed_field.data))
                    app.logger.info(f'{log_username} - los project {proj_name} add_walk_links_to_vec pass')
                    flash('Подготовка комбинаций маршрутных связей завершена')
                    v_dict = los.vec_w_to_dict(ver)
                    app.logger.info(f'{log_username} - los project {proj_name} vec_w_to_dict pass')
                    grf = los.get_G_vec(ver)
                    app.logger.info(f'{log_username} - los project {proj_name} get_G_vec pass')
                    los.all_links_los(grf, v_dict, ver=ver, best_tresh=float(form.best_tresh_field.data), deep_limit=int(form.deep_limit_field.data)+1, change_penalty=int(form.change_penalty_field.data))
                    app.logger.info(f'{log_username} - los project {proj_name} all_links_los pass')
                    flash('Расчет уровня обслуживания завершен')
                    # print('all_links_los done')
                    los.los_agg_allcalc(ver)
                    # print('los_agg_allcalc done')
                    app.logger.info(f'{log_username} - los project {proj_name} los_agg_allcalc pass')
                    los.get_ref_links(ver)
                    # print('get_ref_links done')
                    app.logger.info(f'{log_username} - los project {proj_name} get_ref_links pass')

                    if prj_info['status_src'] == 4:
                        if form.rnd_type_field.data == 'Да':
                            los.do_los_rich(ver, 'day')
                            app.logger.info(f'{log_username} - los project {proj_name} do_los_rich pass')
                            los.flow_distrib(ver, 'day')
                            # print('flow_distrib done')
                            app.logger.info(f'{log_username} - los project {proj_name} flow_distrib pass')
                            if prj_info['status_geo'] == 3:
                                los.indicators_agg_allcalc_flow(ver=ver, period='day', ped_speed=float(form.ped_speed_field.data), max_dist_walk=float(form.max_dist_walk_field.data))
                                app.logger.info(f'{log_username} - los project {proj_name} indicators_agg_allcalc_flow pass')
                            else:
                                los.indicators_agg_allcalc_flow(ver=ver, period='day')
                                app.logger.info(f'{log_username} - los project {proj_name} indicators_agg_allcalc_flow pass')
                        else:
                            los.do_los_rich(ver, form.calc_period_field.data)
                            app.logger.info(f'{log_username} - los project {proj_name} do_los_rich pass')
                            los.flow_distrib(ver, form.calc_period_field.data)
                            app.logger.info(f'{log_username} - los project {proj_name} flow_distrib pass')
                            if prj_info['status_geo'] == 3:
                                los.indicators_agg_allcalc_flow(ver=ver, period=form.calc_period_field.data, ped_speed=float(form.ped_speed_field.data), max_dist_walk=float(form.max_dist_walk_field.data))
                                app.logger.info(f'{log_username} - los project {proj_name} indicators_agg_allcalc_flow pass')
                            else:
                                los.indicators_agg_allcalc_flow(ver=ver, period=form.calc_period_field.data)
                                app.logger.info(f'{log_username} - los project {proj_name} indicators_agg_allcalc_flow pass')
                    else:
                        if prj_info['status_geo'] == 3:
                            los.indicators_agg_allcalc_carry(ver=ver, ped_speed=float(form.ped_speed_field.data), max_dist_walk=float(form.max_dist_walk_field.data))
                            app.logger.info(f'{log_username} - los project {proj_name} indicators_agg_allcalc_carry pass')
                        else:
                            los.indicators_agg_allcalc_carry(ver=ver)
                            app.logger.info(f'{log_username} - los project {proj_name} indicators_agg_allcalc_carry pass')

                    if prj_info['status_geo'] == 3 and prj_info['status_src'] == 4:
                        los.geom_gen(ver, True)
                        app.logger.info(f'{log_username} - los project {proj_name} geom_gen pass')
                    elif prj_info['status_geo'] == 3 and prj_info['status_src'] == 3:
                        los.geom_gen(ver, False)
                        app.logger.info(f'{log_username} - los project {proj_name} geom_gen pass')

                    flash('Расчет интегральных показателей завершен')
                    los.remove_temp(ver)
                    models.los_commit_update_status_prj(
                        session['username'], proj_name, models.LOS_STATUS_COMPLETED
                    )
                    calc_done = 1
                    calc_status = 'completed'
                except Exception:
                    app.logger.exception(
                        '%s - los project %s calculate error', log_username, proj_name
                    )
                    try:
                        models.los_commit_update_status_prj(
                            session['username'], proj_name, models.LOS_STATUS_FAILED
                        )
                    except Exception:
                        app.logger.exception(
                            '%s - los project %s failed to persist calculation error status',
                            log_username,
                            proj_name,
                        )
                    flash('При расчете произошла ошибка. Вы можете повторить расчет или удалить проект.')
                    return redirect(url_for('project_calc', proj_name=proj_name))
                finally:
                    calc_finished_at = datetime.now()
                    app.logger.info(
                        '%s - los project %s calc finished at %s; status=%s; duration_seconds=%.3f',
                        log_username,
                        proj_name,
                        calc_finished_at.isoformat(sep=' ', timespec='seconds'),
                        calc_status,
                        (calc_finished_at - calc_started_at).total_seconds(),
                    )

        return render_template('los_project_calc.html', ver=ver, title=proj_name, form=form, calc_done=calc_done,
                                show_flow_result=form.dist_trash_field.data, flow_data=flow_data)
    else:
        return redirect(url_for('login'))


@app.route('/project_info/<proj_name>', methods=['GET'])
def project_info(proj_name):
    if 'loggedin' in session:
        log_username = session['username']
        app.logger.info(f'{log_username} - los project {proj_name} get info')
        _, path_src = path_gen(session['username'], 'los', proj_name, False)
        prj_info = models.los_get_one_project(session['username'], proj_name)
        flow_stats = None
        unserved_correspondences_available = False
        if prj_info['status_src'] == 3:
            indicators = get_df(path_src, proj_name, 'indicators_agg_carry')
            flow_data = False
        elif prj_info['status_src'] == 4:
            indicators = get_df(path_src, proj_name, 'indicators_agg_flow')
            flow_data = True
            flow_stats_path = path_src + f'flow_distribution_stats_{proj_name}.csv'
            if os.path.isfile(flow_stats_path):
                flow_stats = get_df(path_src, proj_name, 'flow_distribution_stats').iloc[0].to_dict()
                if 'unserved_correspondences_count' not in flow_stats:
                    flow_stats['unserved_correspondences_count'] = flow_stats.get('od_pairs_without_los', 0)
            unserved_correspondences_available = os.path.isfile(
                path_src + f'unserved_correspondences_{proj_name}.csv'
            )
        indicators = indicators.iloc[:1, :]
        los_histo_path = url_for('static', filename=os.path.relpath(
            path_src + 'los_time_histo.png', app.static_folder).replace(os.sep, '/'))
        los_sdp_histo_path = url_for('static', filename=os.path.relpath(
            path_src + 'los_spd_histo.png', app.static_folder).replace(os.sep, '/'))
        flowlos_histo_path = url_for('static', filename=os.path.relpath(
            path_src + 'flow_los_time_histo.png', app.static_folder).replace(os.sep, '/'))
        flowlos_sdp_histo_path = url_for('static', filename=os.path.relpath(
            path_src + 'flow_los_spd_histo.png', app.static_folder).replace(os.sep, '/'))
        if prj_info['status_geo'] == 3:
            show_geo = True
        else:
            show_geo = False
        # try:
        _, userpath_out = path_gen(session['username'], 'los', proj_name, False)
        if os.path.isfile(userpath_out + 'los_time_histo.png') == False:
            def run_histos():
                df_los = get_df(userpath_out, proj_name, 'los')
                df_los = df_los[df_los.los>0]
                los.create_histo(df_los, userpath_out, 'los', 'los_time_histo', 
                            'Гистограмма распределения интегрального времени в пути (все связи)', 
                            'Время в пути (ч)', 'Количество связей')
                los.create_histo(df_los, userpath_out, 'los_speed', 'los_spd_histo', 
                            'Гистограмма распределения интегральной скорости (все связи)',
                            'Скорость (км/ч)', 'Количество связей')
                if flow_data:
                    df_agg = get_df(userpath_out, proj_name, 'los_agg_by_flow_cluster')
                    los.create_histo(df_agg, userpath_out, 'los_by_flow', 'flow_los_time_histo', 
                                'Гистограмма распределения интегрального времени в пути (пассажиропотоки)', 
                                'Время в пути (ч)', 'Количество связей')
                    los.create_histo(df_agg, userpath_out, 'los_spd_by_flow', 'flow_los_spd_histo', 
                                'Гистограмма распределения интегральной скорости (пассажиропотоки)', 
                                'Скорость (км/ч)', 'Количество связей')
                else:
                    df_agg = get_df(userpath_out, proj_name, 'los_agg_by_carry_cluster')
                    los.create_histo(df_agg, userpath_out, 'los_by_carry', 'flow_los_time_histo', 
                                'Гистограмма распределения интегрального времени в пути (провозная способность)', 
                                'Время в пути (ч)', 'Количество связей')
                    los.create_histo(df_agg, userpath_out, 'los_spd_by_carry', 'flow_los_spd_histo', 
                                'Гистограмма распределения интегральной скорости (провозная способность)', 
                                'Скорость (км/ч)', 'Количество связей')
            run_histos()
        # except Exception:
        #     app.logger.info(f'{log_username} - los project {proj_name} error create_histo method')
        #     return redirect(url_for('internal_error'))
        return render_template('los_project_info.html', title=proj_name, indicators=indicators, 
                                los_histo_path=los_histo_path, los_sdp_histo_path=los_sdp_histo_path,
                                flowlos_histo_path=flowlos_histo_path, flowlos_sdp_histo_path=flowlos_sdp_histo_path,
                                show_geo=show_geo, flow_data=flow_data, flow_stats=flow_stats,
                                unserved_correspondences_available=unserved_correspondences_available)
    else:
        return redirect(url_for('login'))


@app.route('/igeomtr/<proj_name>')
def send_geom_los_tr(proj_name):
    _, path_src = path_gen(session['username'], 'los', proj_name, False)
    agg_types = ['cluster', 'tr_district']
    prj_info = models.los_get_one_project(session['username'], proj_name)
    m = los.make_los_map_tr(path_src, proj_name, prj_info)
    iframe = m.get_root()._repr_html_()
    return render_template('igeom_tr.html', iframe=iframe)


@app.route('/igeomcl/<proj_name>')
def send_geom_los_cl(proj_name):
    path_geom_src, path_cluster_src = path_gen(session['username'], 'los', proj_name, False)
    agg_types = ['cluster', 'tr_district']
    prj_info = models.los_get_one_project(session['username'], proj_name)
    cl_color_range, m = los.make_los_map_cl(path_geom_src, path_cluster_src, proj_name, prj_info)
    iframe = m.get_root()._repr_html_()
    return render_template('igeom_cl.html', iframe=iframe, cl_color_range=cl_color_range)


@app.route('/igeomwa/<proj_name>')
def send_geom_walk_avail(proj_name):
    path_src_data, path_processed_data = path_gen(session['username'], 'los', proj_name, False)
    prj_info = models.los_get_one_project(session['username'], proj_name)
    m = los.make_walk_avail_map(path_src_data, path_processed_data, proj_name, prj_info)
    iframe = m.get_root()._repr_html_()
    return render_template('igeom_wa.html', iframe=iframe)


class ShowTargetForm(FlaskForm):
    tr_source_field = SelectField()
    submit = SubmitField("Обновить")

@app.route('/igeomtr_detailed/<proj_name>', methods = ['GET', 'POST'])
def send_geom_los_tr_detailed(proj_name):
    _, path_src = path_gen(session['username'], 'los', proj_name, False)
    prj_info = models.los_get_one_project(session['username'], proj_name)
    req_data = request.args.to_dict()
    tr_source, targets, m = los.make_los_map_tr_separate(path_src, prj_info, req_data, proj_name)
    iframe = m.get_root()._repr_html_()
    if tr_source == None:
        form = ShowTargetForm(tr_source_field=targets[0])
    else:
        form = ShowTargetForm(tr_source_field=tr_source)
    form.tr_source_field.choices = targets
    if form.validate_on_submit():
        return redirect(url_for('send_geom_los_tr_detailed', proj_name=proj_name, tr_source=form.tr_source_field.data))

    return render_template('igeom_tr_detailed.html', iframe=iframe, form=form)


class ShowIdStopForm(FlaskForm):
    id_stop_field = SelectField()
    submit = SubmitField("Обновить")

@app.route('/isochrone/<proj_name>', methods = ['GET', 'POST'])
def show_los_iso(proj_name):
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    path_src, path_out = path_gen(session['username'], 'los', proj_name, False)
    prj_info = models.los_get_one_project(session['username'], proj_name)
    req_data = request.args.to_dict()

    if 'id_stop' not in req_data:
        id_stop = None  
    else:
        id_stop = req_data['id_stop']

    targets, m = los.make_isochrone(path_src, path_out, proj_name, id_stop)
    # print(targets)
    # print(type(targets))

    if len(targets) == 0:
        form = ShowIdStopForm()
        form.id_stop_field.choices = []
        iframe = m.get_root()._repr_html_()
        return render_template('igeom_isochrone.html', iframe=iframe, form=form)

    if id_stop == None:
        form = ShowIdStopForm(id_stop_field=targets[0])
    else:
        form = ShowIdStopForm(id_stop_field=id_stop)

    form.id_stop_field.choices = targets

    try: 
        iframe = m.get_root()._repr_html_()
    except Exception:
        targets, m = los.make_isochrone(path_src, path_out, proj_name, None)
        if len(targets) == 0:
            form = ShowIdStopForm()
            form.id_stop_field.choices = []
        else:
            form = ShowIdStopForm(id_stop_field=targets[0])
            form.id_stop_field.choices = targets
        iframe = m.get_root()._repr_html_()

    if form.validate_on_submit():
        return redirect(url_for('show_los_iso', proj_name=proj_name, id_stop=form.id_stop_field.data))

    return render_template('igeom_isochrone.html', iframe=iframe, form=form)
# <!--
# <div>
# <button onclick="show_more('adata_8')" class="tu_bottom">
#  Изохрона</button> уровня обслуживания в разрезе остановочных пунктов (загрузка и обновление данных может занять продолжительное время)
#  <div id='adata_8' class='adata_map'>
#  <iframe src="{{ url_for('show_los_iso', proj_name=title) }}" width="100%" height="570" scrolling="no"></iframe></div>
# </div>-->


##### Эффективность #####
@app.route('/efficiency/', methods = ['POST', 'GET'])
def eff_projects():
    headers = ['Код проекта', 'Описание проекта', 'Дата создания', 'Статус проекта']
    if 'loggedin' in session:
        log_username = session['username']
        proj_data = models.eff_get_projects(session['username'])
        if len(proj_data) == 0: 
            not_filled = 0 
        else: not_filled = 1
        proj_names = [name[0] for name in proj_data]
        form = CreateProject()
        if form.validate_on_submit():
            if form.proj_name.data in proj_names:
                flash('Проект с указанным кодом уже существует. Выберите другой код.')
            else:
                account = models.get_acc_by_username(session['username'])
                check_expire = account['expire_date'] > date.today()
                if check_expire == False:
                    flash('Необходимо продлить подписку для создания новых проектов.')
                cur_disk = check_disk_size(session['username'])
                user_limits = models.get_user_limits(account['tarif_name'])
                disk_limit = user_limits['disk_limit']
                if cur_disk > disk_limit:
                    flash(f'Вы достигли лимита объема хранения файлов (не более: {disk_limit} Мб).')
                else:
                    models.eff_commit_new_project(session['username'], form.proj_name.data, form.proj_descr.data, 
                    date.today().year, date.today().month, date.today().day)
                    path_gen(session['username'], 'eff', form.proj_name.data, True)
                    app.logger.info(f'{log_username} - create eff project {form.proj_name.data}')
                    return redirect(url_for('eff_src_loader', proj_name=form.proj_name.data))

        return render_template('eff_projects.html', username=session['username'], title="Эффективность", form=form, 
                                headers=headers, rows=proj_data, not_filled=not_filled)
    else:
        return redirect(url_for('login'))


@app.route('/eff_loader/<proj_name>', methods=['GET', 'POST'])
def eff_src_loader(proj_name):
    load_file_state = {
            0:['rnd_short', 'Загрузите описание маршрутов'],
            1:['routes_flow', 'Загрузите корреспонденции на машрутах'],
            2:['carry', 'Загрузите провозные возможности на машрутах']
            }
    if 'loggedin' in session:
        log_username = session['username']
        prj_info = models.eff_get_one_project(session['username'], proj_name)
        if prj_info['status_prj'] == 1:
            return redirect(url_for('eff_project_calc', proj_name=proj_name))
        elif prj_info['status_prj'] == models.EFF_STATUS_CALCULATING:
            return redirect(url_for('eff_projects'))
        elif prj_info['status_prj'] == models.EFF_STATUS_COMPLETED:
            return redirect(url_for('eff_project_info', proj_name=proj_name))
        elif prj_info['status_prj'] == models.EFF_STATUS_FAILED:
            flash('Предыдущий расчет завершился с ошибкой. Вы можете запустить его повторно или удалить проект.')
            return redirect(url_for('eff_project_calc', proj_name=proj_name))

        form = UploadForm()
        upload_folder, _ = path_gen(session['username'], 'eff', proj_name, False)
        load_status = prj_info['status_src']
        load_done = 0
        if prj_info['status_prj'] == 1:  load_done = 1
        if load_status < 3: flash(load_file_state[load_status][1])
        if form.validate_on_submit() and load_status < 3:
            filename_check = secure_filename(form.file.data.filename)
            if allowed_file(filename_check):
                account = models.get_acc_by_username(session['username'])
                user_limits = models.get_user_limits(account['tarif_name'])
                net_limit = user_limits[2]*10 # !!! УВЕЛИЧИЛ ЛИМИТ!!!!!!!!!!!!!!!!!!!!!!!!!11
                prefix_name = load_file_state[prj_info['status_src']][0]
                filename = prefix_name + '_' + proj_name + '.csv'
                form.file.data.save(upload_folder + filename)
                load_result, msg = check_source_data(upload_folder, filename, prefix_name, net_limit)
                if load_result:
                    session['_flashes'].clear()
                    flash(msg)
                    load_status = prj_info['status_src'] + 1
                    models.eff_commit_update_status_src(session['username'], proj_name, load_status)
                    if load_status == 3 and form.flow_load_field.data == True:
                        load_done = 1
                        models.eff_commit_update_status_prj(session['username'], proj_name, 1)
                        app.logger.info(f'{log_username} - eff project {proj_name} src loaded')
                    if load_status == 1 and form.flow_load_field.data == False:
                        load_done = 1
                        models.eff_commit_update_status_prj(session['username'], proj_name, 1)
                        app.logger.info(f'{log_username} - eff project {proj_name} src loaded')

                else:
                    os.remove(upload_folder + filename)
                    flash(msg)
            else:
                flash('Требуется файл в формате csv (кодировка UTF-8)')

        return render_template('eff_src_loader.html', form=form, title=proj_name, load_done=load_done)

    else:
        return redirect(url_for('login'))


class RunEffCalcForm(FlaskForm):
    calc_double_rnd_field = BooleanField("CDF", default=False)
    hard_calc_doubl_field = BooleanField("SCD", default=False)
    # calc_renew_field = BooleanField("CRF", default=False)
    # calc_line_load_rnd_field = BooleanField("CLLF", default=False)
    submit = SubmitField(label="Рассчитать", id='calc_butt')

@app.route('/eff_project_calc/<proj_name>', methods=['GET', 'POST']) #
def eff_project_calc(proj_name):
    if 'loggedin' in session:
        log_username = session['username']
        form = RunEffCalcForm()
        models.recover_stale_calculations(session['username'])
        prj_info = models.eff_get_one_project(session['username'], proj_name)
        if proj_name[0] != '_':
            ver = '_' + proj_name
        else:
            ver = proj_name

        if prj_info['status_src'] == 3:
            show_flow = True
        else:
            show_flow = False

        # для того, чтоб избежать нового расчета при перезагрузке страницы
        if prj_info['status_prj'] == models.EFF_STATUS_CALCULATING:
            return redirect(url_for('eff_projects'))
        if prj_info['status_prj'] == models.EFF_STATUS_COMPLETED:
            return redirect(url_for('eff_project_info', proj_name=proj_name))


        calc_done = 0
        if form.validate_on_submit():
            check_calc_run = models.check_calculate_process(session['username'])
            if check_calc_run == 1: 
                flash('Одновременный запуск расчетов по нескольким проектам невозможен, просьба дождаться завершения вычислений по другому проекту.')
            else:
                app.logger.info(f'{log_username} - eff project {proj_name} calc started')
                try:
                    models.mark_calculation_started(session['username'], 'eff', proj_name)
                    app.logger.info(f'{log_username} - eff project {proj_name} calc started')
                    eff.rnd_combi_prepare(ver)
                    eff.routes_doubl(ver, rnd=form.calc_double_rnd_field.data, hard_calc=form.hard_calc_doubl_field.data)
                    if prj_info['status_src'] == 3:
                        eff.calc_line_load(ver)
                        eff.calc_static_load(ver)
                        eff.calc_irregularity(ver)
                        eff.calc_renew(ver) #, rnd=form.calc_renew_field.data)
                        eff.calc_dynamic(ver)
                    flash('Расчет показателей эффективности завершен')
                    models.eff_commit_update_status_prj(
                        session['username'], proj_name, models.EFF_STATUS_COMPLETED
                    )
                    app.logger.info(f'{log_username} - eff project {proj_name} calc finished')
                    calc_done = 1
                except Exception:
                    app.logger.exception(
                        '%s - eff project %s calculate error', log_username, proj_name
                    )
                    try:
                        models.eff_commit_update_status_prj(
                            session['username'], proj_name, models.EFF_STATUS_FAILED
                        )
                    except Exception:
                        app.logger.exception(
                            '%s - eff project %s failed to persist calculation error status',
                            log_username,
                            proj_name,
                        )
                    flash('При расчете произошла ошибка. Вы можете повторить расчет или удалить проект.')
                    return redirect(url_for('eff_project_calc', proj_name=proj_name))

        return render_template('eff_project_calc.html', title=proj_name, form=form, calc_done=calc_done, show_flow=show_flow)
    else:
        return redirect(url_for('login'))


class ShowRNDForm(FlaskForm):
    rnd_periods = ['h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14','h15','h16', 'h17',
                 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'day']
    rnd_period_field = SelectField(choices = rnd_periods, default='day')
    rnd_variant_field = SelectField()
    submit = SubmitField("Обновить")

@app.route('/eff_project_info/', methods=['GET', 'POST'])
def eff_project_info():
    if 'loggedin' in session:
        log_username = session['username']
        req_rnd = request.args.to_dict()
        proj_name = req_rnd['proj_name']
        prj_info = models.eff_get_one_project(session['username'], proj_name)
        if proj_name[0] != '_':
            ver = '_' + proj_name
        else:
            ver = proj_name

        app.logger.info(f'{log_username} - eff project {proj_name} get info')
        _, path_src = path_gen(session['username'], 'eff', proj_name, False)
        # eff.get_paths(session['username'], proj_name, False)
        df_doubl = get_df(path_src, proj_name, 'matrix_doubl')
        df_doubl['маршрут'] = df_doubl['маршрут'].astype('object')
        double_string = df_doubl.style.pipe(eff.df_styler_double).to_html()

        if prj_info['status_src'] == 3:
            show_flow = True
        else:
            show_flow = False
        # если есть данные по потокам и провозной    
        if show_flow:
            df_irregularity = get_df(path_src, proj_name, 'routes_irregularity')
            df_irregularity['route'] = df_irregularity['route'].astype('object')
            irregularity_string = df_irregularity.style.pipe(eff.df_styler_irregularity).to_html()
            df_renew = get_df(path_src, proj_name, 'routes_renew')
            df_renew['маршрут'] = df_renew['маршрут'].astype('object')
            renew_string = df_renew.style.pipe(eff.df_styler_renew).to_html()
            df_dynamic = get_df(path_src, proj_name, 'routes_dynamic')
            df_dynamic['маршрут'] = df_dynamic['маршрут'].astype('object')
            dynamic_string = df_dynamic.style.pipe(eff.df_styler_renew).to_html()
            df_all_rnd = get_df(path_src, proj_name, 'routes_flow_ext')
            all_rnd = df_all_rnd['rnd'].unique().tolist()

            if 'rnd' not in req_rnd:
                rnd = all_rnd[0]  
                period = 'day'
            else:
                rnd = req_rnd['rnd']  
                period = req_rnd['period']

            load_string = eff.get_rnd_line_load(ver, rnd)
            mean_string  = eff.load_mean_distance(ver)
            img_path = eff.corrs_show(ver, period, rnd)
            if img_path == None: img_path = 'static/img/error_eff_crt.png'
            form = ShowRNDForm(rnd_variant_field=rnd, rnd_period_field=period)
            form.rnd_variant_field.choices = all_rnd

            histo_renew = f'segments_renew{ver}'
            histo_renew_path = url_for('static', filename=os.path.relpath(
                path_src + histo_renew + '.png', app.static_folder).replace(os.sep, '/'))
            df_seg_renew = eff.get_rnd_line_renew(path_src, ver, rnd)
            histo_renew_available = eff.create_histo_renew(df_seg_renew, path_src, histo_renew, f'Суточная загрузка перегонов на маршруте {rnd} (с учетом направления движения)', 
                        'Межостановочные перегоны (в порядке следования маршрута)', 'Коэффициент наполнения')

            if form.validate_on_submit():
                return redirect(url_for('eff_project_info', proj_name=proj_name, period=form.rnd_period_field.data, rnd=form.rnd_variant_field.data) + '#route_info')
            return render_template('eff_project_info.html', ver=ver, rnd_show=rnd, title=proj_name, form=form, table_doubl=double_string, 
                                    table_irregularity=irregularity_string, table_renew=renew_string, table_dynamic=dynamic_string,
                                    load_string=load_string, mean_string=mean_string, img_path=img_path, show_flow=show_flow,
                                    histo_renew_path=histo_renew_path, histo_renew_available=histo_renew_available)
        else:
            return render_template('eff_project_info.html', ver=ver, title=proj_name, table_doubl=double_string, show_flow=show_flow)
    else:
        return redirect(url_for('login'))


### ПРОЕКТИРОВАНИЕ ###

GEO_COLOR_PATTERN = re.compile(r'^#[0-9A-Fa-f]{6}$')


def _random_geo_layer_color():
    """Возвращает случайный цвет слоя в формате #RRGGBB"""
    hue = secrets.randbelow(360) / 360
    red, green, blue = colorsys.hls_to_rgb(hue, 0.52, 0.68)
    return f'#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}'


def _ensure_geo_layer_colors(layers_description):
    """Добавляет цвет новым и существующим описаниям слоев"""
    changed = False
    for layer_description in layers_description.values():
        color = layer_description.get('color')
        if not isinstance(color, str) or GEO_COLOR_PATTERN.fullmatch(color) is None:
            layer_description['color'] = _random_geo_layer_color()
            changed = True
        else:
            normalized_color = color.upper()
            if normalized_color != color:
                layer_description['color'] = normalized_color
                changed = True
    return changed


def _ensure_geo_layer_user_fields(layers_description):
    """Удаляет из описания слоев внутренние поля геометрии"""
    changed = False
    for layer_description in layers_description.values():
        fields = layer_description.get('fields')
        if not isinstance(fields, dict):
            layer_description['fields'] = {}
            changed = True
            continue
        for field_name in list(fields):
            if is_geo_technical_field(field_name):
                del fields[field_name]
                changed = True
    return changed


def _load_geo_layers_description(config_file_path):
    """Загружает конфигурацию и мигрирует старые описания слоев"""
    layers_description = sl_json(config_file_path, None, 'load')
    if (not isinstance(layers_description, dict)
            or not all(isinstance(value, dict) for value in layers_description.values())):
        raise ValueError('Некорректное описание слоёв')
    changed = _ensure_geo_layer_colors(layers_description)
    changed = _ensure_geo_layer_user_fields(layers_description) or changed
    if changed:
        sl_json(config_file_path, layers_description, 'save')
    return layers_description


def _sync_geo_layer_load_states(layers_description, path_src, proj_name):
    """Синхронизирует load_data с фактическим наличием CSV слоя"""
    changed = False
    for layer_name, layer_description in layers_description.items():
        filename = f'{layer_name}_{proj_name}.csv'
        load_data = 'Y' if os.path.isfile(path_src + filename) else 'N'
        if layer_description.get('load_data') != load_data:
            layer_description['load_data'] = load_data
            changed = True
    return changed


def _apply_geo_layer_colors(layers_description, layer_names, layer_colors):
    """Проверяет данные формы и применяет цвета ко всем слоям проекта"""
    if (len(layer_names) != len(layer_colors)
            or len(layer_names) != len(layers_description)
            or set(layer_names) != set(layers_description)):
        return False
    for layer_name, layer_color in zip(layer_names, layer_colors):
        if GEO_COLOR_PATTERN.fullmatch(layer_color or '') is None:
            return False
    for layer_name, layer_color in zip(layer_names, layer_colors):
        layers_description[layer_name]['color'] = layer_color.upper()
    return True


@app.route('/geo/', methods=['GET', 'POST'])
@_validate_geo_request
def geo_projects():
    headers = ['Код проекта', 'Описание проекта', 'Дата создания', 'Статус проекта']
    if 'loggedin' in session:
        proj_data = models.geo_get_projects(session['username'])
        if len(proj_data) == 0: 
            not_filled = 0 
        else: not_filled = 1
        proj_names = [name[0] for name in proj_data]
        form = CreateProject()
        if form.validate_on_submit():
            if form.proj_name.data in proj_names:
                flash('Проект с указанным кодом уже существует. Выберите другой код.')
            else:
                account = models.get_acc_by_username(session['username'])
                check_expire = account['expire_date'] > date.today()
                if check_expire == False: 
                    flash('Необходимо продлить подписку для создания новых проектов.')
                else:
                    cur_disk = check_disk_size(session['username'])
                    user_limits = models.get_user_limits(account['tarif_name'])
                    disk_limit = user_limits['disk_limit']
                    if cur_disk > disk_limit:
                        flash(f'Вы достигли лимита объема хранения файлов (не более: {disk_limit} Мб).')
                    else:
                        models.geo_commit_new_project(session['username'], form.proj_name.data, form.proj_descr.data, 
                        date.today().year, date.today().month, date.today().day)
                        path_gen(session['username'], 'geo', form.proj_name.data, True)
                        user_name_log = session['username']
                        app.logger.info(f'{user_name_log} - create geo project {form.proj_name.data}')
                        return redirect(url_for('geo_src_loader', proj_name=form.proj_name.data))

        return render_template('geo_projects.html', username=session['username'], title="Проектирование", form=form, 
                                headers=headers, rows=proj_data, not_filled=not_filled)
    else:
        return redirect(url_for('login'))


class UploadFormGeo(FlaskForm):
    file = FileField()
    new_layer = BooleanField(default=False)
    submit = SubmitField(u"Загрузить/Создать")

@app.route('/geo_src_loader/<proj_name>', methods=['GET', 'POST'])
@_validate_geo_request
def geo_src_loader(proj_name):
    ''' Загрузка исходных файлов для модуля проектирования '''
    # конфиг проекта, инициируем и сохранеям при первом обращении
    # project_info = {} #{"stops": 1, "routes": 1, "districts": 1}
    # geom - тип геометрии при создании объктов, none - метод создания не рисование
    layers_description = {
    "stops": {"fields": {"id": "Код", "name": "Наименование"}, "layer_name":"Остановки", "zIndex": 103, "geom": "Point", "load_data": "Y", "color": _random_geo_layer_color()},
    "districts": {"fields": {"id": "Код"}, "layer_name":"Районы", "zIndex": 102, "geom": "Polygon", "load_data": "Y", "color": _random_geo_layer_color()},
    "routes": {"fields": {"id": "Код", "name": "Наименование"}, "layer_name":"Маршруты", "zIndex": 101, "geom": "none", "load_data": "Y", "color": _random_geo_layer_color()}
    }
    load_file_state = {
            0:['stops', 'Загрузите слой остановочных пунктов'],
            1:['routes', 'Загрузите слой маршрутов'],
            2:['districts', 'Загрузите слой транспортных райнов']
            }

    if 'loggedin' in session:
        log_username = session['username']
        prj_info = models.geo_get_one_project(session['username'], proj_name)
        if prj_info['status_src'] > 2:
            return redirect(url_for('geo_src_loader_add', proj_name=proj_name))

        form = UploadFormGeo()
        upload_folder, _ = path_gen(session['username'], 'geo', proj_name, False)

        # сохраняем конфиг при первой загруке
        config_file_path = upload_folder + 'layers_description.json'
        if os.path.isfile(config_file_path) is False:
            sl_json(config_file_path, layers_description, 'save')
        else:
            layers_description = _load_geo_layers_description(config_file_path)

        load_status = prj_info['status_src']
        load_done = 0

        flash(load_file_state[prj_info['status_src']][1])
        if prj_info['status_prj'] == 1:  load_done = 1
        if form.validate_on_submit() and load_status < 3:
            prefix_name = load_file_state[prj_info['status_src']][0]
            if form.new_layer.data == True:
                # загружаем конфиг и вносим правки
                layers_description = sl_json(config_file_path, None, 'load')
                layers_description[prefix_name]['load_data'] = "N"
                sl_json(config_file_path, layers_description, 'save')
                # апдейтим статус загрузки основных слоев
                load_status = prj_info['status_src'] + 1
                models.geo_commit_update_status_src(session['username'], proj_name, load_status)
                if load_status < 3:
                    session['_flashes'].clear() 
                    flash(load_file_state[load_status][1])
                form.new_layer.data = False
            else:
                if form.file.data != None:
                    filename_check = secure_filename(form.file.data.filename)
                    if allowed_file(filename_check):
                        filename = prefix_name + '_' + proj_name + '.csv'
                        form.file.data.save(upload_folder + filename)
                        load_result, msg, fields = check_geo_source(upload_folder, filename, prefix_name)
                        if load_result:
                            session['_flashes'].clear()
                            load_status = prj_info['status_src'] + 1
                            if load_status < 3:
                                flash(f'{msg}. {load_file_state[load_status][1]}')
                            else:
                                flash(msg)
                            # апдейтим статус загрузки основных слоев
                            models.geo_commit_update_status_src(session['username'], proj_name, load_status)
                            # project_info[prefix_name] = 1
                            # добавляем пользовательские поля, если они есть
                            for f in fields:
                                if (not is_geo_technical_field(f)
                                        and f not in layers_description[prefix_name]['fields']):
                                    layers_description[prefix_name]['fields'][f] = f
                            sl_json(config_file_path, layers_description, 'save')
                        else:
                            os.remove(upload_folder + filename)
                            session['_flashes'].clear()
                            if load_status < 3:
                                flash(f'{msg}. {load_file_state[load_status][1]}')
                            else:
                                flash(msg)
                    else:
                        flash('Требуется файл в формате csv (кодировка UTF-8)')

            if load_status == 3:
                load_done = 1
                app.logger.info(f'{log_username} - geo project {proj_name} main src loaded')
        return render_template('geo_src_loader.html', form=form, title=proj_name, load_done=load_done)
    else:
        return redirect(url_for('login'))


class UploadAdditionalGeo(FlaskForm):
    file = FileField()
    new_layer = BooleanField(default=False)
    layer_name = StringField(u'Наименование слоя', validators=[InputRequired(), Length(max=15)])
    field_description = StringField(u'Описание полей (пустой слой)', validators=[Length(max=300)])
    geom_type = [('Point', 'Точка'), ('LineString', 'Линия'), ('Polygon', 'Полигон')]
    geom_type_field = SelectField(u'Тип геометрии (пустой слой)', choices = geom_type, default='Point')
    submit = SubmitField(u"Загрузить/Создать")

@app.route('/geo_src_loader_add/<proj_name>', methods=['GET', 'POST'])
@_validate_geo_request
def geo_src_loader_add(proj_name):
    ''' Загружаем доп. слои '''
    if 'loggedin' in session:
        layers_limit = 10
        prj_info = models.geo_get_one_project(session['username'], proj_name)
        if prj_info['status_prj'] == 1:
            return redirect(url_for('geo_project_calc', proj_name=proj_name))
        form_additional = UploadAdditionalGeo()
        upload_folder, _ = path_gen(session['username'], 'geo', proj_name, False)
        log_username = session['username']
        z_level = 1
        config_file_path = upload_folder + 'layers_description' + '.json'
        flash('')
        session['_flashes'].clear()
        flash('При отсутствии дополнительных слоев нажмите "Завершить загрузку"')
        # temp_layer_names = []

        if form_additional.validate_on_submit():
            layers_description = _load_geo_layers_description(config_file_path)
            layer_name = form_additional.layer_name.data.replace(" ", "")
            try:
                geo._validate_layer_name(layer_name)
            except geo.GeoValidationError as error:
                flash(str(error), 'error')
                return render_template('geo_src_loader_add.html', form_additional=form_additional, title=proj_name)
            fields_info = form_additional.field_description.data.replace(" ", "").split(",")
            temp_fields = {}
            # Для пустых слоев
            if form_additional.new_layer.data == True:
                temp_fields['id'] = "Код"
                for f in fields_info:
                    if f and f != 'id' and not is_geo_technical_field(f):
                        temp_fields[f] = f
                # загружаем конфиг и вносим правки
                if layer_name not in layers_description:
                    if len(layers_description) < layers_limit:
                        # temp_layer_names.append(layer_name)
                        # print(temp_layer_names)
                        # print('new one')
                        # print(layers_description.keys())
                        layers_description[layer_name] = {"fields": temp_fields, "layer_name": layer_name, "zIndex": z_level, "geom": form_additional.geom_type_field.data, "load_data": "N", "color": _random_geo_layer_color()}
                        sl_json(config_file_path, layers_description, 'save')
                        z_level += 1
                        session['_flashes'].clear()
                        flash('Слой успешно создан')
                        form_additional.new_layer.data = False
                        form_additional.layer_name.data = ''
                        form_additional.geom_type_field.data = 'Point'
                        form_additional.field_description.data = ''
                        app.logger.info(f'{log_username} - geo project {proj_name} additional src loaded: {layer_name}')
                    else:
                        flash(f'Достигнут максимальный предел количества слоев в проекте ({layers_limit})')
                else:
                    session['_flashes'].clear()
                    flash('Слой уже внесен, измените имя нового слоя')
            # Для загружаемых слоев
            else:
                if form_additional.file.data != None:
                    if layer_name not in layers_description:
                        if allowed_file(form_additional.file.data.filename):
                            filename = layer_name + '_' + proj_name + '.csv'
                            # prefix_name = form_additional.file.data.filename.split('.')[0].lower()[:15]
                            # filename =  + '_' + proj_name + '.csv'
                            form_additional.file.data.save(upload_folder + filename)
                            if len(layers_description) < layers_limit:
                                load_result, msg, fields, geom_type = check_geo_add_source(upload_folder, filename, layer_name)
                                if load_result:
                                    # id должно быть в массиве
                                    for f in fields:
                                        if not is_geo_technical_field(f):
                                            temp_fields[f] = f
                                    # загружаем конфиг и вносим правки
                                    layers_description = _load_geo_layers_description(config_file_path)
                                    layers_description[layer_name] = {"fields": temp_fields, "layer_name":layer_name, "zIndex": z_level, "geom": geom_type, "load_data": "Y", "color": _random_geo_layer_color()}
                                    sl_json(config_file_path, layers_description, 'save')
                                    z_level += 1
                                    form_additional.new_layer.data = False
                                    form_additional.layer_name.data = ''
                                    form_additional.geom_type_field.data = 'Point'
                                    form_additional.field_description.data = ''
                                    app.logger.info(f'{log_username} - geo project {proj_name} additional src loaded: {layer_name}')
                                    session['_flashes'].clear()
                                    flash(msg)
                                else:
                                    os.remove(upload_folder + filename)
                                    session['_flashes'].clear()
                                    flash(msg)
                            else:
                                flash(f'Достигнут максимальный предел количества слоев в проекте ({layers_limit})')
                    else:
                        session['_flashes'].clear()
                        flash('Слой уже внесен, измените имя слоя')
                else:
                    session['_flashes'].clear()
                    flash('Необходимо выбрать файл')

        return render_template('geo_src_loader_add.html', form_additional=form_additional, title=proj_name)
    else:
        return redirect(url_for('login'))


class CalcGeo(FlaskForm):
    area_name = StringField(u'Наименование городского поселения', validators=[InputRequired(), Length(max=550)])
    submit = SubmitField(u"Загрузить", id='load_grph_butt')

class MakeCluster(FlaskForm):
    couple_stop_cluster = [80, 100, 120, 140]
    csc_field = SelectField(choices =couple_stop_cluster, default=120)
    analitic_stop_cluster = [200, 225, 250, 275]
    asc_field = SelectField(choices=analitic_stop_cluster, default=225)
    gen_districts = [300, 325, 350, 375]
    gen_districts_field = SelectField(choices=gen_districts, default=325)
    k_walk = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    k_walk_field = SelectField(choices =k_walk, default=1.15)
    generate_distr = BooleanField(default=False)
    boundary_buffer_m = SelectField(choices=list(range(100, 1501, 100)), coerce=int,
                                  default=100, validators=[InputRequired()])
    submit = SubmitField(u"Рассчитать", id='make_cluster_butt')


class LayerColorsForm(FlaskForm):
    color_submit = SubmitField(u"Сохранить цвета", id='save_layer_colors')


@app.route('/geo_project_calc/<proj_name>', methods=['GET', 'POST'])
@_validate_geo_request
def geo_project_calc(proj_name):
    ''' Оперируем слоями '''
    log_username = session.get('username', '')
    try:
        # без подготовки графа зайти на карту невозможно
        if 'loggedin' in session:
            prj_info = models.geo_get_one_project(session['username'], proj_name)
            if prj_info['status_prj'] == 0:
                models.geo_commit_update_status_prj(session['username'], proj_name, 1)

            if prj_info['status_src'] == 3:
                show_geomap = 0
            elif prj_info['status_src'] == 4:
                show_geomap = 1
            
            path_src, path_out = path_gen(session['username'], 'geo', proj_name, False)
            log_username = session['username']
            config_file_path = path_src + 'layers_description' + '.json'
            layers_description = _load_geo_layers_description(config_file_path)
            table_data = {}
            for i in layers_description:
                file_name = f'{i}_{proj_name}.csv'
                if os.path.isfile(path_src + file_name):
                    file_state = 1
                else:
                    file_state = 0
                table_data[i] = [layers_description[i]['layer_name'], file_state, layers_description[i]['color']]

            analitic_cluster_name = f'geom_clusters_{proj_name}.csv'
            show_link_clusters = 0
            if os.path.isfile(path_out + analitic_cluster_name):
                show_link_clusters = 1
            matrix_name_out = f'matrix_dist_stops_{proj_name}.csv'
            show_link_matrix = 0
            if os.path.isfile(path_out + matrix_name_out):
                show_link_matrix = 1
            file_name_elements = f'routes_elements_{proj_name}.csv'
            show_ref_new_routes = 0 # пока хранится в папке src
            if os.path.isfile(path_src + file_name_elements):
                show_ref_new_routes = 1
            ref_stops_name = f'ref_stops_{proj_name}.csv' 
            show_district_boundary = os.path.isfile(path_out + f'districts_boundary_{proj_name}.csv')
            show_ref_stops = 0
            if os.path.isfile(path_out + ref_stops_name):
                show_ref_stops = 1

            form_CG = CalcGeo()
            form_MC = MakeCluster()
            if request.method == 'GET':
                calculation_settings = geo._read_calculation_settings(path_src)
                saved = calculation_settings.get('voronoi', {})
                buffer_m = saved.get('buffer_m', 100)
                if buffer_m in range(100, 1501, 100):
                    form_MC.boundary_buffer_m.data = int(buffer_m)
                parameters = calculation_settings.get('cluster_parameters', {})
                for field, value, choices in [
                        (form_MC.gen_districts_field, saved.get('radius_m'), MakeCluster.gen_districts),
                        (form_MC.csc_field, parameters.get('stop_group_radius_m'), MakeCluster.couple_stop_cluster),
                        (form_MC.asc_field, parameters.get('analytic_radius_m'), MakeCluster.analitic_stop_cluster),
                        (form_MC.k_walk_field, calculation_settings.get('k_walk'), MakeCluster.k_walk)]:
                    for choice in choices:
                        if value == choice:
                            field.data = str(choice)
                form_MC.generate_distr.data = saved.get('enabled', False)
            form_colors = LayerColorsForm()
            flash('')
            # Сохраняем выбранные пользователем цвета слоев
            if form_colors.color_submit.data and form_colors.validate_on_submit():
                layer_names = request.form.getlist('layer_name')
                layer_colors = request.form.getlist('layer_color')
                session['_flashes'].clear()
                if not _apply_geo_layer_colors(layers_description, layer_names, layer_colors):
                    flash('Не удалось сохранить цвета: проверьте значения палитры.')
                else:
                    sl_json(config_file_path, layers_description, 'save')
                    flash('Цвета слоев сохранены.')
                return redirect(url_for('geo_project_calc', proj_name=proj_name))
            # Загружаем граф
            elif form_CG.validate_on_submit():
                state, msg, G = geo.get_osm_G(log_username, path_src, proj_name, form_CG.area_name.data)
                session['_flashes'].clear()
                flash(msg)
                if state:
                    show_geomap = 1
                    models.geo_commit_update_status_src(session['username'], proj_name, 4)
                    return redirect(url_for('geo_project_calc', proj_name=proj_name))
            # Делаем кластеры
            elif form_MC.validate_on_submit():
                # Расчёт требует существующего слоя остановок.
                file_stops = f'stops_{proj_name}.csv'
                if os.path.isfile(path_src + file_stops):
                    file_districts = f'districts_{proj_name}.csv'
                    if os.path.isfile(path_src + file_districts):
                        use_draw_districts = True
                    else:
                        use_draw_districts = False
                    cluster_state, msg_cl = geo.get_cluster(log_username, proj_name, path_src, path_out, int(form_MC.csc_field.data), 
                                int(form_MC.asc_field.data), int(form_MC.gen_districts_field.data), use_draw_districts, form_MC.generate_distr.data, analitic_cluster_name,
                                k_walk=float(form_MC.k_walk_field.data),
                                boundary_buffer_m=form_MC.boundary_buffer_m.data)
                    session['_flashes'].clear()
                    flash(msg_cl)
                    if cluster_state:
                        # добавляем слой кластеров в проект
                        layers_description = _load_geo_layers_description(config_file_path)
                        if 'clusters' not in layers_description:
                            layers_description['clusters'] = {"fields": {"id": "Код ОП", "id_grp_stp": "Код группы", "id_cluster": "Код кластера"}, "layer_name":"Кластеры", "zIndex": 12, "geom": 'Point', "load_data": "Y", "color": _random_geo_layer_color()}
                            sl_json(config_file_path, layers_description, 'save')
                            table_data['clusters'] = ['Кластеры', 1, layers_description['clusters']['color']]
                        # добавляем слой сгенерированных районов в проект
                        if form_MC.generate_distr.data:
                            if 'districtsGEN' not in layers_description:
                                layers_description['districtsGEN'] = {"fields": {"id": "Код района"}, "layer_name": "Районы (В)", "zIndex": 11, "geom": 'Polygon', "load_data": "Y", "color": _random_geo_layer_color()}
                                sl_json(config_file_path, layers_description, 'save')
                                table_data['districtsGEN'] = ['Районы (В)', 1, layers_description['districtsGEN']['color']]
                        # Матрица и параметры уже сохранены вместе со слоями.
                        flash('Рассчитана матрица межостановочных расстояний')
                        return redirect(url_for('geo_project_calc', proj_name=proj_name))
                else:
                    session['_flashes'].clear()
                    flash("Слой остановок не содержит объектов")

            if form_MC.errors and 'csc_field' in request.form:
                flash('Проверьте параметры расчёта: ' + '; '.join(
                    f'{field}: {", ".join(errors)}' for field, errors in form_MC.errors.items()))

            return render_template('geo_project_calc.html', form_CG=form_CG, form_MC=form_MC, form_colors=form_colors,
                                    title=proj_name, table_data=table_data,
                                    show_geomap=show_geomap, show_link_matrix=show_link_matrix, show_link_clusters=show_link_clusters, 
                                    show_ref_new_routes=show_ref_new_routes, show_ref_stops=show_ref_stops,
                                    show_district_boundary=show_district_boundary)
        else:
            return redirect(url_for('login'))
    except Exception as app_error:
        app.logger.info(f'{log_username} - geo project {proj_name} calculate error:\n{app_error}')
        return redirect(url_for('internal_error'))

@app.route('/geoapp/<proj_name>', methods=['GET', 'POST'])
def geo_app(proj_name):
    return render_template('geo_app.html', proj_name=proj_name)


@app.route('/geo/project_info', methods = ['POST'])
@_validate_geo_request
def geo_project_info():
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if 'loggedin' in session:
        url_vars = request.args.to_dict()
        if not _geo_component(url_vars.get('project')):
            return make_response({'err_info': 'Требуется имя проекта'}, 400, headers)
        path_src, path_out = path_gen(session['username'], 'geo', url_vars["project"], False)
        if os.path.exists(path_src):
            # грузим описание проекта из целевой папки юзер/проект
            config_file_path = path_src + 'layers_description.json'
            try:
                project_info = _load_geo_layers_description(config_file_path)
            except (OSError, ValueError):
                return make_response({'err_info': 'Описание проекта повреждено'}, 400, headers)
            if _sync_geo_layer_load_states(project_info, path_src, url_vars["project"]):
                sl_json(config_file_path, project_info, 'save')
            return make_response(project_info, 200, headers)
        else:
            return make_response({'err_info':'Project not found'}, 400, headers)
    else:
        return make_response({'err_info': ''}, 400, headers)    


@app.route('/geo/get_data', methods = ['GET', 'POST'])
@_validate_geo_request
def geo_get_data():
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if 'loggedin' in session:
        ref_layer = {
                    'stops': ['Остановки', 'Points'],
                    'districts': ['Районы', 'MultiLineString'],
                    'routes': ['Маршруты', 'MultiPolygon']           
        }
        url_vars = request.args.to_dict()
        if (not _geo_component(url_vars.get('project'))
                or not _geo_component(url_vars.get('layer_name'))):
            return make_response({'err_info': 'Требуются имя проекта и слой'}, 400, headers)
        path_src, path_out = path_gen(session['username'], 'geo', url_vars["project"], False)
        layer_name = url_vars["layer_name"]
        config_file_path = path_src + 'layers_description.json'
        try:
            layers_description = _load_geo_layers_description(config_file_path)
        except (OSError, ValueError):
            return make_response({'err_info': 'Проект не найден или повреждён'}, 400, headers)
        filename = f'{layer_name}_{url_vars["project"]}.csv'
        if layer_name not in layers_description:
            return make_response({'err_info': 'Incorrect layer name'}, 400, headers)
        if os.path.isfile(path_src + filename):
            try:
                json_full = geo.prepare_data_to_push(path_src, filename)
            except (OSError, ValueError, TypeError):
                return make_response({'err_info': 'Данные слоя повреждены'}, 400, headers)
            return make_response(json_full, 200, headers)
        else:
            return make_response({'data': {'info': f'Layer data not found'}}, 200, headers)
    else:
        return make_response({'err_info': ''}, 400, headers)


@app.route('/geo/post_data', methods = ['POST'])
@_validate_geo_request
def geo_post_data():
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if 'loggedin' in session:
        act_type = ['createDM', 'createSM', 'modify', 'delete']
        req_data = request.get_json(silent=True)
        username = session['username']
        if (request.is_json and len(request.get_data(cache=True)) < 50*1024
                and isinstance(req_data, dict)):
            action_type = req_data.get('action_type')
            layer_name = req_data.get('layer')
            if (not _geo_component(req_data.get('project'))
                    or not _geo_component(layer_name) or 'objData' not in req_data):
                return make_response({'err_info': 'Некорректные параметры запроса'}, 400, headers)
            filename = f'{layer_name}_{req_data["project"]}.csv'
            geo_stop_name = f'stops_{req_data["project"]}.csv'
            path_src, path_out = path_gen(session['username'], 'geo', req_data['project'], False)
            config_file_path = path_src + 'layers_description.json'
            try:
                layers_description = _load_geo_layers_description(config_file_path)
            except (OSError, ValueError):
                return make_response({'err_info': 'Проект не найден или повреждён'}, 400, headers)
            if layer_name not in layers_description:
                return make_response({'err_info': 'Слой не зарегистрирован в проекте'}, 400, headers)
            def layer_state(name):
                target = path_src + f'{name}_{req_data["project"]}.csv'
                if not os.path.isfile(target):
                    return None
                info = os.stat(target)
                return info.st_mtime_ns, info.st_size
            before = {name: layer_state(name) for name in layers_description}
            if action_type not in act_type:
                return make_response({'err_info': 'Error in sent data headers'}, 400, headers)
            check_modify, req_json = geo.geo_data_modify(req_data['project'], path_src, filename, geo_stop_name, 
                                                        layer_name, action_type, req_data['objData'],
                                                        output_path=path_out,
                                                        expected_geometry=layers_description[layer_name].get('geom'))
            if check_modify:
                changed_layers = []
                for name, description in layers_description.items():
                    after = layer_state(name)
                    description['load_data'] = 'Y' if after is not None else 'N'
                    if before[name] != after:
                        changed_layers.append(name)
                geo._write_geo_files([(layers_description, config_file_path, {})])
                if action_type == act_type[1]:
                    req_json['changed_layers'] = changed_layers
                    return make_response(req_json, 200, headers)
                else:
                    return make_response({'status': 'Data processed and stored',
                                          'changed_layers': changed_layers}, 200, headers)
            else:
                return make_response(req_json, 400, headers)
        else:
            return make_response({'err_info': 'Geo data format error'}, 400, headers)
    else:
        return make_response({'err_info': ''}, 400, headers)


class DelLayer(FlaskForm):
    submit = SubmitField(u"Удалить", id='load_grph_butt')

@app.route('/del_geo_layer/<proj_name>', methods=['GET', 'POST'])
@_validate_geo_request
def del_layer(proj_name):
    '''
    Удаляем слой в гео
    '''
    if 'loggedin' in session:
        log_username = session['username']
        req = request.args.to_dict()
        f = req['f_prefix']
        try:
            form = DelLayer()
            if form.validate_on_submit():
                # Загружаем граф
                path_src, path_out = path_gen(session['username'], req['proj_type'], proj_name, False)
                config_file_path = path_src + 'layers_description.json'
                layers_description = sl_json(config_file_path, None, 'load')
                if not _geo_component(f) or f not in layers_description:
                    abort(400)
                filename = f + '_' + proj_name + '.csv'
                geo.delete_geo_layer(proj_name, path_src, path_out, f)
                config_file_path = path_src + 'layers_description' + '.json'
                layers_description = sl_json(config_file_path, None, 'load')
                if f in layers_description:
                    layers_description.pop(f)
                    sl_json(config_file_path, layers_description, 'save')
                    msg = f'Слой {f} успешно удален'
                    flash(msg)
                else:
                    msg = f'Проблема удаления слоя {f}, обратитесь в техподдержку'
                    flash(msg)
                return render_template('remove_layer.html', form=form, title=proj_name)
            else:
                return render_template('remove_layer.html', title=proj_name, form=form, layer=f)
        except Exception as app_error:
            app.logger.info(f'{log_username} - delete geo layer error - {proj_name}:\n{app_error}')
            return redirect(url_for('internal_error'))
