import os
from datetime import date, datetime
from time import perf_counter

import pandas as pd
from shapely import wkt
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import (
    IntegerField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import InputRequired, Length, NumberRange, Optional

import core_demand
import models
from serv_funcs import check_disk_size, path_gen


demand_bp = Blueprint('demand', __name__, url_prefix='/demand')

DEMAND_FILES = [
    (
        'matrix_mobile',
        'Матрица перемещений абонентов сотовой связи',
        {'source', 'target', 'period', 'demand'},
    ),
    (
        'ref_tr_distr',
        'Справочник транспортных районов',
        {'id_tr_distr', 'wkt'},
    ),
    (
        'ref_stops',
        'Справочник остановочных пунктов',
        {'id_stop', 'wkt'},
    ),
    (
        'los',
        'Уровень обслуживания (характеристики межостановочных связей взвешенных по провозной способности)',
        {
            'source',
            'target',
            'all_vars',
            'los_speed',
            'los_carry',
        },
    ),
]

DEMAND_OPTIONAL_FILES = {
    'carry': (
        'Провозная способность '
        '(опционально, для корректировки временного профиля и маршрутных весов)',
        set(core_demand.HOUR_COLUMNS),
    ),
    'ref_links': (
        'Справочник связей и маршрутов ref_links '
        '(опционально, для маршрутной калибровки)',
        {'link', 'RNDs'},
    ),
    'route_boardings': (
        'Количество поездок на маршрутах '
        '(опционально, для маршрутной калибровки)',
        {'route', 'flow'},
    ),
}


class DemandProjectForm(FlaskForm):
    """Форма создания проекта восстановления спроса"""

    proj_name = StringField(
        validators=[InputRequired(), Length(min=1, max=9)]
    )
    proj_descr = StringField(validators=[Length(min=0, max=150)])
    submit = SubmitField('Продолжить')


class DemandUploadForm(FlaskForm):
    """Форма загрузки очередного исходного CSV"""

    optional_type = SelectField(
        'Тип опционального файла',
        choices=[
            (prefix, label)
            for prefix, (label, _) in DEMAND_OPTIONAL_FILES.items()
        ],
        validators=[Optional()],
    )
    file = FileField(
        validators=[FileRequired(), FileAllowed(['csv'], 'Требуется CSV-файл')]
    )
    submit = SubmitField('Загрузить')


class DemandCalcForm(FlaskForm):
    """Форма параметров восстановления спроса"""

    calculation_period = SelectField(
        'Период расчёта',
        choices=[('day', 'Сутки')] + [
            (str(hour), f'{hour}:00–{hour + 1}:00')
            for hour in range(5, 24)
        ],
        validators=[InputRequired()],
        default='day',
    )
    demand_profile_mode = SelectField(
        'Временной профиль спроса',
        choices=[
            ('mobile_only', 'Только мобильная матрица'),
            (
                'mobile_carry',
                'Мобильная матрица и провозная способность',
            ),
        ],
        validators=[InputRequired()],
        default='mobile_only',
        validate_choice=False,
    )
    pt_share = SelectField(
        'Доля поездок на общественном транспорте',
        choices=[
            (value / 100, f'{value / 100:.2f}')
            for value in range(10, 61, 5)
        ],
        coerce=float,
        validators=[InputRequired()],
        default=0.2,
    )
    pt_share_intrazonal = SelectField(
        'Доля поездок на ОТ внутри района',
        choices=[
            (value / 100, f'{value / 100:.2f}')
            for value in range(0, 31, 5)
        ],
        coerce=float,
        validators=[InputRequired()],
        default=0.05,
    )
    calibration_mode = SelectField(
        'Способ определения и калибровки спроса',
        choices=[
            ('pt_share', 'Доля поездок на общественном транспорте'),
            ('total_boardings', 'Общее количество поездок'),
            (
                'route_boardings',
                'Калибровка по фактическим поездкам на маршрутах',
            ),
        ],
        validators=[InputRequired()],
        default='pt_share',
        validate_choice=False,
    )
    total_boardings_period = IntegerField(
        'Количество поездок за расчетный период',
        validators=[Optional(), NumberRange(min=1)],
        render_kw={'min': 1, 'step': 1},
    )
    transfer_factor = SelectField(
        'Коэффициент пересадочности',
        choices=[
            (value / 100, f'{value / 100:.2f}')
            for value in range(100, 201, 5)
        ],
        coerce=float,
        validators=[InputRequired()],
        default=1.1,
    )
    access_radius_km = SelectField(
        'Радиус буфера пешей доступности остановки, км',
        choices=[
            (value / 10, f'{value / 10:.1f}')
            for value in range(3, 13)
        ],
        coerce=float,
        validators=[InputRequired()],
        default=0.7,
    )
    min_overlap_share = SelectField(
        'Минимальная доля пересечения буфера с районом',
        choices=[
            (value / 100, f'{value / 100:.2f}')
            for value in range(15, 31, 5)
        ],
        coerce=float,
        validators=[InputRequired()],
        default=0.15,
    )
    submit = SubmitField('Рассчитать', id='calc_butt')

    def validate(self, extra_validators=None):
        """Проверяет обязательные поля выбранного режима расчёта."""
        valid = super().validate(extra_validators)
        allowed_profiles = {
            value for value, _ in self.demand_profile_mode.choices
        }
        if self.demand_profile_mode.data not in allowed_profiles:
            self.demand_profile_mode.errors.append(
                'файл провозной способности не загружен.'
            )
            valid = False
        allowed_calibrations = {
            value for value, _ in self.calibration_mode.choices
        }
        if self.calibration_mode.data not in allowed_calibrations:
            self.calibration_mode.errors.append(
                'Для режима не загружены необходимые файлы.'
            )
            valid = False
        if (
            self.calibration_mode.data == 'total_boardings'
            and not self.total_boardings_period.data
        ):
            self.total_boardings_period.errors.append(
                'Укажите количество поездок за расчетный период.'
            )
            valid = False
        if (
            self.calculation_period.data != 'day'
            and self.demand_profile_mode.data != 'mobile_only'
        ):
            self.demand_profile_mode.errors.append(
                'Для одного часа используется только мобильная матрица.'
            )
            valid = False
        return valid


def _project_or_404(proj_name):
    """Возвращает demand-проект текущего пользователя или ответ 404"""
    project = models.demand_get_one_project(
        session['username'], proj_name
    )
    if project is None:
        abort(404)
    return project


def _validate_uploaded_csv(path, file_prefix, required_columns):
    """Проверяет структуру CSV и WKT-геометрию справочников"""
    try:
        data = pd.read_csv(path, sep=';')
    except Exception as error:
        return False, f'Не удалось прочитать CSV: {error}'

    columns = set(data.columns)
    missing = required_columns - columns
    if missing:
        return False, f'Отсутствуют столбцы: {", ".join(sorted(missing))}'

    if file_prefix == 'matrix_mobile':
        try:
            data['period'].map(core_demand._hour_name)
            demand_values = pd.to_numeric(data['demand'], errors='raise')
        except (TypeError, ValueError) as error:
            return False, f'Некорректные значения matrix_mobile: {error}'
        if demand_values.isna().any() or (demand_values < 0).any():
            return False, (
                'Спрос должен быть заполненным неотрицательным числом'
            )

    if file_prefix == 'los':
        los_values = data[['los_speed', 'los_carry']].apply(
            pd.to_numeric, errors='coerce'
        )
        if los_values.isna().any().any():
            return False, 'los_speed и los_carry должны быть числами'

    if file_prefix == 'ref_tr_distr' and 'pt_share_origin' in data:
        shares = pd.to_numeric(data['pt_share_origin'], errors='coerce')
        invalid = data['pt_share_origin'].notna() & shares.isna()
        if invalid.any() or not shares.dropna().between(0.1, 0.6).all():
            return False, 'pt_share_origin должен быть от 0.1 до 0.6'

    if file_prefix == 'carry':
        carry = data[list(core_demand.HOUR_COLUMNS)].apply(
            pd.to_numeric, errors='coerce'
        )
        if carry.isna().any().any() or (carry < 0).any().any():
            return False, (
                'Значения провозной способности должны быть '
                'неотрицательными числами'
            )

    if file_prefix == 'ref_links':
        if data['link'].duplicated().any():
            return False, 'В ref_links есть повторяющиеся link'
        try:
            data['RNDs'].map(core_demand._parse_link_rnds)
        except ValueError as error:
            return False, str(error)

    if file_prefix == 'route_boardings':
        if set(data.columns) != {'route', 'flow'}:
            return False, 'Файл должен содержать только route и flow'
        if data['route'].isna().any() or data['route'].astype(str).str.strip().eq(
            ''
        ).any():
            return False, 'В route_boardings есть пустые route'
        if data['route'].astype(str).str.strip().duplicated().any():
            return False, 'В route_boardings есть повторяющиеся route'
        values = pd.to_numeric(data['flow'], errors='coerce')
        invalid = data['flow'].notna() & values.isna()
        if invalid.any() or (values.dropna() < 0).any():
            return False, 'Поездки должны быть неотрицательными числами'
        if not values.notna().any():
            return False, 'Файл не содержит поездки'

    # WKT проверяется при загрузке, до запуска ресурсоёмкого расчёта
    geometry_types = {
        'ref_tr_distr': {'Polygon', 'MultiPolygon'},
        'ref_stops': {'Point', 'MultiPoint'},
    }
    if file_prefix in geometry_types:
        for row_number, value in data['wkt'].items():
            try:
                geometry = wkt.loads(str(value))
            except Exception:
                return False, f'Некорректный WKT в строке {row_number + 2}'
            if (
                geometry.is_empty
                or not geometry.is_valid
                or geometry.geom_type not in geometry_types[file_prefix]
            ):
                return False, f'Некорректная геометрия в строке {row_number + 2}'

    return True, 'Файл загружен'


@demand_bp.route('/', methods=['GET', 'POST'])
def projects():
    """Показывает проекты восстановления спроса и создаёт новый проект"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    username = session['username']
    rows = models.demand_get_projects(username)
    form = DemandProjectForm()

    if form.validate_on_submit():
        project_names = {row['proj_name'] for row in rows}
        if form.proj_name.data in project_names:
            flash('Проект с указанным кодом уже существует.')
        else:
            # Проверяем основные ограничения учётной записи перед созданием
            account = models.get_acc_by_username(username)
            limits = models.get_user_limits(account['tarif_name'])
            if account['expire_date'] <= date.today():
                flash('Необходимо продлить подписку.')
            elif check_disk_size(username) > limits['disk_limit']:
                flash('Достигнут лимит объёма хранения файлов.')
            else:
                today = date.today()
                models.demand_commit_new_project(
                    username,
                    form.proj_name.data,
                    form.proj_descr.data or '',
                    today.year,
                    today.month,
                    today.day,
                )
                path_gen(username, 'demand', form.proj_name.data, True)
                current_app.logger.info(
                    '%s - create demand project %s',
                    username,
                    form.proj_name.data,
                )
                return redirect(url_for(
                    'demand.upload', proj_name=form.proj_name.data
                ))

    return render_template(
        'demand_projects.html',
        title='Восстановление и прогнозирование спроса',
        headers=[
            'Код проекта',
            'Описание проекта',
            'Дата создания',
            'Статус проекта',
        ],
        rows=rows,
        form=form,
    )


@demand_bp.route('/<proj_name>/upload', methods=['GET', 'POST'])
def upload(proj_name):
    """Последовательно загружает исходные файлы demand-проекта"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    username = session['username']
    project = _project_or_404(proj_name)
    if project['status_prj'] == models.DEMAND_STATUS_COMPLETED:
        return redirect(url_for('demand.result', proj_name=proj_name))
    required_loaded = project['status_src'] >= len(DEMAND_FILES)
    if required_loaded:
        file_prefix = request.form.get('optional_type', 'carry')
        if file_prefix not in DEMAND_OPTIONAL_FILES:
            abort(400)
        file_label, required_columns = DEMAND_OPTIONAL_FILES[file_prefix]
    else:
        file_prefix, file_label, required_columns = DEMAND_FILES[
            project['status_src']
        ]
    form = DemandUploadForm()

    if form.validate_on_submit():
        # Загруженный файл сохраняется под системным именем проекта.
        original_name = secure_filename(form.file.data.filename)
        if not original_name.lower().endswith('.csv'):
            flash('Требуется файл в формате CSV.')
        else:
            path_src, _ = path_gen(username, 'demand', proj_name, False)
            filename = f'{file_prefix}_{proj_name}.csv'
            destination = path_src + filename
            form.file.data.save(destination)
            valid, message = _validate_uploaded_csv(
                destination, file_prefix, required_columns
            )
            if not valid:
                os.remove(destination)
                flash(message)
            else:
                if not required_loaded:
                    # После обязательных файлов показываем опциональный carry.
                    status_src = project['status_src'] + 1
                    models.demand_commit_update_status_src(
                        username, proj_name, status_src
                    )
                    if status_src == len(DEMAND_FILES):
                        models.demand_commit_update_status_prj(
                            username, proj_name, 1
                        )
                        current_app.logger.info(
                            '%s - demand project %s source files loaded',
                            username,
                            proj_name,
                        )
                flash(message)
                return redirect(url_for(
                    'demand.upload', proj_name=proj_name
                ))

    return render_template(
        'demand_upload.html',
        title=proj_name,
        form=form,
        files=DEMAND_FILES,
        loaded_count=project['status_src'],
        current_file=file_label,
        required_loaded=required_loaded,
        optional_files=DEMAND_OPTIONAL_FILES,
        optional_loaded={
            prefix: os.path.isfile(
                path_gen(username, 'demand', proj_name, False)[0]
                + f'{prefix}_{proj_name}.csv'
            )
            for prefix in DEMAND_OPTIONAL_FILES
        },
    )


@demand_bp.route('/<proj_name>/calculate', methods=['GET', 'POST'])
def calculate(proj_name):
    """Принимает параметры и запускает восстановление спроса."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    username = session['username']
    project = _project_or_404(proj_name)
    if project['status_src'] < len(DEMAND_FILES):
        return redirect(url_for('demand.upload', proj_name=proj_name))
    if project['status_prj'] == models.DEMAND_STATUS_COMPLETED:
        return redirect(url_for('demand.result', proj_name=proj_name))
    if project['status_prj'] == models.DEMAND_STATUS_CALCULATING:
        return redirect(url_for('demand.projects'))

    path_src, _ = path_gen(username, 'demand', proj_name, False)
    carry_path = path_src + f'carry_{proj_name}.csv'
    ref_links_path = path_src + f'ref_links_{proj_name}.csv'
    route_boardings_path = path_src + f'route_boardings_{proj_name}.csv'
    carry_available = os.path.isfile(carry_path)
    route_files_available = (
        os.path.isfile(ref_links_path)
        and os.path.isfile(route_boardings_path)
    )
    form = DemandCalcForm()
    if not carry_available:
        # Без файла предложения пользователю доступен только базовый профиль.
        form.demand_profile_mode.choices = [
            ('mobile_only', 'Только мобильная матрица')
        ]
    if form.validate_on_submit():
        if models.check_calculate_process(username):
            flash('Дождитесь завершения расчёта другого проекта.')
        else:
            calibration_mode = form.calibration_mode.data
            pt_share = float(form.pt_share.data)
            pt_share_intrazonal = float(form.pt_share_intrazonal.data)
            calculation_period = form.calculation_period.data
            demand_profile_mode = form.demand_profile_mode.data
            total_boardings_period = (
                int(form.total_boardings_period.data)
                if (
                    calibration_mode in {
                        'total_boardings', 'route_boardings'
                    }
                    and form.total_boardings_period.data is not None
                )
                else None
            )
            transfer_factor = (
                float(form.transfer_factor.data)
                if calibration_mode in {
                    'total_boardings', 'route_boardings'
                }
                else None
            )
            access_radius_km = float(form.access_radius_km.data)
            min_overlap_share = float(form.min_overlap_share.data)
            # Нормирование временно отключено, в том числе для старых POST-форм.
            normalize_target_volume = False
            calculation_started = False
            calc_started_at = datetime.now()
            calc_started = perf_counter()
            calc_status = 'failed'
            current_app.logger.info(
                '%s - demand project %s calc started at %s',
                username, proj_name,
                calc_started_at.isoformat(sep=' ', timespec='seconds'),
            )
            try:
                # Провозная способность могла быть загружена на прошлом шаге.
                # Проверяем весь комплект до изменения статуса и тяжёлого расчёта.
                active_carry_path = (
                    carry_path
                    if os.path.isfile(carry_path)
                    else None
                )
                core_demand.validate_demand_files(
                    demand_path=(
                        path_src + f'matrix_mobile_{proj_name}.csv'
                    ),
                    districts_path=(
                        path_src + f'ref_tr_distr_{proj_name}.csv'
                    ),
                    stops_path=path_src + f'ref_stops_{proj_name}.csv',
                    los_path=path_src + f'los_{proj_name}.csv',
                    calculation_period=calculation_period,
                    carry_path=active_carry_path,
                    demand_profile_mode=demand_profile_mode,
                    calibration_mode=calibration_mode,
                    ref_links_path=(
                        ref_links_path if route_files_available else None
                    ),
                    route_boardings_path=(
                        route_boardings_path if route_files_available else None
                    ),
                )
                current_app.logger.info(
                    '%s - demand project %s validate_demand_files pass',
                    username, proj_name,
                )
                # Статус запуска сохраняется до выполнения тяжёлого расчёта.
                models.mark_calculation_started(
                    username, 'demand', proj_name
                )
                calculation_started = True
                current_app.logger.info(
                    '%s - demand project %s mark_calculation_started pass',
                    username, proj_name,
                )
                summary = core_demand.calculate_demand(
                    proj_name,
                    pt_share=pt_share,
                    pt_share_intrazonal=pt_share_intrazonal,
                    calculation_period=calculation_period,
                    demand_profile_mode=demand_profile_mode,
                    calibration_mode=calibration_mode,
                    total_boardings_period=total_boardings_period,
                    transfer_factor=transfer_factor,
                    access_radius_km=access_radius_km,
                    min_overlap_share=min_overlap_share,
                    normalize_target_volume=normalize_target_volume,
                )
                models.demand_commit_update_status_prj(
                    username=username,
                    proj_name=proj_name,
                    status_prj=models.DEMAND_STATUS_COMPLETED,
                    pt_share=pt_share,
                    pt_share_intrazonal=pt_share_intrazonal,
                    calculation_period=calculation_period,
                    demand_profile_mode=demand_profile_mode,
                    access_radius_km=access_radius_km,
                    min_overlap_share=min_overlap_share,
                    normalize_target_volume=normalize_target_volume,
                    calibration_mode=calibration_mode,
                    total_boardings_period=summary[
                        'total_boardings_period'
                    ],
                    transfer_factor=transfer_factor,
                    base_period_trip_volume=summary[
                        'base_period_trip_volume'
                    ],
                    target_period_trip_volume=summary[
                        'target_period_trip_volume'
                    ],
                    effective_pt_share=summary['effective_pt_share'],
                    effective_pt_share_status=summary[
                        'effective_pt_share_status'
                    ],
                )
                current_app.logger.info(
                    '%s - demand project %s persist_calculation_results pass',
                    username, proj_name,
                )
                calc_status = 'completed'
                return redirect(url_for(
                    'demand.result', proj_name=proj_name
                ))
            except ValueError as error:
                current_app.logger.warning(
                    '%s - demand project %s validation error: %s',
                    username,
                    proj_name,
                    error,
                )
                if calculation_started:
                    try:
                        models.demand_commit_update_status_prj(
                            username, proj_name, models.DEMAND_STATUS_FAILED
                        )
                    except Exception:
                        current_app.logger.exception(
                            '%s - demand project %s failed to persist calculation error status',
                            username, proj_name,
                        )
                flash(str(error))
            except Exception:
                current_app.logger.exception(
                    '%s - demand project %s endpoint calculate error',
                    username,
                    proj_name,
                )
                if calculation_started:
                    try:
                        models.demand_commit_update_status_prj(
                            username, proj_name, models.DEMAND_STATUS_FAILED
                        )
                    except Exception:
                        current_app.logger.exception(
                            '%s - demand project %s failed to persist calculation error status',
                            username, proj_name,
                        )
                flash('При расчёте произошла ошибка. Расчёт можно повторить.')
            finally:
                current_app.logger.info(
                    '%s - demand project %s calc finished at %s; '
                    'status=%s; duration_seconds=%.3f',
                    username, proj_name,
                    datetime.now().isoformat(sep=' ', timespec='seconds'),
                    calc_status, perf_counter() - calc_started,
                )

    return render_template(
        'demand_calculate.html',
        title=proj_name,
        form=form,
    )


@demand_bp.route('/<proj_name>/result')
def result(proj_name):
    """Показывает краткий отчёт и ссылку на итоговую матрицу."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    username = session['username']
    project = _project_or_404(proj_name)
    if project['status_prj'] != models.DEMAND_STATUS_COMPLETED:
        return redirect(url_for('demand.calculate', proj_name=proj_name))

    # Для отчёта читается только компактный файл контрольных показателей.
    _, path_out = path_gen(username, 'demand', proj_name, False)
    stats_path = path_out + f'demand_stats_{proj_name}.csv'
    stats = pd.read_csv(stats_path, sep=';').iloc[0].to_dict()
    stats.setdefault('control_errors', 0)
    stats['served_share'] = (
        100 * stats['served_pt_demand'] / stats['requested_pt_demand']
        if stats['requested_pt_demand'] else 0
    )
    # Базовая оценка появляется у проектов, пересчитанных после ее добавления.
    baseline_stats = None
    baseline_stats_path = path_out + f'demand_stats_{proj_name}_bl.csv'
    if os.path.isfile(baseline_stats_path):
        baseline_stats = pd.read_csv(baseline_stats_path, sep=';').iloc[0].to_dict()
        baseline_stats['served_share'] = (
            100 * baseline_stats['served_pt_demand'] / baseline_stats['requested_pt_demand']
            if baseline_stats['requested_pt_demand'] else 0
        )
    baseline_matrix_available = os.path.isfile(path_out + f'demand_matrix_{proj_name}_bl.csv')
    baseline_diagnostic_available = os.path.isfile(path_out + f'baseline_{proj_name}.csv')
    control_available = os.path.isfile(
        path_out + f'demand_control_{proj_name}.csv'
    )
    supply_available = os.path.isfile(
        path_out + f'demand_supply_{proj_name}.csv'
    )
    route_stops_matrix_available = os.path.isfile(
        path_out + f'route_stops_matrix_{proj_name}.csv'
    )
    route_calibration_available = os.path.isfile(
        path_out + f'route_calibration_{proj_name}.csv'
    )
    demand_diagnostic_prefix = (
        'excluded_demand' if project['normalize_target_volume']
        else 'unserved_demand'
    )
    demand_diagnostic_available = os.path.isfile(
        path_out + f'{demand_diagnostic_prefix}_{proj_name}.csv'
    )
    # Ранее исключенный спрос сохранялся под именем unserved_demand.
    if project['normalize_target_volume'] and not demand_diagnostic_available:
        demand_diagnostic_prefix = 'unserved_demand'
        demand_diagnostic_available = os.path.isfile(
            path_out + f'{demand_diagnostic_prefix}_{proj_name}.csv'
        )

    return render_template(
        'demand_info.html',
        title=proj_name,
        project=project,
        stats=stats,
        baseline_stats=baseline_stats,
        baseline_matrix_available=baseline_matrix_available,
        baseline_diagnostic_available=baseline_diagnostic_available,
        control_available=control_available,
        supply_available=supply_available,
        route_stops_matrix_available=route_stops_matrix_available,
        route_calibration_available=route_calibration_available,
        demand_diagnostic_available=demand_diagnostic_available,
        demand_diagnostic_prefix=demand_diagnostic_prefix,
    )


@demand_bp.route('/<proj_name>/baseline/<file_kind>')
def baseline_download(proj_name, file_kind):
    """Выдает матрицу или диагностику базовой оценки текущего пользователя."""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    project = _project_or_404(proj_name)
    if project['status_prj'] != models.DEMAND_STATUS_COMPLETED:
        abort(409)

    # Фиксированные имена сохраняют суффикс _bl без подмены имени проекта.
    filenames = {
        'matrix': f'demand_matrix_{proj_name}_bl.csv',
        'diagnostic': f'baseline_{proj_name}.csv',
    }
    if file_kind not in filenames:
        abort(404)
    _, path_out = path_gen(session['username'], 'demand', proj_name, False)
    return send_from_directory(path_out, filenames[file_kind], as_attachment=True)
