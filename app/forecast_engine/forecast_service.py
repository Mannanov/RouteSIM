"""Чтение CSV, хранение проектов и сохранение отчетов отдельных запусков прогноза."""

import json
import logging
import re
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import core_forecast as core


SOURCE_ROOT = Path(__file__).resolve().parents[1]
# Версия относится к математической методике и не меняется при переименовании файлов.
METHOD_VERSION = '2026-10-22-baseline-2-new-connection-modes-v21'
TARGET_RECONCILIATION_VERSIONS = ('2026-10-20-baseline-2-feasible-targets-v5',
                                  '2026-10-20-baseline-2-generation-validation-v6',
                                  '2026-10-20-baseline-2-spatial-validation-v7',
                                  '2026-10-20-baseline-2-los-speed-floor-v8',
                                  '2026-10-20-baseline-2-los-preprocess-4p5-v9',
                                  '2026-10-20-baseline-2-generation-regularization-v10',
                                  '2026-10-20-baseline-2-reconciliation-diagnostics-v11',
                                  '2026-10-20-baseline-2-los-continuity-v12',
                                  '2026-10-20-baseline-2-correction-diagnostics-v13',
                                  '2026-10-20-baseline-2-transport-speed-v14',
                                  '2026-10-21-baseline-2-spatial-access-global-los-v15',
                                  '2026-10-21-baseline-2-spatial-observed-los-v16',
                                  '2026-10-21-baseline-2-analogue-los-transfer-v17',
                                  '2026-10-21-baseline-2-ipf-tolerance-1e4-v18',
                                  '2026-10-21-baseline-2-generation-without-speed-v19',
                                  '2026-10-22-baseline-2-fitted-los-curve-v20', METHOD_VERSION)
FILE_TYPES = core.FILE_TYPES
DEFAULT_CONFIG = core.DEFAULT_CONFIG
HOUR_COLUMNS = core.HOUR_COLUMNS
STATUS_LABELS = {0: 'Загрузка данных', 1: 'Готов к расчету', 2: 'Выполняется расчет',
                 3: 'Расчет завершен', 4: 'Ошибка расчета'}
# Реестр одновременно задает подписи отчетов и разрешенные имена скачиваемых файлов.
ARTIFACT_LABELS = {
    'T_final': 'Итоговая матрица выбранных часов', 'M_forecast': 'Модель прогнозного состояния',
    'M_base': 'Модель базового состояния', 'B': 'Наблюдаемая опора',
    'generation': 'Почасовая и суточная генерация', 'coefficients': 'Коэффициенты и масштабы',
    'statuses': 'Статусы состояний и сборки', 'margins': 'Итоги районов и внутрирайонные доли',
    'corrections': 'Ограниченная поправка', 'training_targets': 'Обучающие районные итоги',
    'correction_diagnostics': 'Вклад поправки C в объем и прирост по часам',
    'correction_districts': 'Поправка C по районам и условия компенсации',
    'transport_speed': 'Транспортный признак: скорости, соседи и причины отключения',
    'los': 'LOS и транспортная маска', 'los_donors': 'Доноры модели скорости',
    'los_preprocessing': 'Предобработка LOS: исходные и скорректированные времена',
    'calibration': 'Калибровка beta', 'calibration_trials': 'Пробы и перепроверка beta',
    'fold_assignments': 'Распределение районов по фолдам',
    'test_predictions': 'Внеобучающие прогнозы OD', 'metrics': 'Ошибки и покрытие OD',
    'district_predictions': 'Расширение территории: районные итоги и внутрирайонные доли',
    'district_metrics': 'Расширение территории: ошибки генерации, целей и сальдо',
    'generation_predictions': 'Полные районные итоги: внеобучающие прогнозы генерации',
    'generation_metrics': 'Полные районные итоги: ошибки и покрытие генерации',
    'generation_fold_statuses': 'Полные районные итоги: статусы обучения генерации',
    'generation_diagnostics': 'Устойчивость генерации и выбранный штраф',
    'generation_alpha_trials': 'Внутренние фолды подбора штрафа генерации',
    'generation_fit_diagnostics': 'Полные районные итоги: устойчивость генерации и штраф',
    'generation_fit_alpha_trials': 'Полные районные итоги: внутренние фолды подбора штрафа',
    'training_coverage': 'Усечение обучающих итогов R→R по районам',
    'training_coverage_summary': 'Усечение обучающих итогов R→R по фолдам и часам',
    'benchmark_predictions': 'Прогнозы эталона ближайшего аналога',
    'benchmark_donors': 'Доноры, масштабы и наличие аналогов эталона',
    'benchmark_comparison': 'Сравнение модели с эталоном по группам и наличию аналогов',
    'fold_statuses': 'Готовность фолдов',
    'validation_los': 'Проверка LOS', 'validation_los_metrics': 'Ошибки и покрытие LOS',
    'failure': 'Диагностика отказа',
    'parameters': 'Параметры и версия методики', 'los_models': 'Модели LOS по контекстам',
    'analogues': 'Аналоги новых районов', 'failures': 'Отказы модельных состояний',
    'aggregation': 'Контроль агрегации',
    'stop_access': 'Доступность остановок, пересечения буферов и веса районов',
    'unmapped': 'Исключенные при привязке корреспонденции',
    'daily_controls': 'Полнота и баланс суточных результатов',
    'target_reconciliation': 'Согласование модельных целей с доступными связями',
    'reconciliation_delta': 'Влияние сокращения целей на модельное изменение поездок',
    'reconciliation_delta_districts': 'Разложение изменения целей по районам и направлениям',
    'context_timings': 'Время этапов расчета по контекстам',
    'ipf_support': 'Вынужденные нули опоры IPF',
    'T_final_day': 'Полная суточная итоговая матрица',
    'M_forecast_day': 'Полная суточная модель прогноза', 'M_base_day': 'Полная суточная базовая модель',
}
ARTIFACT_LABELS.update({f'spatial_{name}': 'Пространственные блоки: ' + ARTIFACT_LABELS[name] for name in (
    'fold_assignments', 'test_predictions', 'metrics', 'district_predictions', 'district_metrics', 'fold_statuses',
    'validation_los', 'validation_los_metrics', 'generation_predictions', 'generation_metrics',
    'generation_fold_statuses', 'training_coverage', 'training_coverage_summary',
    'generation_fit_diagnostics', 'generation_fit_alpha_trials',
    'benchmark_predictions', 'benchmark_donors', 'benchmark_comparison')})
MATRIX_ARTIFACTS = ('T_final', 'M_forecast', 'M_base', 'B', 'T_final_day', 'M_forecast_day', 'M_base_day')
PRIMARY_ARTIFACTS = ('statuses', 'target_reconciliation', 'correction_diagnostics',
                     'spatial_benchmark_comparison')
# Включать только для отладки: полные OD-прогнозы проверок, LOS и промежуточные матрицы.
SAVE_DIAGNOSTIC_REPORTS = False
LARGE_DIAGNOSTIC_ARTIFACTS = frozenset((
    'test_predictions', 'benchmark_predictions', 'validation_los',
    'spatial_test_predictions', 'spatial_benchmark_predictions', 'spatial_validation_los',
    'los', 'los_donors', 'los_preprocessing', 'corrections', 'ipf_support',
    'M_forecast', 'M_base', 'B', 'M_forecast_day', 'M_base_day',
))


class _ProjectLogger(logging.LoggerAdapter):
    """Добавляет контекст запуска в сообщения существующего журнала приложения."""

    def log(self, level, msg, *args, **kwargs):
        super().log(level, '%s ' + msg, self.extra['prefix'], *args, **kwargs)


def validate_config(config=None):
    """Normalize historical settings before validating a new eight-feature run."""
    supplied = dict(config or {})
    supplied.pop('use_transport_speed', None)
    return core.validate_config(supplied)


def read_csv(path):
    """Читает CSV с разделителем «;», сохраняя строковые ID и пустые значения как NaN."""
    try:
        return pd.read_csv(path, sep=';', keep_default_na=False, na_values=[''], dtype={
            key: 'string' for key in ('source', 'target', 'id_stop', 'id_tr_distr', 'agg_source', 'agg_target',
                                     'district_id', 'witness_source', 'witness_target')
        })
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError(f'Не удалось прочитать CSV: {error}') from error


def init_project_storage():
    """Создает таблицу проектов, если хранилище еще не инициализировано."""
    # Прежние имена таблицы и каталога сохраняют доступ к проектам после переименования модуля.
    import models
    conn, cur = models.get_db_cur()
    try:
        cur.execute('''CREATE TABLE IF NOT EXISTS forecast_bl_2_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL, proj_name TEXT NOT NULL, proj_descr TEXT NOT NULL,
            date_crt TEXT NOT NULL, status_src INTEGER NOT NULL DEFAULT 0,
            status_prj INTEGER NOT NULL DEFAULT 0, config_json TEXT,
            run_id TEXT, outcome TEXT, error_message TEXT, calculation_started_at TEXT,
            UNIQUE(username, proj_name)
        )''')
        conn.commit()
    finally:
        conn.close()


def project_paths(username, proj_name, create=False):
    """Возвращает каталоги входов и результатов проекта; при необходимости создает их."""
    # Идентификаторы не должны превращать путь проекта в выход за каталог пользователя.
    for value in (username, proj_name):
        if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\:\0'):
            raise ValueError('Некорректный идентификатор каталога проекта')
    root = (SOURCE_ROOT / 'static' / 'userdata').resolve()
    project = (root / username / 'forecast_bl_2' / proj_name).resolve()
    if not project.is_relative_to(root):
        raise ValueError('Каталог проекта выходит за пределы пользовательских данных')
    paths = project / 'input', project / 'output'
    if create:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def get_project(username, proj_name):
    """Возвращает проект владельца после восстановления зависших расчетов либо None."""
    import models
    models.recover_stale_calculations(username)
    conn, cur = models.get_db_cur()
    try:
        row = cur.execute('SELECT * FROM forecast_bl_2_projects WHERE username=? AND proj_name=?',
                          (username, proj_name)).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def list_projects(username):
    """Возвращает проекты пользователя от новых к старым с актуализированными статусами."""
    import models
    models.recover_stale_calculations(username)
    conn, cur = models.get_db_cur()
    try:
        return [dict(row) for row in cur.execute(
            'SELECT * FROM forecast_bl_2_projects WHERE username=? ORDER BY id DESC', (username,)
        ).fetchall()]
    finally:
        conn.close()


def delete_project(username, proj_name):
    """Удаляет файлы и запись проекта владельца, если расчет не выполняется."""
    import models
    source, _ = project_paths(username, proj_name)
    directory = source.parent
    expected = (SOURCE_ROOT / 'static' / 'userdata' / username / 'forecast_bl_2' / proj_name).absolute()
    if directory != expected:
        raise ValueError('Каталог проекта перенаправлен; удаление запрещено')
    conn, cur = models.get_db_cur()
    try:
        cur.execute('BEGIN IMMEDIATE')
        project = cur.execute('''SELECT status_prj FROM forecast_bl_2_projects
                                 WHERE username=? AND proj_name=?''', (username, proj_name)).fetchone()
        if project is None:
            raise ValueError('Проект не найден')
        if project['status_prj'] == 2:
            raise ValueError('Дождитесь завершения расчета перед удалением проекта')
        if directory.exists():
            shutil.rmtree(directory)
        cur.execute('DELETE FROM forecast_bl_2_projects WHERE username=? AND proj_name=?',
                    (username, proj_name))
        conn.commit()
    finally:
        conn.close()


def create_project(username, proj_name, description):
    """Проверяет код и описание, создает запись проекта и его входной/выходной каталоги."""
    import models
    if not re.fullmatch(r'[\w-]{1,9}', proj_name) or len(description) > 150:
        raise ValueError('Код проекта: 1–9 букв, цифр, _ или -; описание: до 150 символов')
    conn, cur = models.get_db_cur()
    try:
        cur.execute('''INSERT INTO forecast_bl_2_projects (username, proj_name, proj_descr, date_crt)
                       VALUES (?, ?, ?, ?)''', (username, proj_name, description, date.today().isoformat()))
        project_paths(username, proj_name, create=True)
        conn.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError('Проект с указанным кодом уже существует') from error
    finally:
        conn.close()


def save_upload(username, proj_name, file_type, stream):
    """Проверяет загружаемый CSV и сохраняет его на ожидаемом этапе проекта."""
    import models
    coordinate_update = file_type == 'ref_stops'
    if file_type not in FILE_TYPES:
        raise ValueError('Неизвестный тип файла')
    source, _ = project_paths(username, proj_name)
    temporary = None
    try:
        # Непроверенный файл хранится отдельно и не заменяет входные данные проекта.
        with tempfile.NamedTemporaryFile(dir=source, suffix='.tmp', delete=False) as uploaded:
            temporary = Path(uploaded.name)
            shutil.copyfileobj(stream, uploaded)
        core.validate_upload(read_csv(temporary), file_type)
        conn, cur = models.get_db_cur()
        try:
            # Блокировка защищает проверку этапа от одновременной загрузки или запуска.
            cur.execute('BEGIN IMMEDIATE')
            project = cur.execute('SELECT * FROM forecast_bl_2_projects WHERE username=? AND proj_name=?',
                                  (username, proj_name)).fetchone()
            coordinate_update = coordinate_update and project is not None and project['status_src'] == 4
            stage = 4 if coordinate_update else FILE_TYPES.index(file_type)
            if (project is None or project['status_src'] != stage
                    or (project['status_prj'] == 2 if coordinate_update else project['status_prj'] != 0)):
                raise ValueError('Этап загрузки изменился; обновите страницу')
            temporary.replace(source / f'{file_type}.csv')
            if not coordinate_update:
                cur.execute('''UPDATE forecast_bl_2_projects SET status_src=?, status_prj=?
                               WHERE username=? AND proj_name=?''',
                            (stage + 1, 1 if stage == 3 else 0, username, proj_name))
            conn.commit()
        finally:
            conn.close()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def start_project(username, proj_name, config):
    """Фиксирует новый запуск и параметры до предпроверки CSV, блокируя параллельный расчет."""
    import models
    config = validate_config(config)
    models.recover_stale_calculations(username)
    conn, cur = models.get_db_cur()
    run_id = uuid.uuid4().hex
    try:
        cur.execute('BEGIN IMMEDIATE')
        # Проверка занятости пользователя и смена run_id выполняются в одной транзакции.
        if models._calculation_running(cur, username):
            raise ValueError('Дождитесь завершения расчета другого проекта')
        cur.execute('''UPDATE forecast_bl_2_projects SET status_prj=2, config_json=?, run_id=?,
                       outcome=NULL, error_message=NULL, calculation_started_at=?
                       WHERE username=? AND proj_name=? AND status_src=4 AND status_prj!=2''',
                    (json.dumps(config, allow_nan=False), run_id, datetime.now(timezone.utc).isoformat(), username, proj_name))
        if cur.rowcount != 1:
            raise ValueError('Проект не готов к расчету')
        conn.commit()
    finally:
        conn.close()
    return run_id


def finish_project(username, proj_name, run_id, outcome, message=''):
    """Завершает только текущий выполняемый запуск, не позволяя старому изменить новый."""
    import models
    conn, cur = models.get_db_cur()
    try:
        cur.execute('''UPDATE forecast_bl_2_projects SET status_prj=?, outcome=?, error_message=?,
                       calculation_started_at=NULL WHERE username=? AND proj_name=? AND run_id=? AND status_prj=2''',
                    (4 if outcome == 'FAILED' else 3, outcome, message, username, proj_name, run_id))
        if cur.rowcount != 1:
            raise ValueError('Запуск больше не является текущим')
        conn.commit()
    finally:
        conn.close()


def _run_directory(username, proj_name, run_id):
    """Проверяет формат run_id и возвращает каталог отчетов этого запуска."""
    if not isinstance(run_id, str) or not re.fullmatch('[0-9a-f]{32}', run_id):
        raise ValueError('Нет сохраненного запуска')
    return project_paths(username, proj_name)[1] / 'runs' / run_id


def _json_value(value):
    """Приводит значения NumPy к обычным типам, а неконечные числа заменяет на None."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_csv(directory, name, table, *, append=False):
    """Записывает или дополняет CSV; вложенные списки и словари сохраняет как JSON."""
    frame = table.copy()
    # Явная сериализация вложенной диагностики исключает зависимость от представления Python.
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda value: json.dumps(_json_value(value), ensure_ascii=False)
                                              if isinstance(value, (dict, list)) else value)
    path = directory / f'{name}.csv'
    has_header = append and path.exists()
    frame.to_csv(path, sep=';', index=False, encoding='utf-8-sig',
                 mode='a' if has_header else 'w', header=not has_header)


def _write_reports(directory, tables, summary, *, matrix_chunks=(), existing_artifacts=None):
    """Сохраняет отчеты и порции матриц, затем атомарно заменяет сводный manifest.json."""
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {name: path for name, path in (existing_artifacts or {}).items()
                 if SAVE_DIAGNOSTIC_REPORTS or name not in LARGE_DIAGNOSTIC_ARTIFACTS}
    for name, table in tables.items():
        if not SAVE_DIAGNOSTIC_REPORTS and name in LARGE_DIAGNOSTIC_ARTIFACTS:
            continue
        if not len(table.columns):
            continue
        _write_csv(directory, name, table)
        artifacts[name] = f'{name}.csv'
    # Матрицы дописываются порциями, а уже сохраненные отчеты фолдов остаются в реестре.
    for name, table in matrix_chunks:
        if not SAVE_DIAGNOSTIC_REPORTS and name in LARGE_DIAGNOSTIC_ARTIFACTS:
            continue
        _write_csv(directory, name, table.assign(context='main'), append=True)
        artifacts[name] = f'{name}.csv'
    summary['artifacts'] = artifacts
    # Итоговый реестр публикуется лишь после записи перечисленных в нем файлов.
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(_json_value(summary), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(directory / 'manifest.json')
    return summary


def calculate_project(username, proj_name, run_id, config):
    """Читает входы проекта без копирования и сохраняет отчеты под переданным run_id.

    Диагностика фолдов записывается по мере готовности; полные матрицы основного
    контекста экспортируются порциями после завершения расчета.
    """
    # Дочерний журнал использует обработчики app.logger из rsim без импорта Flask.
    log = _ProjectLogger(logging.getLogger('rsim.forecast'), {
        'prefix': f'{username} - forecast project {proj_name} run_id={run_id}',
    })
    config = validate_config(config)
    directory = _run_directory(username, proj_name, run_id)
    directory.mkdir(parents=True)
    source, _ = project_paths(username, proj_name)
    # start_project блокирует загрузки до завершения расчета; копия входов не нужна.
    log.info('prepare_data started')
    prepared = core.prepare_data(*(read_csv(source / f'{name}.csv') for name in FILE_TYPES),
                                 config=config)
    log.info('prepare_data pass; existing_districts=%s; forecast_districts=%s',
             prepared['n_existing'], len(prepared['ids']))
    _write_csv(directory, 'aggregation', prepared['aggregation']['control'])
    _write_csv(directory, 'unmapped', prepared['aggregation']['excluded'])
    if 'stop_access' in prepared['aggregation']:
        _write_csv(directory, 'stop_access', prepared['aggregation']['stop_access'])
    context_metadata = {}
    context_artifacts = {'aggregation': 'aggregation.csv', 'unmapped': 'unmapped.csv'}
    if 'stop_access' in prepared['aggregation']:
        context_artifacts['stop_access'] = 'stop_access.csv'

    def save_context(label, context):
        """Сохраняет диагностику контекста и краткие метаданные без полных OD-таблиц."""
        # Полные OD-таблицы фолда не строим; ядро сможет освободить матрицы после вызова.
        for name, table in core.context_tables(context, prepared, include_matrices=False).items():
            if not SAVE_DIAGNOSTIC_REPORTS and name in LARGE_DIAGNOSTIC_ARTIFACTS:
                continue
            if len(table.columns):
                _write_csv(directory, name, table.assign(context='main' if label == 'main' else f'fold_{label}'), append=True)
                context_artifacts[name] = f'{name}.csv'
        context_metadata[label] = {
            'training_ids': [context['ids'][i] for i in context['training']],
            'zero_observed_day': context['zero_observed_day'],
            'timings': context.get('timings', {}),
            'generation_diagnostics': context['generation_model']['diagnostics'] if context['generation_model'] else {},
            'los_model': context['los']['model'] if context['los'] is not None else {},
            'analogues': context['los']['analogues'] if context['los'] is not None else {},
            'los_errors': context['los']['errors'] if context['los'] is not None else [],
            'error': context.get('error'),
            'target_reconciliation': {state: core._reconciliation_summary(gen.get('reconciliation', []))
                                      for state, gen in context['generation'].items()
                                      if gen.get('reconciliation')},
            'failed_states': [{'hour': hour, 'state': state, **{key: value for key, value in item[state].items()
                              if key not in ('matrix', 'diagnostic_matrix', 'boundary')}}
                             for hour, item in context['hours'].items() for state in ('base', 'forecast')
                             if item[state]['status'] not in ('OK', 'NOT_REQUIRED')],
        }

    result = core.forecast_baseline(prepared, context_sink=save_context, log=log)
    log.info('main context reports started')
    save_context('main', result['main'])
    tables = dict(result['validation']['tables'])
    # Дополняем проверочные отчеты параметрами, версиями и причинами отказов состояний.
    tables.update(
        parameters=pd.DataFrame([{'parameter': key, 'value': value} for key, value in {
            **result['config'], 'method_version': METHOD_VERSION, 'metric_crs': result['metric_crs'],
        }.items()]),
        los_models=pd.DataFrame([{'context': label, **meta['los_model'], 'zero_observed_day': meta['zero_observed_day']}
                                 for label, meta in context_metadata.items()]),
        analogues=pd.DataFrame([{'context': label, 'district_id': key, 'analogues': values}
                                for label, meta in context_metadata.items() for key, values in meta['analogues'].items()],
                               columns=['context', 'district_id', 'analogues']),
        failures=pd.DataFrame([{'context': label, **failure} for label, meta in context_metadata.items()
                                for failure in meta['failed_states']], columns=[
                                    'context', 'hour', 'state', 'status', 'iterations', 'residual', 'outgoing', 'incoming', 'reason', 'target_status', 'support_status']),
    )
    summary = {'run_id': run_id, 'method_version': METHOD_VERSION, 'outcome': result['status'],
               'config': result['config'], 'metric_crs': result['metric_crs'],
               'stop_assignment': prepared['stop_assignment'],
               'validation': {key: value for key, value in result['validation'].items() if key not in ('contexts', 'tables')},
               'contexts': context_metadata, 'created_at': datetime.now(timezone.utc).isoformat(),
               'error_message': result['main'].get('error'),
               'daily_od_complete': len(config['hours']) == 19 and result['status'] == 'OK'}
    log.info('reports prepared; matrix export started')
    summary = _write_reports(directory, tables, summary, existing_artifacts=context_artifacts,
                             matrix_chunks=core.iter_matrix_tables(result['main'], prepared,
                                 artifacts=None if SAVE_DIAGNOSTIC_REPORTS else ('T_final', 'T_final_day')))
    log.info('reports and matrix export pass; outcome=%s', summary['outcome'])
    return summary


def save_failure(username, proj_name, run_id, message, diagnostics=None, config=None):
    """Записывает диагностику и сводку отказа для текущего запуска."""
    table = diagnostics if diagnostics is not None and len(diagnostics.columns) else pd.DataFrame([{'reason': message}])
    tables = {'failure': table}
    if config is not None:
        tables['parameters'] = pd.DataFrame([{'parameter': key, 'value': value} for key, value in config.items()])
    directory = _run_directory(username, proj_name, run_id)
    input_artifacts = {name: f'{name}.csv' for name in ('aggregation', 'unmapped', 'stop_access')
                       if (directory / f'{name}.csv').is_file()}
    return _write_reports(directory, tables, {
        'run_id': run_id, 'method_version': METHOD_VERSION, 'outcome': 'FAILED', 'error_message': message,
        'config': config,
    }, existing_artifacts=input_artifacts)


def read_summary(username, proj_name, run_id):
    """Читает сводку текущего запуска и согласует ее готовность со статусом в БД."""
    summary = json.loads((_run_directory(username, proj_name, run_id) / 'manifest.json').read_text(encoding='utf-8'))
    project = get_project(username, proj_name)
    if project is None or project['run_id'] != run_id or summary.get('run_id') != run_id:
        raise ValueError('Запуск больше не является текущим')
    if project['status_prj'] == 4 and summary['outcome'] != 'FAILED':
        # Записанный manifest не подтверждает завершение, если после прерывания БД хранит отказ.
        summary.update(outcome='FAILED', daily_od_complete=False, error_message=project['error_message'] or
                       'Расчет был прерван до фиксации завершения. Выполните новый запуск.')
        summary['artifacts'] = {name: value for name, value in summary['artifacts'].items()
                                if name not in ('T_final', 'T_final_day')}
    return summary


def artifact_path(username, proj_name, run_id, artifact, summary):
    """Проверяет реестр разрешенных отчетов и принадлежность файла каталогу запуска."""
    if artifact not in ARTIFACT_LABELS or summary.get('artifacts', {}).get(artifact) != f'{artifact}.csv':
        raise ValueError('Неизвестный файл результата')
    if not SAVE_DIAGNOSTIC_REPORTS and artifact in LARGE_DIAGNOSTIC_ARTIFACTS:
        raise ValueError('Подробная диагностика отключена')
    directory = _run_directory(username, proj_name, run_id).resolve()
    path = (directory / f'{artifact}.csv').resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise FileNotFoundError('Файл недоступен')
    return path


def result_table(username, proj_name, run_id, artifact, summary):
    """Готовит HTML-отчет с экранированием строк и нужным отбором основного контекста."""
    table = read_csv(artifact_path(username, proj_name, run_id, artifact, summary))
    if 'context' in table and artifact in ('margins', 'generation', 'statuses', 'calibration', 'target_reconciliation', 'ipf_support', 'los_preprocessing', 'generation_diagnostics', 'generation_alpha_trials', 'reconciliation_delta', 'reconciliation_delta_districts', 'correction_diagnostics', 'correction_districts'):
        table = table.loc[table.context.eq('main')]
    # Экранирование позволяет безопасно вставлять таблицу в шаблон; числа имеют пять значащих цифр.
    return table.to_html(index=False, escape=True, na_rep='—', classes='table_t1', float_format=lambda v: f'{v:.5g}')


def result_overview(username, proj_name, run_id, summary):
    """Сводит сохраненные районные итоги и статусы, не читая полные OD-матрицы."""
    overview = dict.fromkeys(('districts', 'total_trips', 'intrazonal_share', 'successful_hours',
                             'expected_hours', 'correction_share', 'correction_volume',
                             'unmapped_demand', 'unmapped_share'))
    overview['expected_hours'] = len((summary.get('config') or {}).get('hours', [])) or None
    diagnostics = []
    if 'aggregation' in summary.get('artifacts', {}):
        try:
            aggregation = read_csv(artifact_path(username, proj_name, run_id, 'aggregation', summary))
            source = float(aggregation.iloc[0].stop_demand)
            excluded = float(aggregation.iloc[0].excluded_demand)
            if not np.isfinite([source, excluded]).all() or not 0 <= excluded <= source:
                raise ValueError('Некорректный объем потерь')
            overview['unmapped_demand'] = excluded
            overview['unmapped_share'] = excluded / source if source > 0 else 0.0
        except (ValueError, OSError, KeyError, AttributeError, IndexError):
            diagnostics.append('Диагностика потерь при привязке недоступна.')
    available = 'T_final' in summary.get('artifacts', {})
    for name in ('statuses', 'margins'):
        if name not in summary.get('artifacts', {}):
            continue
        try:
            table = read_csv(artifact_path(username, proj_name, run_id, name, summary))
            if 'context' in table:
                table = table.loc[table.context.eq('main')]
            table = table.loc[table.state.eq('T_final') & table.hour.ne('day')]
            if name == 'statuses':
                failed = table.loc[table.status.ne('OK')]
                if len(failed):
                    diagnostics.append(f'Не собраны часы: {len(failed)} из {len(table)}.')
                if available:
                    successful = table.loc[table.status.eq('OK')]
                    overview['successful_hours'] = len(successful)
                    overview['expected_hours'] = overview['expected_hours'] or len(table)
                    if len(successful) and {'Q_C', 'Q_final'} <= set(successful):
                        total = successful.Q_final.sum(min_count=1)
                        correction = successful.Q_C.sum(min_count=1)
                        if pd.notna(correction):
                            overview['correction_volume'] = float(correction)
                        if total > 0 and pd.notna(correction):
                            overview['correction_share'] = float(correction / total)
            elif available and not table.empty:
                total = table.outgoing.sum(min_count=1)
                overview['districts'] = int(table.district_id.nunique())
                overview['total_trips'] = float(total) if pd.notna(total) else None
                overview['successful_hours'] = int(table.hour.nunique())
                if (total > 0 and 'intrazonal_share' in table
                        and table.loc[table.outgoing.gt(0), 'intrazonal_share'].notna().all()):
                    overview['intrazonal_share'] = float((table.outgoing * table.intrazonal_share).sum() / total)
        except (ValueError, OSError, KeyError, AttributeError):
            diagnostics.append(f'Сводные показатели отчета {name} недоступны.')
    main = (summary.get('contexts') or {}).get('main', {})
    transport = summary.get('transport_speed') or {}
    if transport.get('status') == 'FALLBACK_WITHOUT_TRANSPORT_SPEED':
        diagnostics.append(f"Транспортный признак отключен для всего расчета: для {len(transport['missing_ids'])} "
                           'районов нет исходной скорости и соседа с исходным значением в заданном радиусе. '
                           'Использованы прежние восемь признаков; список районов сохранен в отчете транспортного признака.')
    elif transport.get('enabled'):
        diagnostics.append('Интегральная скорость включена в генерацию и одинакова для base и forecast. '
                           'Независимая проверка с этим признаком недоступна: для пересчета внутри фолдов '
                           'нужны исходные скорости связей и веса потока; готовый признак взвешен по факту.')
    if 'correction_diagnostics' in summary.get('artifacts', {}):
        try:
            audit = read_csv(artifact_path(username, proj_name, run_id, 'correction_diagnostics', summary))
            audit = audit.loc[audit.context.eq('main')]
            subtotal = audit.loc[audit.hour.eq('selected_successful_hours')].iloc[0]
            if subtotal.status == 'NOT_APPLICABLE':
                diagnostics.append('Поправка C в режиме preserve_old не применяется.')
            elif subtotal.status in ('OK', 'PARTIAL'):
                overview['correction_volume'] = float(subtotal.Q_C) if pd.notna(subtotal.Q_C) else None
                overview['correction_share'] = float(subtotal.correction_share) if pd.notna(subtotal.correction_share) else None
                share = f'{subtotal.correction_share * 100:.4f} %' if pd.notna(subtotal.correction_share) else 'не определена'
                diagnostics.append(f'Поправка C: добавлено {subtotal.Q_C:.4f} поездки; доля итогового объема {share}. '
                                   f'Учтено часов {int(subtotal.available_hours)} из {int(subtotal.expected_hours)}. '
                                   'Критерий допустимости поправки C не задан.')
                hourly = audit.loc[audit.hour.isin(HOUR_COLUMNS)].dropna(subset=['correction_share'])
                if len(hourly):
                    peak = hourly.loc[hourly.correction_share.idxmax()]
                    diagnostics.append(f'Максимальная часовая доля C: {peak.hour}, {peak.correction_share * 100:.4f} %. '
                                       'Условия компенсации по районам сохранены в отдельном отчете; '
                                       'совместное сохранение отправлений и прибытий не проверялось.')
            else:
                diagnostics.append('Диагностика поправки C недоступна: нет успешных сборок.')
        except (ValueError, OSError, KeyError, AttributeError, IndexError):
            diagnostics.append('Диагностика поправки C недоступна.')
    timings = main.get('timings') or {}
    if timings:
        diagnostics.append(f"Согласование целей всех 19 часов: {timings['reconciliation_seconds']:.2f} с; "
                           f"весь основной контекст: {timings['total_seconds']:.2f} с. "
                           'Разложение прироста и сокращений доступно в отчете влияния на модельное изменение.')
    generation = main.get('generation_diagnostics') or {}
    if generation:
        diagnostics.append(f"Генерация: alpha = {generation['selected_alpha']:g}; {generation['selection_status']}; "
                           f"обучающих районов {generation['training_districts']}, "
                           f"ранг признаков {generation['feature_rank']} из {generation['active_features']}.")
        if generation.get('warnings'):
            diagnostics.append('Предупреждения генерации: ' + ', '.join(generation['warnings']))
    if main.get('zero_observed_day'):
        diagnostics.append('Все наблюдаемые часы нулевые; ненулевой спрос не оценен.')
    los_errors = Counter(error['status'] for error in main.get('los_errors', []))
    for status, count in los_errors.most_common(3):
        diagnostics.append(f'LOS: {status} — связей: {count}.')
    warnings = (main.get('los_model') or {}).get('warnings', [])
    los_model = main.get('los_model') or {}
    minimum_speed = los_model.get('minimum_speed_kmh')
    if isinstance(minimum_speed, (int, float)) and np.isfinite(minimum_speed) and minimum_speed > 0:
        source = {'observed_training_los': 'наблюдаемый обучающий LOS',
                  'fixed_preprocessing_threshold': 'фиксированный порог предобработки LOS'}.get(
                      los_model.get('minimum_speed_source'), 'технический резерв при отсутствии межрайонного LOS')
        diagnostics.append(f'Нижняя граница скорости: {minimum_speed:.3f} км/ч; источник — {source}.')
    if los_model.get('minimum_speed_source') == 'fixed_preprocessing_threshold':
        diagnostics.append(f"Предобработка LOS: скорректировано связей {los_model['preprocessed_pairs']} из {los_model['observed_pairs']}.")
    if warnings:
        diagnostics.append('Предупреждения LOS: ' + ', '.join(warnings[:3]))
    failures = main.get('failed_states', [])
    if failures:
        codes = ', '.join(dict.fromkeys(item['status'] for item in failures))
        diagnostics.append(f'Отказы модельных состояний: {len(failures)} ({codes[:200]}).')
        reason = next((item.get('reason') for item in failures if item.get('reason')), None)
        if reason:
            diagnostics.append('Основная причина: ' + str(reason)[:250])
    overview['target_reconciliation'] = []
    overview['target_reconciliation_status'] = ('unavailable' if summary.get('method_version') in TARGET_RECONCILIATION_VERSIONS
                                                else 'historical')
    if (summary.get('method_version') in TARGET_RECONCILIATION_VERSIONS
            and 'target_reconciliation' in summary.get('artifacts', {})):
        try:
            target = read_csv(artifact_path(username, proj_name, run_id, 'target_reconciliation', summary))
            required = {'context', 'state', 'hour', 'status', 'Q_model', 'Q_feasible', 'Q_reduced'}
            if not required <= set(target):
                raise ValueError('Incomplete reconciliation report')
            target = target.loc[target.context.eq('main') & target.hour.ne('day')]
            if target.empty or target.duplicated(['state', 'hour']).any():
                raise ValueError('Missing or duplicate reconciliation rows')
            for state, label in (('base', 'Базовое состояние'), ('forecast', 'Прогнозное состояние')):
                rows = target.loc[target.state.eq(state)]
                selected = (summary.get('config') or {}).get('hours', [])
                rows = rows.loc[rows.hour.isin(selected)]
                if rows.empty or rows.status.eq('NOT_REQUIRED').all():
                    continue
                good = rows.loc[rows.status.isin(('TARGETS_UNCHANGED', 'TARGETS_REDUCED', 'ZERO_FEASIBLE_VOLUME'))]
                values = good[['Q_model', 'Q_feasible', 'Q_reduced']].apply(pd.to_numeric, errors='coerce')
                if len(good) and (not np.isfinite(values.to_numpy()).all()
                                  or (values.to_numpy() < 0).any()
                                  or not np.allclose(values.Q_model, values.Q_feasible + values.Q_reduced,
                                                     rtol=1e-8, atol=1e-8)):
                    raise ValueError('Invalid reconciliation volumes')
                item = core._reconciliation_summary(good.to_dict('records'))
                item.update(state=state, label=label, successful_hours=len(good),
                            expected_hours=len(selected), complete=len(good) == len(selected))
                overview['target_reconciliation'].append(item)
            if not any(item['state'] == 'forecast' for item in overview['target_reconciliation']):
                raise ValueError('Forecast reconciliation is missing')
            overview['target_reconciliation_status'] = 'available'
        except (ValueError, OSError, KeyError, AttributeError):
            overview['target_reconciliation_status'] = 'unavailable'
            overview['target_reconciliation'] = []
            diagnostics.append('Диагностика согласования модельных целей недоступна.')
    overview['diagnostics'] = diagnostics
    return overview


def result_downloads(summary):
    """One final matrix and the available primary diagnostic reports."""
    available = summary.get('artifacts') or {}
    matrix = ('T_final_day' if summary.get('daily_od_complete') and 'T_final_day' in available
              else 'T_final' if 'T_final' in available else None)
    names = [matrix] if matrix else []
    alternatives = {'statuses': ('failure', 'statuses'),
                    'spatial_benchmark_comparison': ('spatial_benchmark_comparison', 'spatial_metrics', 'metrics')}
    for name in PRIMARY_ARTIFACTS:
        selected = next((item for item in alternatives.get(name, (name,)) if item in available), None)
        if selected:
            names.append(selected)
    return [(name, ARTIFACT_LABELS[name]) for name in names]


def result_presentation(summary, overview):
    """Localize saved results and select seven indicators without reading matrices."""
    config = summary.get('config') or {}
    main = (summary.get('contexts') or {}).get('main') or {}
    hours = config.get('hours') or []

    def number(value, digits=2):
        if isinstance(value, (bool, str)) or value is None or not np.isfinite(value):
            return 'нет данных'
        return f'{value:,.{digits}f}'.replace(',', ' ')

    def percent(value):
        return number(value * 100, 4) + ' %' if value is not None else 'не определена'

    def coverage(item):
        status = {'VALIDATION_COMPLETE': 'выполнена', 'VALIDATION_PARTIAL': 'частичная',
                  'VALIDATION_UNAVAILABLE': 'недоступна'}.get(item.get('status'), 'нет данных')
        if item.get('expected_forecast_hours') is not None:
            status += (f"; успешных часов фолдов {number(item.get('successful_forecast_hours'), 0)} "
                       f"из {number(item['expected_forecast_hours'], 0)}")
        return status

    status = {'OK': 'успешно', 'FAILED': 'ошибка', 'PARTIAL': 'выполнен частично'}.get(
        str(summary.get('outcome', '')).upper(), 'нет данных')
    period = ('Сутки' if set(hours) == set(HOUR_COLUMNS) else 'Час' if len(hours) == 1 else
              f'Несколько часов ({len(hours)})' if hours else 'Нет данных')
    reconciliation = []
    for item in overview['target_reconciliation']:
        text = (f"{item['label']}: исходный объем {number(item.get('Q_model'))}, "
                f"доступный {number(item.get('Q_feasible'))}, сокращение {number(item.get('Q_reduced'))}")
        if not item['complete']:
            text += f" (частичные данные: {item['successful_hours']} из {item['expected_hours']} часов)"
        reconciliation.append(text)
    reconciled = '; '.join(reconciliation) or (
        'Не рассчитывалось в этой версии' if overview['target_reconciliation_status'] == 'historical' else 'Нет данных')
    loss = 'Нет данных'
    if overview['unmapped_demand'] is not None:
        loss = f"{number(overview['unmapped_demand'])} поездок ({number(overview['unmapped_share'] * 100)} % исходного спроса)"
    correction = ('Не применяется' if config.get('recalculation_mode') == 'preserve_old' else
                  f"{number(overview['correction_volume'], 4)} поездки; доля {percent(overview['correction_share'])}"
                  if overview['correction_volume'] is not None else 'Нет данных')
    generation = main.get('generation_diagnostics') or {}
    fit = 'Нет данных'
    if generation:
        fit = (f"alpha = {number(generation.get('selected_alpha'), 2)}; ранг признаков "
               f"{number(generation.get('feature_rank'), 0)} из {number(generation.get('active_features'), 0)}")
        if generation.get('warnings'):
            fit += '; предупреждения: ' + ', '.join(generation['warnings'][:3])
    failures, los_errors = main.get('failed_states', []), main.get('los_errors', [])
    errors = 'Нет данных'
    if 'failed_states' in main or 'los_errors' in main:
        errors = f'Отказы модельных состояний: {len(failures)}; ошибки LOS, связей: {len(los_errors)}'
        codes = list(dict.fromkeys(item['status'] for item in [*failures, *los_errors]))
        if codes:
            errors += ' (' + ', '.join(codes[:3]) + ')'
        reason = next((item.get('reason') for item in failures if item.get('reason')), None)
        if reason:
            errors += '; основная причина: ' + str(reason)[:250]
    validation = summary.get('validation') or {}
    return {'status': status, 'period': period,
            'mode': {'recalculate_all': 'Полный пересчет спроса',
                     'preserve_old': 'Сохранение исходного спроса существующих районов'}.get(
                         config.get('recalculation_mode'), 'Нет данных'),
            'diagnostics': [
                ('Исключено поездок при привязке исходной матрицы за сутки', loss),
                ('Согласование модельных целей', reconciled),
                ('Поправка C', correction),
                ('Проверка переноса: случайные фолды', coverage(validation)),
                ('Проверка пространственными блоками', coverage(validation.get('spatial_validation') or {})),
                ('Генерация', fit),
                ('Ошибки расчета', errors),
            ]}
