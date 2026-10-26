import os
import math
import logging
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import shutil
from shapely.errors import GEOSException


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger('rsim')
STRING_DTYPES = ['object', 'str', 'string', 'string[python]', 'string[pyarrow]']
INT16_DTYPES = ['int16', 'Int16', 'int16[pyarrow]']
INT32_DTYPES = ['int32', 'Int32', 'int32[pyarrow]']
INT64_DTYPES = ['int64', 'Int64', 'int64[pyarrow]']
FLOAT16_DTYPES = ['float16']
FLOAT32_DTYPES = ['float32', 'Float32', 'float32[pyarrow]']
FLOAT64_DTYPES = ['float64', 'Float64', 'double[pyarrow]', 'float64[pyarrow]']
INT_DTYPES = INT16_DTYPES + INT32_DTYPES + INT64_DTYPES
NUMERIC_DTYPES = INT_DTYPES + FLOAT16_DTYPES + FLOAT32_DTYPES + FLOAT64_DTYPES


def _dtype_matches(actual_dtype, allowed_dtypes):
    # Сравнивает dtype Pandas с допустимыми типами, поддерживая строковые dtype новых версий Pandas
    dtype_name = str(actual_dtype)
    if isinstance(allowed_dtypes, str):
        allowed_dtypes = [allowed_dtypes]
    return dtype_name in allowed_dtypes


def allowed_file(filename):
    # Проверяет, что файл имеет допустимое расширение CSV
    allowed_ext = {'csv'}
    result = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext
    logger.info('allowed_file: filename=%s, allowed=%s', filename, result)
    return result


def check_disk_size(user_name):
    # Считает суммарный размер файлов пользователя в мегабайтах
    logger.info('check_disk_size: user_name=%s', user_name)
    userpath = os.path.join(BASE_DIR, f'static/userdata/{user_name}/')
    disk_size = 0
    for path, dirs, files in os.walk(userpath):
        for i in files:
            fp = os.path.join(path, i)
            disk_size += os.path.getsize(fp)
    result = int(disk_size/1024/1024)
    logger.info('check_disk_size: user_name=%s, size_mb=%s', user_name, result)
    return result

    
def path_gen(user_name, proj_type, proj_name, create_folder):
    # Создает структуру директорий проекта или возвращает пути к input/output
    logger.info(
        'path_gen: user_name=%s, proj_type=%s, proj_name=%s, create_folder=%s',
        user_name, proj_type, proj_name, create_folder
    )
    if create_folder:
        userpath = os.path.join(BASE_DIR, f'static/userdata/{user_name}/')
        if os.path.exists(userpath) == False:
            os.makedirs(userpath)
        prj_type_path = os.path.join(BASE_DIR, f'static/userdata/{user_name}/{proj_type}/')
        if os.path.exists(prj_type_path) == False:
            os.mkdir(prj_type_path)
        src_files_path = prj_type_path + proj_name + '/'
        if os.path.exists(src_files_path) == False:
            os.mkdir(src_files_path)
            input_folder = os.path.join(src_files_path, 'input')
            os.mkdir(input_folder)
            output_folder = os.path.join(src_files_path, 'output')
            os.mkdir(output_folder)
        logger.info('path_gen: project folders ensured for %s/%s/%s', user_name, proj_type, proj_name)
    else:
        userpath_src = os.path.join(BASE_DIR, f'static/userdata/{user_name}/{proj_type}/{proj_name}/input/')
        userpath_out = os.path.join(BASE_DIR, f'static/userdata/{user_name}/{proj_type}/{proj_name}/output/')
        logger.info('path_gen: source_path=%s, output_path=%s', userpath_src, userpath_out)
        return userpath_src, userpath_out


def delete_project(user_name, proj_type, proj_name):
        # Удаляет файлы проекта и вложенные каталоги input/output
        logger.info('delete_project: user_name=%s, proj_type=%s, proj_name=%s', user_name, proj_type, proj_name)
        dir_project = os.path.join(BASE_DIR, f'static/userdata/{user_name}/{proj_type}/{proj_name}')
        base = os.path.join(BASE_DIR, 'static', 'userdata')
        target = os.path.abspath(dir_project)
        if (any(not isinstance(value, str) or not value or value in ('.', '..')
                or any(char in value for char in '/\\:')
                for value in (user_name, proj_type, proj_name))
                or os.path.commonpath([base, target]) != base):
            raise ValueError('Некорректный путь проекта')
        shutil.rmtree(target)
        logger.info('delete_project: removed %s', dir_project)


def get_df(path_src, ver, f_prefix):
    # Загружает CSV-файл проекта в DataFrame по префиксу и версии
    filename = f'{f_prefix}_{ver}.csv'
    logger.info('get_df: path=%s, filename=%s', path_src, filename)
    df = pd.read_csv(path_src + filename, delimiter=';')
    logger.info('get_df: loaded shape=%s', df.shape)
    return df


def _invalid_numeric_values(df, columns):
    values = df[columns].to_numpy(dtype='float64')
    return not np.isfinite(values).all()


def _invalid_links(df):
    expected = df['source'].astype(str) + '>' + df['target'].astype(str)
    return df['link'].astype(str).str.strip() != expected


def _validate_rnd_semantics(rnd):
    if rnd.empty:
        return 'таблица описания маршрутов не содержит записей'
    if rnd.isna().any().any():
        return 'таблица описания маршрутов содержит пустые значения'
    if rnd['rnd'].astype(str).str.strip().eq('').any():
        return 'поле "rnd" содержит пустой код маршрута'
    if _invalid_numeric_values(rnd, ['order', 'source', 'target', 'distance', 'tr_time', 'interval', 'carry', 'tarif']):
        return 'таблица описания маршрутов содержит бесконечные или некорректные числовые значения'
    if rnd.duplicated(['rnd', 'order']).any():
        return 'обнаружены повторяющиеся значения пары "rnd-order"'
    if (rnd['source'] == rnd['target']).any():
        return 'обнаружен маршрутный перегон с одинаковыми source и target'
    if (rnd[['distance', 'tr_time', 'interval', 'carry']] <= 0).any().any():
        return 'поля distance, tr_time, interval и carry должны содержать положительные значения'
    if (rnd['tarif'] < 0).any():
        return 'поле tarif не может содержать отрицательные значения'

    block_starts = rnd['rnd'].ne(rnd['rnd'].shift())
    if rnd.loc[block_starts, 'rnd'].duplicated().any():
        return 'строки одного rnd должны находиться в одном непрерывном блоке'

    for rnd_code, route in rnd.groupby('rnd', sort=False):
        if not route['order'].is_monotonic_increasing:
            return f'для rnd "{rnd_code}" нарушен возрастающий порядок поля order'
        if route.shape[0] > 1:
            targets = route['target'].iloc[:-1].to_numpy()
            next_sources = route['source'].iloc[1:].to_numpy()
            if not np.array_equal(targets, next_sources):
                return f'для rnd "{rnd_code}" нарушена последовательность target-source соседних перегонов'
    return None


def _validate_matrix_semantics(mtrx):
    if mtrx.empty:
        return 'матрица межостановочных расстояний не содержит записей'
    if mtrx.isna().any().any():
        return 'матрица межостановочных расстояний содержит пустые значения'
    if _invalid_numeric_values(mtrx, ['source', 'target', 'distance']):
        return 'матрица межостановочных расстояний содержит бесконечные или некорректные числовые значения'
    if (mtrx['source'] == mtrx['target']).any():
        return 'матрица содержит связь с одинаковыми source и target'
    if (mtrx['distance'] < 0).any():
        return 'поле distance не может содержать отрицательные значения'
    if _invalid_links(mtrx).any():
        return 'значение link должно соответствовать формату source>target'
    if mtrx.duplicated(['source', 'target']).any() or mtrx['link'].duplicated().any():
        return 'матрица содержит повторяющиеся связи source-target'
    return None


def _validate_stops_semantics(stops):
    if stops.empty:
        return 'справочник остановочных пунктов не содержит записей'
    if stops.isna().any().any():
        return 'справочник остановочных пунктов содержит пустые значения'
    if _invalid_numeric_values(stops, ['id_stop', 'cluster', 'tr_district']):
        return 'справочник остановочных пунктов содержит бесконечные или некорректные числовые значения'
    if stops['id_stop'].duplicated().any():
        return 'поле id_stop должно содержать уникальные идентификаторы остановок'
    return None


def _validate_flow_semantics(flow, flow_columns):
    if flow.empty:
        return 'матрица пассажирских корреспонденций не содержит записей'
    if flow.isna().any().any():
        return 'матрица пассажирских корреспонденций содержит пустые значения'
    if _invalid_numeric_values(flow, ['source', 'target'] + flow_columns):
        return 'матрица пассажирских корреспонденций содержит бесконечные или некорректные числовые значения'
    if (flow[flow_columns] < 0).any().any():
        return 'пассажирский спрос не может содержать отрицательные значения'
    if _invalid_links(flow).any():
        return 'значение link должно соответствовать формату source>target'
    if flow.duplicated(['source', 'target']).any() or flow['link'].duplicated().any():
        return 'матрица пассажирских корреспонденций содержит повторяющиеся связи source-target'
    return None


def check_source_data(proj_path_src, filename, prefix_name, net_limit):
    '''
    Проверям входные данные на корректность
    '''
    logger.info(
        'check_source_data: path=%s, filename=%s, prefix_name=%s, net_limit=%s',
        proj_path_src, filename, prefix_name, net_limit
    )
    if prefix_name == 'rnd':
        # check rnd
        rnd_valid_types = {'rnd': STRING_DTYPES, 
                            'order': INT_DTYPES, 
                            'source': INT32_DTYPES + INT64_DTYPES, 
                            'target': INT32_DTYPES + INT64_DTYPES, 
                            'distance': FLOAT16_DTYPES + FLOAT32_DTYPES + FLOAT64_DTYPES, 
                            'tr_time': FLOAT16_DTYPES + FLOAT32_DTYPES + FLOAT64_DTYPES, 
                            'interval': NUMERIC_DTYPES, 
                            'carry': INT_DTYPES + FLOAT32_DTYPES + FLOAT64_DTYPES, 
                            'tarif': INT_DTYPES + FLOAT32_DTYPES + FLOAT64_DTYPES}

        rnd = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in rnd_valid_types:
            if i not in rnd.columns: 
                logger.warning('check_source_data[rnd]: missing column %s', i)
                return False, f'Rsim: в таблице описания маршрутов не обнаружено поле "{i}"'
        for c, t in rnd.dtypes.items():
            if c not in rnd_valid_types:
                logger.warning('check_source_data[rnd]: unknown column %s', c)
                return False, f'Rsim: в таблице описания маршрутов неизвестное поле "{c}"'
            if not _dtype_matches(t, rnd_valid_types[c]):
                logger.warning('check_source_data[rnd]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице описания маршрутов поле "{c}" содержит некорректный тип данных {t}'
        if rnd.shape[0] > net_limit:
            logger.warning('check_source_data[rnd]: row limit exceeded, rows=%s', rnd.shape[0])
            return False, f'Rsim: необходимо уменьшить количество записей в таблице - не более {net_limit} строк'
        semantic_error = _validate_rnd_semantics(rnd)
        if semantic_error:
            logger.warning('check_source_data[rnd]: semantic validation failed: %s', semantic_error)
            return False, f'Rsim: {semantic_error}'
        logger.info('check_source_data[rnd]: validation passed, rows=%s', rnd.shape[0])
        return True, 'Таблица описания маршрутов успешно загружена. Загрузите матрицу межостановочных расстояний.'

    if prefix_name == 'matrix_dist_stops':
        # check matrix
        mtrx_valid_types = {'link': STRING_DTYPES, 
                            'source': INT_DTYPES, 
                            'target': INT_DTYPES, 
                            'distance': FLOAT64_DTYPES}
        
        mtrx = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in mtrx_valid_types:
            if i not in mtrx.columns:
                logger.warning('check_source_data[matrix_dist_stops]: missing column %s', i)
                return False, f'Rsim: в таблице межостановочных расстояний не обнаружено поле "{i}"'
        for c, t in mtrx.dtypes.items():
            if c not in mtrx_valid_types: 
                logger.warning('check_source_data[matrix_dist_stops]: unknown column %s', c)
                return False, f'Rsim: в таблице межостановочных расстояний неизвестное поле "{c}"'
            if not _dtype_matches(t, mtrx_valid_types[c]):
                logger.warning('check_source_data[matrix_dist_stops]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице межостановочных расстояний поле "{c}" содержит некорректный тип данных {t}'

        semantic_error = _validate_matrix_semantics(mtrx)
        if semantic_error:
            logger.warning('check_source_data[matrix_dist_stops]: semantic validation failed: %s', semantic_error)
            return False, f'Rsim: {semantic_error}'

        suffix = filename[len(prefix_name):]
        rnd_path = os.path.join(proj_path_src, 'rnd' + suffix)
        if os.path.isfile(rnd_path):
            rnd = pd.read_csv(rnd_path, delimiter=';', encoding='utf-8')
            rnd_stops = set(rnd['source']) | set(rnd['target'])
            matrix_stops = set(mtrx['source']) | set(mtrx['target'])
            missing_stops = sorted(rnd_stops - matrix_stops)
            if missing_stops:
                logger.warning('check_source_data[matrix_dist_stops]: route stops missing in matrix: %s', missing_stops[:5])
                return False, f'Rsim: в матрице расстояний отсутствуют остановки маршрутов: {missing_stops[:5]}'

        logger.info('check_source_data[matrix_dist_stops]: validation passed, rows=%s', mtrx.shape[0])
        return True, 'Таблица межостановочных расстояний успешно загружена. Загрузите справочник остановочных пунктов.'

    if prefix_name == 'flow':
    # check flow
        f_hours = ['h' + str(h) for h in range(5, 24)]
        h_dict = {h:FLOAT64_DTYPES + INT64_DTYPES + INT32_DTYPES + INT16_DTYPES for h in f_hours}
        flow_valid_types = {'link': STRING_DTYPES, 
                            'source': INT_DTYPES, 
                            'target': INT_DTYPES, 
                            'day': NUMERIC_DTYPES}
        flow_valid_types = flow_valid_types | h_dict

        flow = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in flow_valid_types:
            if i not in flow.columns:
                logger.warning('check_source_data[flow]: missing column %s', i)
                return False, f'Rsim: в таблице пассажирских корреспонденций не обнаружено поле "{i}"'
        for c, t in flow.dtypes.items():
            if c not in flow_valid_types: 
                logger.warning('check_source_data[flow]: unknown column %s', c)
                return False, f'Rsim: в таблице пассажирских корреспонденций неизвестное поле "{c}"'
            if not _dtype_matches(t, flow_valid_types[c]):
                logger.warning('check_source_data[flow]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице пассажирских корреспонденций поле "{c}" содержит некорректный тип данных {t}'

        flow_columns = ['day'] + f_hours
        semantic_error = _validate_flow_semantics(flow, flow_columns)
        if semantic_error:
            logger.warning('check_source_data[flow]: semantic validation failed: %s', semantic_error)
            return False, f'Rsim: {semantic_error}'

        suffix = filename[len(prefix_name):]
        stops_path = os.path.join(proj_path_src, 'ref_stops' + suffix)
        if os.path.isfile(stops_path):
            stops = pd.read_csv(stops_path, delimiter=';', encoding='utf-8')
            stop_ids = set(stops['id_stop'])
            flow_stops = set(flow['source']) | set(flow['target'])
            missing_stops = sorted(flow_stops - stop_ids)
            if missing_stops:
                logger.warning('check_source_data[flow]: stops missing in ref_stops: %s', missing_stops[:5])
                return False, f'Rsim: в справочнике остановок отсутствуют остановки матрицы корреспонденций: {missing_stops[:5]}'

        matrix_path = os.path.join(proj_path_src, 'matrix_dist_stops' + suffix)
        if os.path.isfile(matrix_path):
            mtrx = pd.read_csv(matrix_path, delimiter=';', encoding='utf-8', usecols=['link'])
            non_diagonal_flow = flow[flow['source'] != flow['target']]
            missing_links = sorted(set(non_diagonal_flow['link']) - set(mtrx['link']))
            if missing_links:
                logger.warning('check_source_data[flow]: links missing in distance matrix: %s', missing_links[:5])
                return False, f'Rsim: в матрице расстояний отсутствуют корреспонденции: {missing_links[:5]}'

        logger.info('check_source_data[flow]: validation passed, rows=%s', flow.shape[0])
        return True, 'Матрица корреспонденций успешно загружена.'
        
    if prefix_name == 'ref_stops':
    # check ref_stops
        stops_valid_types = {'id_stop': INT_DTYPES, 'cluster': INT_DTYPES, 'tr_district': INT_DTYPES}

        stops = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in stops_valid_types:
            if i not in stops.columns:
                logger.warning('check_source_data[ref_stops]: missing column %s', i)
                return False, f'Rsim: в таблице остановочных пунктов не обнаружено поле "{i}"'
        for c, t in stops.dtypes.items():
            if c not in stops_valid_types:
                logger.warning('check_source_data[ref_stops]: unknown column %s', c)
                return False, f'Rsim: в таблице остановочных пунктов неизвестное поле "{c}"'
            if not _dtype_matches(t, stops_valid_types[c]):
                logger.warning('check_source_data[ref_stops]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице остановочных пунктов поле "{c}" содержит некорректный тип данных {t}'

        semantic_error = _validate_stops_semantics(stops)
        if semantic_error:
            logger.warning('check_source_data[ref_stops]: semantic validation failed: %s', semantic_error)
            return False, f'Rsim: {semantic_error}'

        suffix = filename[len(prefix_name):]
        referenced_stops = set()
        rnd_path = os.path.join(proj_path_src, 'rnd' + suffix)
        if os.path.isfile(rnd_path):
            rnd = pd.read_csv(rnd_path, delimiter=';', encoding='utf-8', usecols=['source', 'target'])
            referenced_stops.update(rnd['source'])
            referenced_stops.update(rnd['target'])
        matrix_path = os.path.join(proj_path_src, 'matrix_dist_stops' + suffix)
        if os.path.isfile(matrix_path):
            mtrx = pd.read_csv(matrix_path, delimiter=';', encoding='utf-8', usecols=['source', 'target'])
            referenced_stops.update(mtrx['source'])
            referenced_stops.update(mtrx['target'])
        missing_stops = sorted(referenced_stops - set(stops['id_stop']))
        if missing_stops:
            logger.warning('check_source_data[ref_stops]: referenced stops missing: %s', missing_stops[:5])
            return False, f'Rsim: в справочнике отсутствуют используемые остановки: {missing_stops[:5]}'

        logger.info('check_source_data[ref_stops]: validation passed, rows=%s', stops.shape[0])
        return True, 'Справочник остановочных пунктов успешно загружен. Загрузите матрицу корреспонденций (опционально).'

    if prefix_name == 'geom_stops':
    # check geom stops
        gstops_valid_types = {'id_stop': INT_DTYPES, 'wkt': STRING_DTYPES, 'name': STRING_DTYPES} #'id_cluster': 'int64', 

        geom_stops = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in gstops_valid_types:
            if i not in geom_stops.columns:
                logger.warning('check_source_data[geom_stops]: missing column %s', i)
                return False, f'Rsim: в таблице пространственных данных остановочных пунктов не обнаружено поле "{i}"'
        for c, t in geom_stops.dtypes.items():
            if c not in gstops_valid_types:
                logger.warning('check_source_data[geom_stops]: unknown column %s', c)
                return False, f'Rsim: в таблице пространственных данных остановочных пунктов неизвестное поле "{c}"'
            if not _dtype_matches(t, gstops_valid_types[c]):
                logger.warning('check_source_data[geom_stops]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице пространственных данных остановочных пунктов поле "{c}" содержит некорректный тип данных {t}'

        logger.info('check_source_data[geom_stops]: validation passed, rows=%s', geom_stops.shape[0])
        return True, 'Справочник пространственных данных остановочных пунктов успешно загружен. Загрузите данные о кластерах.'

    if prefix_name == 'geom_clusters':
    # check geom clusters
        gstops_valid_types = {'id_cluster': INT_DTYPES, 'wkt': STRING_DTYPES}

        geom_stops = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in gstops_valid_types:
            if i not in geom_stops.columns:
                logger.warning('check_source_data[geom_clusters]: missing column %s', i)
                return False, f'Rsim: в таблице пространственных данных кластеров ОП не обнаружено поле "{i}"'
        for c, t in geom_stops.dtypes.items():
            if c not in gstops_valid_types:
                logger.warning('check_source_data[geom_clusters]: unknown column %s', c)
                return False, f'Rsim: в таблице пространственных данных кластеров ОП неизвестное поле "{c}"'
            if not _dtype_matches(t, gstops_valid_types[c]):
                logger.warning('check_source_data[geom_clusters]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице пространственных данных кластеров ОП поле "{c}" содержит некорректный тип данных {t}'

        logger.info('check_source_data[geom_clusters]: validation passed, rows=%s', geom_stops.shape[0])
        return True, 'Справочник пространственных данных кластеров ОП успешно загружен. Загрузите данные о транспортных районах.'

    if prefix_name == 'geom_tr_distr':
    # check geom clusters
        gstops_valid_types = {'id_tr_distr': INT_DTYPES, 'wkt': STRING_DTYPES}

        geom_stops = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in gstops_valid_types:
            if i not in geom_stops.columns:
                logger.warning('check_source_data[geom_tr_distr]: missing column %s', i)
                return False, f'Rsim: в таблице пространственных данных транспортных районов не обнаружено поле "{i}"'
        for c, t in geom_stops.dtypes.items():
            if c not in gstops_valid_types:
                logger.warning('check_source_data[geom_tr_distr]: unknown column %s', c)
                return False, f'Rsim: в таблице пространственных данных транспортных районов неизвестное поле "{c}"'
            if not _dtype_matches(t, gstops_valid_types[c]):
                logger.warning('check_source_data[geom_tr_distr]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице пространственных данных транспортных районов поле "{c}" содержит некорректный тип данных {t}'

        logger.info('check_source_data[geom_tr_distr]: validation passed, rows=%s', geom_stops.shape[0])
        return True, 'Справочник пространственных данных транспортных районов успешно загружен.'

    if prefix_name == 'rnd_short':
        # check rnd
        rnd_valid_types = {'rnd': STRING_DTYPES, 
                            'order': INT_DTYPES, 
                            'source': INT_DTYPES, 
                            'target': INT_DTYPES, 
                            'distance': NUMERIC_DTYPES}

        rnd = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in rnd_valid_types:
            if i not in rnd.columns: 
                logger.warning('check_source_data[rnd_short]: missing column %s', i)
                return False, f'Rsim: в таблице описания маршрутов не обнаружено поле "{i}"'
        for c, t in rnd.dtypes.items():
            if c not in rnd_valid_types: 
                logger.warning('check_source_data[rnd_short]: unknown column %s', c)
                return False, f'Rsim: в таблице описания маршрутов неизвестное поле "{c}"'
            if not _dtype_matches(t, rnd_valid_types[c]):
                logger.warning('check_source_data[rnd_short]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице описания маршрутов поле "{c}" содержит некорректный тип данных {t}'
        if rnd.shape[0] > net_limit:
            logger.warning('check_source_data[rnd_short]: row limit exceeded, rows=%s', rnd.shape[0])
            return False, f'Rsim: необходимо уменьшить количество записей в таблице - не более {net_limit} строк'
        logger.info('check_source_data[rnd_short]: validation passed, rows=%s', rnd.shape[0])
        return True, 'Таблица описания маршрутов успешно загружена. Загрузите таблицу корреспонденций на маршрутах (опционально).'

    if prefix_name == 'routes_flow':
    # check routes_flow
        f_hours = ['h' + str(h) for h in range(5, 24)]
        h_dict = {h:FLOAT64_DTYPES + FLOAT32_DTYPES + INT64_DTYPES + INT32_DTYPES + INT16_DTYPES for h in f_hours}
        routes_flow_valid_types = {'route': STRING_DTYPES + INT_DTYPES, 
                                    'link': STRING_DTYPES, 
                                    'source': INT_DTYPES, 
                                    'target': INT_DTYPES, 
                                    'day': NUMERIC_DTYPES}
        routes_flow_valid_types = routes_flow_valid_types | h_dict

        routes_flow = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in routes_flow_valid_types:
            if i not in routes_flow.columns:
                logger.warning('check_source_data[routes_flow]: missing column %s', i)
                return False, f'Rsim: в таблице маршрутных корреспонденций не обнаружено поле "{i}"'
        for c, t in routes_flow.dtypes.items():
            if c not in routes_flow_valid_types: 
                logger.warning('check_source_data[routes_flow]: unknown column %s', c)
                return False, f'Rsim: в таблице маршрутных корреспонденций неизвестное поле "{c}"'
            if not _dtype_matches(t, routes_flow_valid_types[c]):
                logger.warning('check_source_data[routes_flow]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице маршрутных корреспонденций поле "{c}" содержит некорректный тип данных {t}'

        logger.info('check_source_data[routes_flow]: validation passed, rows=%s', routes_flow.shape[0])
        return True, 'Матрица маршрутных корреспонденций успешно загружена. Загрузите таблицу провозных возможностей.'

    if prefix_name == 'carry':
    # check carry
        f_hours = ['h' + str(h) for h in range(5, 24)]
        h_dict = {h:FLOAT64_DTYPES + FLOAT32_DTYPES + INT64_DTYPES + INT32_DTYPES + INT16_DTYPES for h in f_hours}
        carry_valid_types = {'rnd': STRING_DTYPES, 
                                'day': NUMERIC_DTYPES}
        carry_valid_types = carry_valid_types | h_dict

        carry = pd.read_csv(proj_path_src + filename, delimiter=';', encoding='utf-8')
        for i in carry_valid_types:
            if i not in carry.columns:
                logger.warning('check_source_data[carry]: missing column %s', i)
                return False, f'Rsim: в таблице провозных возможностей не обнаружено поле "{i}"'
        for c, t in carry.dtypes.items():
            if c not in carry_valid_types: 
                logger.warning('check_source_data[carry]: unknown column %s', c)
                return False, f'Rsim: в таблице провозных возможностей неизвестное поле "{c}"'
            if not _dtype_matches(t, carry_valid_types[c]):
                logger.warning('check_source_data[carry]: invalid dtype for %s -> %s', c, t)
                return False, f'Rsim: в таблице провозных возможностей поле "{c}" содержит некорректный тип данных {t}'
        logger.info('check_source_data[carry]: validation passed, rows=%s', carry.shape[0])
        return True, 'Таблица провозных возможностей успешно загружена.'

    logger.warning('check_source_data: unknown prefix_name=%s', prefix_name)


def select_utm_crs(layer):
    """Выбирает зону и полушарие WGS 84 / UTM по охвату одного слоя.

    Принимает GeoDataFrame или GeoSeries с известной CRS; возвращает CRS
    для to_crs(). Пустой слой и территория вне охвата UTM дают ValueError.
    """
    if not isinstance(layer, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise ValueError('Для выбора UTM нужен географический слой')
    if layer.empty:
        raise ValueError('Нельзя выбрать UTM по пустому слою')
    if layer.crs is None:
        raise ValueError('Для выбора UTM должна быть задана исходная CRS')
    if not np.isfinite(layer.total_bounds).all():
        raise ValueError('Для выбора UTM нужны конечные координаты слоя')
    try:
        crs = layer.estimate_utm_crs()
    except (RuntimeError, ValueError) as error:
        raise ValueError('Не удалось подобрать UTM для расположения слоя') from error
    if crs is None:
        raise ValueError('Не удалось подобрать UTM для расположения слоя')
    return crs


def geo_geometry_error(layer):
    """Проверяет геометрию, область Web Mercator и координаты WGS 84."""
    if layer.empty:
        return None
    if (layer.geometry.isna().any() or layer.geometry.is_empty.any()
            or not layer.geometry.is_valid.all()
            or not np.isfinite(layer.geometry.get_coordinates().to_numpy()).all()):
        return 'Некорректная геометрия слоя'
    if layer.crs is None:
        return 'Не задана исходная CRS слоя'
    if layer.crs.to_epsg() == 3857:
        if (np.abs(layer.geometry.get_coordinates().to_numpy()) > 20037508.342789244).any():
            return 'Координаты выходят за область EPSG:3857'
    geographic = layer.to_crs(4326)
    coordinates = geographic.geometry.get_coordinates().to_numpy()
    if (not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:, 0]) > 180).any()
            or (np.abs(coordinates[:, 1]) > 90).any()
            or not geographic.geometry.is_valid.all()):
        return 'Некорректные координаты после преобразования в EPSG:4326'
    return None


def load_geo_csv(filename, encoding='utf-8', crs='epsg:4326', dtype=None, keep_default_na=True):
    """Загружает CSV с разделителем ``;`` и геометрией в поле WKT."""
    df = pd.read_csv(filename, delimiter=';', encoding=encoding, dtype=dtype,
                     keep_default_na=keep_default_na)
    if 'wkt' not in df.columns:
        raise ValueError(f'В географическом CSV отсутствует поле "wkt": {filename}')

    geometry = gpd.GeoSeries.from_wkt(df['wkt'], crs=crs)
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


GEO_TECHNICAL_FIELDS = frozenset({'wkt', 'geometry'})


def is_geo_technical_field(field_name):
    """Проверяет, является ли поле внутренним полем хранения геометрии."""
    return (isinstance(field_name, str)
            and field_name.strip().lower() in GEO_TECHNICAL_FIELDS)


def geo_user_fields(columns):
    """Возвращает только атрибуты слоя, которые можно показывать пользователю."""
    return [field for field in columns if not is_geo_technical_field(field)]


def load_geo_df(path, proj_name, prefix):
    # Загружает геоданные в GeoDataFrame и удаляет исходное поле WKT.
    name_geom = f'{prefix}_{proj_name}.csv'
    logger.info('load_geo_df: path=%s, filename=%s', path, name_geom)
    geo_df = load_geo_csv(path + name_geom)
    geo_df.drop(columns=['wkt'], inplace=True)
    logger.info('load_geo_df: loaded shape=%s', geo_df.shape)
    return geo_df

# max_los_speed, min_los_speed, df, geo_df, targets = prepare_geo_los_detailed('static/userdata/user/los/v11/output/', 'v11', 'tr_district', "id_tr_distr", True, None)
# print(geo_df)


def _geo_upload_error(gdf):
    if gdf.empty:
        return 'Rsim: загружаемый слой пуст'
    if 'id' not in gdf:
        return 'Rsim: отсутствует поле id'
    if gdf['id'].isna().any() or gdf['id'].astype(str).str.strip().eq('').any():
        return 'Rsim: поле id содержит пустые значения'
    if gdf['id'].astype(str).duplicated().any():
        return 'Rsim: обнаружены повторяющиеся id'
    geometry_error = geo_geometry_error(gdf)
    if geometry_error:
        return f'Rsim: {geometry_error}'
    return None


def check_geo_source(proj_path_src, filename, prefix_name):
    '''
    Проверям основные входные данные модуля проектирования на корректность
    '''
    logger.info('check_geo_source: path=%s, filename=%s, prefix_name=%s', proj_path_src, filename, prefix_name)
    net_limit = 10000
    minimum_requirement = {
    'stops': {'id': 'object', 'name': 'object'},
    'routes': {'id': 'object', 'name': 'object'},
    'districts': {'id': 'object'}
    }
    geom_requirement = {
    'stops': ['Point'],
    'routes': ['LineString', 'MultiLineString'],
    'districts': ['Polygon', 'MultiPolygon']
    }

    gdf = load_geo_csv(proj_path_src + filename, dtype={'id': str}, keep_default_na=False)
    error = _geo_upload_error(gdf)
    if error:
        return False, error, None
    # проверяем поля
    for i in minimum_requirement[prefix_name]:
        if i not in gdf.columns:
            logger.warning('check_geo_source[%s]: missing column %s', prefix_name, i)
            return False, f'Rsim: в таблице не обнаружено поле "{i}"', None
        # проверяем тип геометрии
    geom_type = set(gdf.geom_type.to_list())
    geom_type_value = list(geom_type)[0]
    all_gdf_fields = gdf.columns.to_list()
    gdf_fields = geo_user_fields(all_gdf_fields)
    if gdf.shape[0] > net_limit:
        logger.warning('check_geo_source[%s]: row limit exceeded, rows=%s', prefix_name, gdf.shape[0])
        return False, f'Rsim: необходимо уменьшить количество записей в таблице - не более {net_limit} строк', None
    if geom_type.issubset(geom_requirement[prefix_name]):
        if len(all_gdf_fields) == len(set(all_gdf_fields)):
            if 'id' in gdf_fields:
                logger.info('check_geo_source[%s]: validation passed, rows=%s, geom_type=%s', prefix_name, gdf.shape[0], geom_type_value)
                return True, 'Таблица успешно загружена', gdf_fields
            else:
                logger.warning('check_geo_source[%s]: missing id field', prefix_name)
                return False, f'Rsim: в таблице "{prefix_name}" отсутствует поле "id"', None
        else:
            logger.warning('check_geo_source[%s]: duplicate field names detected', prefix_name)
            return False, f'Rsim: требуется проверка названий полей таблицы, обнаружены повторы "{prefix_name}"', None
    else:
        logger.warning('check_geo_source[%s]: invalid geometry type=%s', prefix_name, geom_type_value)
        return False, f'Rsim: требуется проверка типа геометрии для слоя "{prefix_name}"', None


def check_geo_add_source(proj_path_src, filename, prefix_name):
    ''' 
    Проверям и формируем описание дополнительных входных данных модуля проектирования на корректность
    '''
    logger.info('check_geo_add_source: path=%s, filename=%s, prefix_name=%s', proj_path_src, filename, prefix_name)
    net_limit = 10000
    gdf = load_geo_csv(proj_path_src + filename, dtype={'id': str}, keep_default_na=False)
    error = _geo_upload_error(gdf)
    if error:
        return False, error, None, None
    # определяем тип геометрии
    geom_compare = {
    'Point': 'Point',
    'LineString': 'LineString',
    'MultiLineString': 'LineString',
    'Polygon': 'Polygon',
    'MultiPolygon': 'Polygon'
    }
    geom_type = set(gdf.geom_type.to_list())
    geom_type_value = list(geom_type)[0]
    gdf_fields = geo_user_fields(gdf.columns.to_list())
    if gdf.shape[0] > net_limit:
        logger.warning('check_geo_add_source[%s]: row limit exceeded, rows=%s', prefix_name, gdf.shape[0])
        return False, f'Rsim: необходимо уменьшить количество записей в таблице - не более {net_limit} строк', None, None
    if (geom_type.issubset(geom_compare)
            and len({geom_compare[value] for value in geom_type}) == 1):
        if 'id' in gdf_fields:
            logger.info('check_geo_add_source[%s]: validation passed, rows=%s, geom_type=%s', prefix_name, gdf.shape[0], geom_compare[geom_type_value])
            return True, 'Таблица дополнительного слоя успешно загружена', gdf_fields, geom_compare[geom_type_value]
        else:
            logger.warning('check_geo_add_source[%s]: missing id field', prefix_name)
            return False, f'Rsim: в таблице "{prefix_name}" отсутствует поле "id"', None, None
    else:
        logger.warning('check_geo_add_source[%s]: invalid geometry type=%s', prefix_name, geom_type_value)
        return False, f'Rsim: требуется проверка типа геометрии для слоя "{prefix_name}"', None, None


def sl_json(file_path, json_object, action_type):
    # Сохраняет JSON в файл или загружает JSON из файла
    logger.info('sl_json: file_path=%s, action_type=%s', file_path, action_type)
    if action_type == 'save':
        json_out = json.dumps(json_object)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_out)
        logger.info('sl_json: saved file %s', file_path)
    # load json
    if action_type == 'load':
        with open(file_path, encoding='utf-8') as json_in:
            json_out = json.load(json_in)
        logger.info('sl_json: loaded file %s', file_path)
        return json_out


def validate_access_parameters(access_radius_km, min_overlap_share):
    for name, value, lower, upper in (
        ('access_radius_km', access_radius_km, .3, 1.2),
        ('min_overlap_share', min_overlap_share, .15, .3),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value <= upper):
            raise ValueError(f'{name}: требуется число от {lower} до {upper}')


def stop_geodata(frame):
    try:
        geometry = gpd.GeoSeries.from_wkt(frame['wkt'], crs='EPSG:4326')
        layer = gpd.GeoDataFrame(frame[['id_stop']].copy(), geometry=geometry)
    except (KeyError, ValueError, TypeError, GEOSException) as error:
        raise ValueError('ref_stops: требуются id_stop и корректная WKT-геометрия') from error
    error = geo_geometry_error(layer)
    if error or not geometry.geom_type.isin({'Point', 'MultiPoint'}).all():
        raise ValueError(f'ref_stops: {error or "недопустимый тип геометрии"}')
    bounds = geometry.bounds
    if ((bounds[['minx', 'maxx']].abs() > 180).any().any()
            or (bounds[['miny', 'maxy']].abs() > 90).any().any()):
        raise ValueError('ref_stops: требуются координаты EPSG:4326')
    return layer


def build_stop_access(ref_districts, ref_stops, access_radius_km, min_overlap_share):
    """Keep every eligible district and the fraction of the stop buffer it covers."""
    validate_access_parameters(access_radius_km, min_overlap_share)
    try:
        geometry = gpd.GeoSeries.from_wkt(ref_districts['wkt'], crs='EPSG:4326')
        districts = gpd.GeoDataFrame(ref_districts[['id_tr_distr']].copy(), geometry=geometry)
    except (KeyError, ValueError, TypeError, GEOSException) as error:
        raise ValueError('ref_tr_distr: некорректная WKT-геометрия') from error
    error = geo_geometry_error(districts)
    if error or not geometry.geom_type.isin({'Polygon', 'MultiPolygon'}).all():
        raise ValueError(f'ref_tr_distr: {error or "недопустимый тип геометрии"}')
    stops = stop_geodata(ref_stops)
    columns = ['id_tr_distr', 'id_stop', 'overlap_share']
    if stops.empty or districts.empty:
        return pd.DataFrame(columns=columns)
    crs = select_utm_crs(stops)
    districts, stops = districts.to_crs(crs), stops.to_crs(crs)
    buffers = stops.geometry.buffer(access_radius_km * 1000)
    parts = []
    for district in districts.itertuples():
        share = buffers.intersection(district.geometry).area / buffers.area
        selected = share >= min_overlap_share
        if selected.any():
            parts.append(pd.DataFrame({'id_tr_distr': district.id_tr_distr,
                                       'id_stop': stops.loc[selected, 'id_stop'].to_numpy(),
                                       'overlap_share': share.loc[selected].to_numpy()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# Это сценарные ёмкости, а не коэффициенты генерации транспортных поездок.
RATIOS = np.array([1.0, .50, .13, .03, .01, .03, .45, .10])
CENTRAL = np.array([.275, .50, .035, .045, .015, .020, .075, .035])
RESIDENTIAL = np.array([.750, .125, .055, .005, .005, .015, .035, .010])
INDUSTRIAL = np.array([.060, .820, .005, .000, .005, .010, .080, .020])


def _integer_allocation(weights, total):
    """Метод наибольших остатков сохраняет целый городской итог точно."""
    exact = weights / weights.sum() * total
    result = np.floor(exact).astype(np.int64)
    remaining = int(total - result.sum())
    order = np.argsort(-(exact - result), kind='stable')
    result[order[:remaining]] += 1
    return result


def generate_characteristics(districts, population=255_000, seed=42):
    """Меняет только восемь характеристик, возвращает CSV и пространственный аудит.

    255 тысяч относятся к существующему base. Новые районы используют те же
    масштабы ёмкости, добавляя население. Для существующего forecast сохраняем
    относительные изменения. Профили нормированы до 100%, итоги — по столбцам.
    """
    from forecast_engine.config import FEATURE_COLUMNS
    from forecast_engine.inputs import _normalize_districts
    from forecast_engine.los import _district_geodata

    if not isinstance(population, int) or population <= 0:
        raise ValueError('Население должно быть положительным целым числом')
    _normalize_districts(districts, {})
    base = districts.loc[districts.state.eq('base')].sort_values('id_tr_distr')
    unique = districts.drop_duplicates('id_tr_distr').sort_values('id_tr_distr')
    _, crs = _district_geodata(base)
    geodata, _ = _district_geodata(unique)
    # Проекция определяется существующей территорией, как в основном ядре.
    geodata = geodata.to_crs(crs)
    geometry = geodata.geometry
    centers = geometry.centroid
    center = geometry.union_all().centroid
    distance = centers.distance(center).to_numpy() / 1000
    area = geometry.area.to_numpy() / 1e6
    existing = unique.is_new.eq(0).to_numpy()
    inner, outer = np.quantile(distance[existing], [.20, .90])
    if outer <= inner:
        raise ValueError('Недостаточный диапазон расстояний для радиального профиля')
    peripheral = np.clip((distance - inner) / (outer - inner), 0, 1)
    profiles = CENTRAL + peripheral[:, None] * (RESIDENTIAL - CENTRAL)
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(existing & (distance >= np.median(distance[existing])))
    industry_count = max(3, int(np.floor(existing.sum() * .05 + .5)))
    if len(candidates) < industry_count:
        raise ValueError('Недостаточно периферийных районов для минимум трёх промышленных')
    industry = rng.choice(candidates, size=industry_count, replace=False)
    profiles[industry] = INDUSTRIAL
    # Независимые относительные возмущения ±1–5%, затем сумма профиля равна 1.
    fluctuation = rng.choice([-1, 1], profiles.shape) * rng.uniform(.01, .05, profiles.shape)
    profiles *= 1 + fluctuation
    profiles /= profiles.sum(axis=1, keepdims=True)
    # Крупные незастроенные полигоны не должны забирать весь объём города.
    area_cap = 4 * np.median(area[existing])
    size = np.sqrt(np.minimum(area, area_cap))
    weights = size[:, None] * profiles
    # Крупные специализированные объекты сосредоточены в нескольких узлах.
    hub_ids = {}
    for feature, count in [('higher_education', 4), ('hospital', 6)]:
        column = FEATURE_COLUMNS.index(feature)
        eligible = np.flatnonzero(existing & (peripheral < .5))
        eligible = eligible[~np.isin(eligible, industry)]
        count = min(count, len(eligible))
        if count == 0:
            raise ValueError('Нет центральных районов для специализированных объектов')
        chosen = rng.choice(eligible, count, replace=False,
                            p=weights[eligible, column] / weights[eligible, column].sum())
        weights[:, column] = np.where(np.isin(np.arange(len(unique)), chosen),
                                      weights[:, column], 0)
        hub_ids[feature] = unique.iloc[chosen].id_tr_distr.tolist()
    totals = np.rint(population * RATIOS).astype(np.int64)
    values = np.zeros(weights.shape, dtype=np.int64)
    scales = totals / weights[existing].sum(axis=0)
    for column, total in enumerate(totals):
        values[existing, column] = _integer_allocation(weights[existing, column], total)
    values[~existing] = np.rint(weights[~existing] * scales).astype(np.int64)
    generated = pd.DataFrame(values, index=unique.id_tr_distr, columns=FEATURE_COLUMNS)
    result = districts.copy()
    result[FEATURE_COLUMNS] = generated.loc[result.id_tr_distr].to_numpy()
    original_base = base.set_index('id_tr_distr')[FEATURE_COLUMNS]
    forecast = districts.is_new.eq(0) & districts.state.eq('forecast')
    old_base = original_base.loc[districts.loc[forecast, 'id_tr_distr']].to_numpy()
    old_future = districts.loc[forecast, FEATURE_COLUMNS].to_numpy(dtype=float)
    if ((old_base == 0) & (old_future > 0)).any():
        raise ValueError('Нельзя перенести относительный рост от нулевого base')
    ratio = np.divide(old_future, old_base, out=np.ones_like(old_future), where=old_base > 0)
    result.loc[forecast, FEATURE_COLUMNS] = np.rint(
        result.loc[forecast, FEATURE_COLUMNS].to_numpy() * ratio).astype(np.int64)
    types = np.where(distance <= inner, 'central', 'residential')
    types[industry] = 'industrial'
    audit = unique[['id_tr_distr', 'is_new']].reset_index(drop=True).copy()
    audit['district_type'] = types
    audit['distance_km'] = distance
    audit['area_km2'] = area
    audit['effective_area_km2'] = np.minimum(area, area_cap)
    audit['peripheral_index'] = peripheral
    for column, feature in enumerate(FEATURE_COLUMNS):
        audit[f'profile_{feature}'] = profiles[:, column]
        audit[f'fluctuation_{feature}'] = fluctuation[:, column]
        audit[feature] = values[:, column]
    summary = {
        'seed': seed, 'metric_crs': crs,
        'center_metric': [center.x, center.y],
        'inner_distance_km': inner, 'outer_distance_km': outer,
        'area_cap_km2': area_cap,
        'existing_districts': int(existing.sum()),
        'industrial_ids': unique.iloc[industry].id_tr_distr.tolist(),
        'specialized_hubs': hub_ids,
        'base_totals': dict(zip(FEATURE_COLUMNS, totals.tolist())),
        'new_totals': dict(zip(FEATURE_COLUMNS, values[~existing].sum(axis=0).tolist())),
        'existing_forecast_policy': 'relative_changes',
    }
    return result, audit, summary


def generate_characteristics_cli():
    """Генерирует демонстрационные характеристики районов из командной строки."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Генерация характеристик демонстрационного города")
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--population', type=int, default=255_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.input, sep=';')
    result, audit, summary = generate_characteristics(
        data, args.population, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep=';', index=False, lineterminator='\n')
    audit.to_csv(args.audit_dir / 'district_profiles.csv', sep=';', index=False)
    (args.audit_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    generate_characteristics_cli()
