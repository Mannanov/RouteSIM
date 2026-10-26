"""Расчеты и редактирование геоданных проекта GEO.
CSV-слои хранят геометрию WKT в EPSG:4326, редактор карты передает
GeoJSON в EPSG:3857, а расстояния и пространственные расчеты используют
локальную UTM.
"""

import json
from serv_funcs import (load_geo_csv, is_geo_technical_field, select_utm_crs,
                        geo_geometry_error)
import numpy as np
from rsim import app
import geopandas as gpd
import pandas as pd
import osmnx as ox
import shapely
from shapely.ops import substring
import sklearn.cluster as skc
from scipy.spatial.distance import cdist
from collections import deque
from heapq import heappop, heappush
import os
import tempfile
import hashlib
import sqlite3
import threading
import requests
from pathlib import Path
from pyproj.exceptions import CRSError
from functools import wraps


ox.settings.cache_folder = Path(__file__).resolve().parent / 'cache'
ox.settings.log_console = True

OVERPASS_ALTERNATIVES = (
    'https://maps.mail.ru/osm/tools/overpass/api',
    'https://overpass.private.coffee/api',
)
_OVERPASS_LOCK = threading.Lock()


class _OverpassUnavailable(RuntimeError):
    """Сервер Overpass недоступен или вернул неполные данные."""


def _check_overpass_response(response, **kwargs):
    # Hook вызывается до встроенных рекурсивных повторов OSMnx на 429/504.
    endpoint = ox.settings.overpass_url.rstrip('/') + '/interpreter'
    if response.url.split('?', 1)[0] != endpoint:
        return response
    try:
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise _OverpassUnavailable(f'HTTP {response.status_code}')
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as error:
            raise _OverpassUnavailable('Ответ Overpass не является JSON') from error
        if (not isinstance(data, dict) or not isinstance(data.get('elements'), list)
                or data.get('remark')):
            raise _OverpassUnavailable('Overpass вернул неполные данные')
    except Exception:
        response.close()
        raise
    return response


def _download_osm_graph(area_name):
    """Пробует каждый Overpass один раз, сохраняя глобальные настройки OSMnx."""
    # Настройки OSMnx общие для потоков: загрузки графов выполняются по очереди.
    with _OVERPASS_LOCK:
        original_url = ox.settings.overpass_url
        original_kwargs = ox.settings.requests_kwargs
        hooks = dict(original_kwargs.get('hooks') or {})
        response_hooks = hooks.get('response') or []
        if callable(response_hooks):
            response_hooks = [response_hooks]
        hooks['response'] = [_check_overpass_response, *response_hooks]
        servers = dict.fromkeys((original_url.rstrip('/'), *OVERPASS_ALTERNATIVES))
        try:
            ox.settings.requests_kwargs = {**original_kwargs, 'hooks': hooks}
            for server in servers:
                ox.settings.overpass_url = server
                app.logger.info('Загрузка графа: Overpass %s', server)
                try:
                    return ox.graph_from_place(area_name, network_type='drive')
                except (_OverpassUnavailable, requests.RequestException) as error:
                    if isinstance(error, requests.RequestException):
                        request = error.request
                        # Сбой Nominatim или ошибка запроса не исправятся сменой Overpass.
                        if request is None or not request.url.startswith(server + '/'):
                            raise
                        if isinstance(error, requests.HTTPError):
                            status = error.response.status_code if error.response is not None else None
                            if status not in (408, 429) and (status is None or status < 500):
                                raise
                    app.logger.warning('Overpass %s недоступен: %s', server, error)
                    last_error = error
            raise _OverpassUnavailable('Все серверы Overpass недоступны') from last_error
        finally:
            ox.settings.overpass_url = original_url
            ox.settings.requests_kwargs = original_kwargs

# Менять версию при изменении проекции, схемы хранения или подготовки ребер.
ROUTE_GRAPH_PREPARATION_VERSION = 2
PREPARED_ROUTE_GRAPH_FILENAME = 'g_prepared.sqlite'


class GeoValidationError(ValueError):
    """Ожидаемая ошибка данных с безопасным сообщением для пользователя."""


def _positive_distance(value, label):
    """Возвращает положительный конечный float из числа или числовой строки.

    bool, ноль, отрицательные и нечисловые значения дают GeoValidationError;
    label используется в сообщении, пригодном для показа пользователю.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise GeoValidationError(f'{label} должен быть положительным числом') from None
    if isinstance(value, bool) or not np.isfinite(number) or number <= 0:
        raise GeoValidationError(f'{label} должен быть положительным числом')
    return number


def _validate_id_column(frame, column, layer_name, unique=False):
    """Проверяет ID до строкового преобразования и возвращает строковую Series."""
    if column not in frame.columns:
        raise GeoValidationError(f'В слое {layer_name} отсутствует поле {column}')
    values = frame[column]
    if values.isna().any():
        raise GeoValidationError(f'Поле {column} слоя {layer_name} содержит пустые значения')
    identifiers = values.astype(str)
    if identifiers.str.strip().eq('').any():
        raise GeoValidationError(f'Поле {column} слоя {layer_name} содержит пустые значения')
    if unique and identifiers.duplicated().any():
        raise GeoValidationError(f'Слой {layer_name} содержит повторяющиеся {column}')
    return identifiers


def _load_required_geo_layer(filename, label):
    """Читает обязательный слой, превращая отсутствие файла в ошибку данных."""
    try:
        return _load_geo_layer(filename)
    except FileNotFoundError as error:
        raise GeoValidationError(f'Не найден {label}') from error


def _validate_layer_name(layer):
    """Проверяет имя слоя для включения в имя файла, не изменяя строку.

    Разрешены буквы, цифры, '_', '-', точки и пробелы старых реестров.
    Пустые имена, '.'/'..' и разделители пути дают GeoValidationError.
    """
    if (not isinstance(layer, str) or not layer.strip() or layer in ('.', '..')
            or not all(c.isalnum() or c in '_-. ' for c in layer)):
        raise GeoValidationError('Некорректное имя слоя')


def _metric_frame(frame, metric_crs=None):
    """Возвращает геоданные в локальной UTM, выбранной по охвату слоя.

    Для слоя без CRS предполагается EPSG:4326. Исходный объект не изменяется;
    невозможность подобрать UTM преобразуется в GeoValidationError.
    metric_crs — уже выбранная CRS текущего конвейера. Без неё UTM
    подбирается по слою, а не берется из настроек прошлого расчета.
    """
    if frame.crs is None:
        frame = frame.set_crs(4326)
    try:
        target = metric_crs if metric_crs is not None else select_utm_crs(frame)
    except ValueError as error:
        raise GeoValidationError(str(error)) from error
    return frame if frame.crs == target else frame.to_crs(target)

def _load_geo_layer(filename, encoding='utf-8'):
    """Читает CSV с разделителем ';' и WKT в EPSG:4326 в GeoDataFrame.

    id, id_grp_stp и id_cluster читаются строками, включая ведущие нули
    и литералы NA/NULL. Проверяются геометрия и координаты; исходное поле
    wkt остается рядом с активной geometry. Ошибки данных преобразуются
    в GeoValidationError, ошибки доступа к файлу передаются вызывающему коду.
    """
    try:
        frame = load_geo_csv(filename, encoding=encoding, keep_default_na=False, dtype={
            'id': str, 'id_grp_stp': str, 'id_cluster': str,
        })
        error = geo_geometry_error(frame)
        if error:
            raise GeoValidationError(error)
        return frame
    except GeoValidationError:
        raise
    except (ValueError, KeyError, shapely.errors.ShapelyError) as error:
        raise GeoValidationError('Данные географического слоя повреждены') from error


def _load_route_elements(filename):
    """Читает CSV сегментов маршрутов с полями rnd/source/target/length.

    Идентификаторы сохраняются строками, включая ведущие нули и NA/NULL.
    Проверяются наличие полей и разбор CSV; значения length дополнительно
    не валидируются. Возвращается DataFrame в порядке строк файла.
    """
    try:
        elements = pd.read_csv(filename, sep=';', keep_default_na=False,
                               dtype={'rnd': str, 'source': str, 'target': str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise GeoValidationError('Файл сегментов маршрутов поврежден') from error
    if not {'rnd', 'source', 'target', 'length'}.issubset(elements.columns):
        raise GeoValidationError('В файле сегментов маршрутов отсутствуют обязательные поля')
    return elements


def _remove_geo_technical_properties(obj_data):
    """Удаляет wkt/geometry из properties GeoJSON, изменяя obj_data на месте.

    Поле geometry самого Feature сохраняется. Неподходящие по структуре
    элементы пропускаются; проверку запроса выполняет _validate_geo_payload.
    """
    if not isinstance(obj_data, dict):
        return
    for feature in obj_data.get('features', []):
        if not isinstance(feature, dict):
            continue
        properties = feature.get('properties')
        if not isinstance(properties, dict):
            continue
        for field_name in list(properties):
            if is_geo_technical_field(field_name):
                del properties[field_name]


def _concat_geo_frames(frames):
    """Объединяет непустой список GeoDataFrame с общими CRS и именем геометрии.

    Возвращает новый GeoDataFrame со сквозным индексом. CRS автоматически
    не преобразуются: несовместимые слои дают GeoValidationError,
    объекты другого типа — TypeError.
    """
    frames = list(frames)
    if not frames:
        raise GeoValidationError('Не переданы геоданные для объединения')
    if not all(isinstance(frame, gpd.GeoDataFrame) for frame in frames):
        raise TypeError('Для объединения ожидается GeoDataFrame')

    geometry_name = frames[0].geometry.name
    crs = frames[0].crs
    for frame in frames:
        if frame.geometry.name != geometry_name:
            raise GeoValidationError('Объединяемые слои используют разные поля геометрии')
        if frame.crs != crs:
            raise GeoValidationError('Объединяемые слои используют разные CRS')

    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(
        combined,
        geometry=geometry_name,
        crs=crs,
    )


def _build_route_feature(route_line, obj_data, crs='epsg:3857'):
    """Создает однообъектный GeoDataFrame маршрута в EPSG:4326.

    route_line задан в crs. Атрибуты берутся из obj_data, кроме списка
    elements и служебных полей геометрии; id преобразуется в строку.
    Геометрия результата проверяется до последующего сохранения.
    """
    route = gpd.GeoDataFrame(
        index=[0],
        crs=crs,
        geometry=[route_line],
    ).to_crs('epsg:4326')
    error = geo_geometry_error(route)
    if error:
        raise GeoValidationError(error)
    for field_name, value in obj_data.items():
        if field_name != 'elements' and not is_geo_technical_field(field_name):
            route[field_name] = str(value) if field_name == 'id' else value
    return route


def _replace_active_geometry(frame, new_geometry_name):
    """Делает существующий столбец новой геометрией под прежним активным именем.

    Возвращает новый GeoDataFrame; при смене столбца прежняя геометрия
    удаляется. Помощник позволяет заменить объединенные объекты их
    центроидами без смены схемы.
    """
    old_geometry_name = frame.active_geometry_name
    if old_geometry_name == new_geometry_name:
        return frame.set_geometry(new_geometry_name)

    return (
        frame.set_geometry(new_geometry_name)
        .drop(columns=old_geometry_name)
        .rename_geometry(old_geometry_name)
    )


def _write_geo_files(writes, deletes=()):
    """Формирует временные файлы, заменяет целевые и выполняет удаления.

    writes — последовательность (data, target, csv_options): DataFrame
    записывается в CSV, dict — в JSON UTF-8, callable получает временный путь.
    deletes содержит пути удаляемых файлов. Каталоги должны существовать;
    один и тот же путь нельзя указать дважды в комплекте.

    Сбой формирования временных файлов сохраняет прежние целевые файлы.
    Замены атомарны по отдельности, но сбой установки может оставить
    частично обновленный комплект: отката нет. Замок держит вызывающая
    операция проекта; этот низкоуровневый помощник сам его не берёт. Временные
    файлы очищаются в finally; исключения передаются вызывающему коду.
    """
    writes = list(writes)
    deletes = [os.fspath(target) for target in deletes]
    staged_files = []
    target_paths = [os.fspath(target) for _, target, _ in writes]
    all_targets = target_paths + deletes
    if len(all_targets) != len(set(all_targets)):
        raise GeoValidationError(
            'Один файл указан в комплекте записи или удаления несколько раз'
        )

    try:
        # Сначала полностью формируем все новые файлы рядом с целевыми.
        for frame, target, csv_options in writes:
            target = os.fspath(target)
            target_dir = os.path.dirname(target) or '.'
            fd, temp_path = tempfile.mkstemp(
                prefix=f'.{os.path.basename(target)}.',
                suffix='.tmp',
                dir=target_dir,
            )
            os.close(fd)
            staged_files.append((temp_path, target))
            if callable(frame):
                frame(temp_path)
            elif isinstance(frame, dict):
                with open(temp_path, 'w', encoding='utf-8') as stream:
                    json.dump(frame, stream, ensure_ascii=False, indent=2)
            else:
                frame.to_csv(temp_path, **csv_options)

        # Устанавливаем новые версии; удаления выполняем после всех замен.
        for temp_path, target in staged_files:
            os.replace(temp_path, target)
        for target in deletes:
            if os.path.exists(target):
                os.remove(target)
    finally:
        for temp_path, _ in staged_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def _normalise_geometry_name(frame):
    """Возвращает копию GeoDataFrame с активным столбцом geometry.

    Если активная геометрия называлась иначе, одноименный неактивный
    столбец geometry удаляется перед переименованием.
    """
    frame = frame.copy()
    geometry_name = frame.active_geometry_name
    if geometry_name != 'geometry':
        if 'geometry' in frame.columns:
            frame = frame.drop(columns='geometry')
        frame = frame.rename_geometry('geometry')
    return frame


def _prepare_cluster_frame(cluster_frame, metric_crs=None):
    """Проверяет назначения остановок и возвращает копию слоя в локальной UTM.

    Нужны непустые Point и поля id/id_grp_stp/id_cluster. ID остановок
    уникальны; каждая группа принадлежит одному аналитическому кластеру.
    Все ID приводятся к строкам после проверки пропусков, поле wkt удаляется.
    Отсутствующая CRS считается EPSG:4326. Соответствие исходному stops
    здесь не проверяется; ошибки проверок дают GeoValidationError.
    metric_crs позволяет использовать уже выбранную CRS текущего расчёта.
    """
    if not isinstance(cluster_frame, gpd.GeoDataFrame):
        raise GeoValidationError('Слой кластеров должен быть GeoDataFrame')
    cluster_frame = _normalise_geometry_name(cluster_frame)
    cluster_frame.drop(columns=['wkt'], errors='ignore', inplace=True)
    required_columns = {'id', 'id_grp_stp', 'id_cluster'}
    missing_columns = required_columns.difference(cluster_frame.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise GeoValidationError(f'В слое кластеров отсутствуют поля: {missing}')
    for column in sorted(required_columns):
        cluster_frame[column] = _validate_id_column(
            cluster_frame, column, 'кластеров', unique=column == 'id')
    if cluster_frame.groupby('id_grp_stp')['id_cluster'].nunique().gt(1).any():
        raise GeoValidationError('Каждая группа остановок должна принадлежать одному аналитическому кластеру')
    if cluster_frame.crs is None:
        cluster_frame = cluster_frame.set_crs('epsg:4326')
    if cluster_frame.empty or not cluster_frame.geometry.geom_type.eq('Point').all():
        raise GeoValidationError('Для расчета нужен непустой слой точечных кластеров')
    error = geo_geometry_error(cluster_frame)
    if error:
        raise GeoValidationError(error)
    return _metric_frame(cluster_frame, metric_crs)


def _build_geom_clusters(cluster_frame):
    """Возвращает DataFrame центроидов с индексом id_cluster и геометрией wkt.

    cluster_frame должен быть подготовлен в метрической CRS. Геометрии
    объединяются по id_cluster, их центроиды переводятся в EPSG:4326.
    Справочник содержит только id_cluster и wkt; файл здесь не записывается.
    """
    geom_clusters = cluster_frame[['id_cluster', 'geometry']].dissolve(by='id_cluster')
    geom_clusters['sgc'] = geom_clusters.centroid
    geom_clusters = _replace_active_geometry(geom_clusters, 'sgc')
    geom_clusters = geom_clusters.to_crs('epsg:4326')
    geom_clusters = geom_clusters.rename_geometry('wkt')
    return pd.DataFrame(geom_clusters)


def _stop_group_centers(cluster_frame):
    """Возвращает GeoSeries центроидов объединенных точек по id_grp_stp.

    Индекс содержит ID групп, CRS сохраняется. Для расстояний вызывающий
    код должен передавать слой в метрической проекции.
    """
    stop_groups = cluster_frame.dissolve(by='id_grp_stp')
    return stop_groups.centroid


def _iter_stop_distance_matrix(centers, k_walk, max_cells=250000):
    """Выдает DataFrame-блоки всех направленных пар групп без диагонали.

    centers — GeoSeries точек в метрической CRS с ID групп в индексе.
    distance = прямое расстояние * k_walk / 1000, округленное до 0,001 км;
    поля результата: link ('source>target'), source, target, distance.
    max_cells задает целевой размер блока вычислений; хотя бы одна строка
    центров обрабатывается за шаг. Итоговый объем остается n*(n-1) строк.
    """
    group_ids = centers.index.to_numpy()
    coordinates = np.column_stack((centers.x, centers.y))
    count = len(group_ids)
    # Вычисляем полосы матрицы: все столбцы при ограниченном числе источников.
    block_size = max(1, min(256, max_cells // max(1, count)))
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        distances = cdist(coordinates[start:end], coordinates, metric='euclidean')
        non_diagonal = np.arange(start, end)[:, None] != np.arange(count)[None, :]
        result = pd.DataFrame({
            'source': np.broadcast_to(group_ids[start:end, None], distances.shape)[non_diagonal],
            'target': np.broadcast_to(group_ids[None, :], distances.shape)[non_diagonal],
            'distance': np.round(k_walk * distances[non_diagonal] / 1000, 3),
        })
        result['link'] = result['source'].astype(str) + '>' + result['target'].astype(str)
        yield result[['link', 'source', 'target', 'distance']]


def _stop_distance_matrix_writer(cluster_frame, k_walk, metric_crs=None):
    """Возвращает callable(filename) для блочной записи матрицы групп в CSV.

    Проверяет k_walk и заранее вычисляет центры групп в локальной UTM.
    metric_crs задаёт уже выбранную проекцию, исключая повторный подбор UTM.
    Возвращаемая функция пишет заголовок и все направленные пары в UTF-8
    с разделителем ';'. Она совместима с заданиями _write_geo_files;
    полный DataFrame размера n² не создается.
    """
    k_walk = _positive_distance(k_walk, 'Коэффициент непрямолинейности')
    centers = _stop_group_centers(_metric_frame(cluster_frame, metric_crs))
    app.logger.info('Матрица расстояний: групп=%s, направленных пар=%s',
                    len(centers), len(centers) * (len(centers) - 1))
    def write(filename):
        """Записывает подготовленные центры блоками; для одной группы — заголовок."""
        with open(filename, 'w', encoding='utf-8', newline='') as stream:
            stream.write('link;source;target;distance\n')
            for block in _iter_stop_distance_matrix(centers, k_walk):
                block.to_csv(stream, sep=';', index=False, header=False)
    return write


def _build_ref_stops(cluster_frame, district_frame, stop_frame=None):
    """Возвращает справочник id_stop/id_grp_stp/cluster/tr_district.

    База — stop_frame, по умолчанию cluster_frame. Все исходные остановки
    сохраняются; отсутствующее назначение кластера или района остается пустым.
    Слои приводятся к CRS cluster_frame, отсутствующая CRS считается 4326.
    При нескольких пересеченных районах выбирается первый по строковой
    сортировке ID района. Идентификаторы сопоставляются как строки.
    """
    if stop_frame is None:
        stop_frame = cluster_frame
    stop_frame = _normalise_geometry_name(stop_frame)
    if stop_frame.crs is None:
        stop_frame = stop_frame.set_crs('epsg:4326')
    stop_frame = stop_frame.to_crs(cluster_frame.crs)
    stop_ids = _validate_id_column(stop_frame, 'id', 'остановок', unique=True)

    district_frame = _normalise_geometry_name(district_frame)
    if district_frame.crs is None:
        district_frame = district_frame.set_crs('epsg:4326')
    district_frame = district_frame.to_crs(cluster_frame.crs)
    if 'id' not in district_frame.columns:
        raise GeoValidationError('В слое транспортных районов отсутствует поле id')
    district_frame = district_frame[['id', district_frame.geometry.name]].rename(
        columns={'id': 'tr_district'})
    stop_points = stop_frame[['id', stop_frame.geometry.name]].copy()
    stop_points['_id_key'] = stop_ids
    stops_with_districts = stop_points.overlay(
        district_frame,
        how='intersection',
        keep_geom_type=True,
    )
    # Точка на общей границе или в перекрывающихся полигонах может попасть
    # сразу в несколько районов. Справочник обязан содержать одну строку на ОП.
    district_by_stop = pd.DataFrame(stops_with_districts)[
        ['_id_key', 'tr_district']
    ]
    district_by_stop['_district_sort'] = district_by_stop['tr_district'].astype(str)
    district_by_stop = (
        district_by_stop.sort_values(['_id_key', '_district_sort'], kind='stable')
        .drop_duplicates(subset='_id_key', keep='first')
        .drop(columns='_district_sort')
    )

    cluster_attributes = pd.DataFrame(cluster_frame)[
        ['id', 'id_grp_stp', 'id_cluster']
    ].copy()
    cluster_attributes['_id_key'] = cluster_attributes['id'].astype(str)
    cluster_attributes = cluster_attributes.drop(columns='id')

    # Базой результата является исходный слой остановок. Поэтому отсутствие
    # кластера или пространственного соответствия не удаляет id_stop.
    result = pd.DataFrame({
        'id_stop': stop_frame['id'].to_numpy(),
        '_id_key': stop_ids.to_numpy(),
    })
    result = result.merge(cluster_attributes, how='left', on='_id_key')
    result = result.merge(district_by_stop, how='left', on='_id_key')
    result = result.rename(columns={'id_cluster': 'cluster'})
    return result[
        ['id_stop', 'id_grp_stp', 'cluster', 'tr_district']
    ].reset_index(drop=True)


def _read_calculation_settings(path_src):
    """Читает объект geo_calculation_settings.json в UTF-8 или возвращает {}.

    Отсутствие файла допустимо для старого проекта. Ошибка синтаксиса JSON
    или другой тип верхнего уровня дает GeoValidationError; вложенная
    схема настроек здесь не проверяется.
    """
    filename = os.path.join(path_src, 'geo_calculation_settings.json')
    if not os.path.isfile(filename):
        return {}
    try:
        with open(filename, encoding='utf-8') as stream:
            settings = json.load(stream)
    except json.JSONDecodeError as error:
        raise GeoValidationError('Файл настроек геопроекта поврежден') from error
    if not isinstance(settings, dict):
        raise GeoValidationError('Настройки геопроекта должны быть объектом JSON')
    voronoi = settings.get('voronoi')
    if isinstance(voronoi, dict):
        # Старый выбор внешнего слоя больше не участвует в генерации районов.
        voronoi.pop('boundary_layer', None)
    return settings


def _polygonal_geometry(geometry):
    """Возвращает полигональную часть геометрии, рекурсивно объединяя коллекции.

    Линии, точки и пустой результат заменяются пустым Polygon. Используется
    после отсечения ячейки Вороного, которое может создать GeometryCollection.
    """
    if geometry.is_empty:
        return shapely.Polygon()
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry
    if geometry.geom_type == 'GeometryCollection':
        parts = [_polygonal_geometry(part) for part in geometry.geoms]
        return shapely.union_all([part for part in parts if not part.is_empty])
    return shapely.Polygon()


def _build_voronoi_districts(stops, radius_m, buffer_m=100.0):
    """Возвращает (districts, border, diagnostics) для районов Вороного.

    stops — непустой слой Point с уникальными id в метрической CRS.
    Центры получают из групп DBSCAN с порогом radius_m в метрах.
    Граница — выпуклая оболочка всех исходных остановок с буфером
    buffer_m метров. Внешние полигональные слои не используются.

    Районы отсекаются границей и получают последовательные ID в пределах
    расчета. Проверяются покрытие, перекрытия и назначения внутри границы.
    diagnostics содержит outside_boundary, unassigned_inside, seed_count
    и district_count. Ошибки геометрии/параметров дают GeoValidationError.
    """
    if stops.empty:
        raise GeoValidationError('Для генерации районов требуется хотя бы одна остановка')
    radius_m = _positive_distance(radius_m, 'Порог группировки районов')
    try:
        buffer_m = float(buffer_m)
    except (TypeError, ValueError):
        raise GeoValidationError('Буфер границы должен быть неотрицательным числом') from None
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise GeoValidationError('Буфер границы должен быть неотрицательным числом')
    _validate_id_column(stops, 'id', 'остановок', unique=True)
    if (stops.geometry.is_empty.any() or not stops.geometry.is_valid.all()
            or not stops.geom_type.eq('Point').all()):
        raise GeoValidationError('Для генерации районов нужны корректные точки остановок')
    coordinates = np.column_stack((stops.geometry.x, stops.geometry.y))
    labels = skc.DBSCAN(eps=radius_m, min_samples=1).fit_predict(coordinates)
    # Собираем индексы групп за один проход, не сканируя все метки для каждой.
    group_indices = {}
    for position, label in enumerate(labels):
        group_indices.setdefault(label, []).append(position)
    # Совпадающие после округления до микрометра центры задают одну ячейку.
    seeds = set()
    for indices in group_indices.values():
        group = stops.iloc[indices]
        centre = shapely.union_all(group.geometry.to_numpy()).centroid
        key = (round(centre.x, 6), round(centre.y, 6))
        seeds.add(key)
    points = [shapely.Point(coordinate) for coordinate in sorted(seeds)]

    border = shapely.union_all(stops.geometry.to_numpy()).convex_hull.buffer(buffer_m)
    if border.is_empty or border.area <= 0:
        raise GeoValidationError('Для одной остановки или остановок на линии нужен положительный буфер границы')
    if border.is_empty or not border.is_valid or border.area <= 0:
        raise GeoValidationError('Граница территории пуста или некорректна')

    if len(points) == 1:
        cells = [border]
    else:
        # extend_to расширяет до envelope; реальная отсечка всегда явная.
        diagram = shapely.voronoi_polygons(shapely.MultiPoint(points), extend_to=border)
        cells = list(diagram.geoms)
    assigned = {}
    # Порядок ячеек GEOS не задает ID районов: связываем ячейки с их центрами.
    tree = shapely.STRtree(points)
    for cell in cells:
        if len(points) == 1:
            seed_index = 0
        else:
            indices = tree.query(cell, predicate='covers')
            if len(indices) != 1:
                raise GeoValidationError('Не удалось однозначно связать ячейку с центром района')
            seed_index = int(indices[0])
        clipped = _polygonal_geometry(cell.intersection(border))
        if not clipped.is_empty and clipped.area > 0:
            assigned[seed_index] = clipped
    district_geometries = [assigned[index] for index in sorted(assigned)]
    districts = gpd.GeoDataFrame({'id': range(1, len(district_geometries) + 1)},
                                geometry=district_geometries, crs=stops.crs)
    coverage = shapely.union_all(districts.geometry.to_numpy())
    # Допуск учитывает масштаб границы и погрешность геометрических операций.
    tolerance = max(1e-6, border.area * 1e-9)
    if border.difference(coverage).area > tolerance:
        raise GeoValidationError('Сгенерированные районы не покрывают границу территории')
    if districts.geometry.area.sum() - coverage.area > tolerance:
        raise GeoValidationError('Сгенерированные районы перекрываются')
    inside = np.asarray(shapely.covers(border, stops.geometry.to_numpy()), dtype=bool)
    covered = np.asarray(shapely.covers(coverage, stops.geometry.to_numpy()), dtype=bool)
    diagnostics = {'outside_boundary': int((~inside).sum()),
                   'unassigned_inside': int((inside & ~covered).sum()),
                   'seed_count': len(points), 'district_count': len(districts)}
    if diagnostics['unassigned_inside']:
        raise GeoValidationError('Остановки внутри границы остались без района')
    return districts, border, diagnostics


def save_geo_calculation_settings(path_src, k_walk, district_layer):
    """Обновляет k_walk и активный слой районов в JSON, сохраняя прочие настройки.

    district_layer — None, 'districts' или 'districtsGEN'; k_walk должен быть
    положительным конечным числом. Один JSON заменяется атомарно.
    Функция не пересчитывает и не удаляет зависимые CSV; возвращает None.
    """
    k_walk = _positive_distance(k_walk, 'Коэффициент непрямолинейности')
    if district_layer not in (None, 'districts', 'districtsGEN'):
        raise GeoValidationError('Некорректный слой районов в настройках геопроекта')
    settings_path = os.path.join(path_src, 'geo_calculation_settings.json')
    settings = _read_calculation_settings(path_src)
    settings.update(k_walk=k_walk, district_layer=district_layer, distance_unit='km')
    # До первого расчёта конкретная CRS неизвестна; старый маркер нормализуем.
    if settings.get('metric_crs') == 'local_utm' or 'metric_crs' not in settings:
        settings['metric_crs'] = None
    _write_geo_files([(settings, settings_path, {})])


def _load_geo_calculation_settings(path_src, proj_name, settings=None):
    """Возвращает проверенную пару (k_walk, district_layer) без записи настроек.

    Использует переданный dict settings либо читает JSON. По умолчанию
    k_walk=1.15. Для старого проекта без ключа district_layer ищет сначала
    districts, затем districtsGEN; явный None означает отсутствие выбора
    и не заменяется найденным файлом.
    """
    if settings is None:
        settings = _read_calculation_settings(path_src)

    k_walk = _positive_distance(settings.get('k_walk', 1.15), 'Коэффициент непрямолинейности')
    district_layer = settings.get('district_layer')
    if district_layer not in (None, 'districts', 'districtsGEN'):
        raise GeoValidationError('Некорректный слой районов в настройках геопроекта')
    if 'district_layer' not in settings:
        for candidate in ('districts', 'districtsGEN'):
            if os.path.isfile(os.path.join(path_src, f'{candidate}_{proj_name}.csv')):
                district_layer = candidate
                break
    return k_walk, district_layer


def _cluster_bundle_writes(
        proj_name, path_src, path_out, cluster_output, cluster_target,
        stop_target):
    """Возвращает (writes, deletes) для сохранения кластеров и их производных.

    cluster_output — измененный слой для cluster_target; stop_target указывает
    исходный слой остановок. Проверяет назначения, читает расчетные настройки
    и готовит центры, матрицу и ref_stops в path_out. Неизменные центры групп
    позволяют сохранить прежнюю матрицу; неизменная структура и наличие
    производных позволяют записать только слой. Без активных районов
    справочник остановок ставится на удаление. Файлы здесь не меняются.
    """
    cluster_output = cluster_output.copy()
    cluster_frame = _prepare_cluster_frame(cluster_output)
    for column in ('id', 'id_grp_stp', 'id_cluster'):
        cluster_output[column] = cluster_frame[column].to_numpy()
    stop_frame = _load_required_geo_layer(stop_target, 'слой остановок')
    stop_ids = _validate_id_column(stop_frame, 'id', 'остановок', unique=True)
    if not cluster_frame['id'].isin(stop_ids).all():
        raise GeoValidationError('ID кластерных точек отсутствуют в исходных остановках')
    k_walk, district_layer = _load_geo_calculation_settings(path_src, proj_name)
    district_frame = None
    if district_layer is not None:
        district_frame = _load_required_geo_layer(
            os.path.join(path_src, f'{district_layer}_{proj_name}.csv'),
            f'слой транспортных районов: {district_layer}')
    matrix_path = os.path.join(path_out, f'matrix_dist_stops_{proj_name}.csv')
    centers_path = os.path.join(path_out, f'geom_clusters_{proj_name}.csv')
    ref_stops_path = os.path.join(path_out, f'ref_stops_{proj_name}.csv')
    previous = None
    same_projection = False
    if os.path.isfile(cluster_target):
        previous = _prepare_cluster_frame(_load_geo_layer(cluster_target))
        same_projection = previous.crs == cluster_frame.crs
        previous = previous.to_crs(cluster_frame.crs)
    matrix_unchanged = False
    if previous is not None:
        # Матрица зависит от центров групп и CRS, а не от аналитических ID.
        old_centers = _stop_group_centers(previous)
        new_centers = _stop_group_centers(cluster_frame)
        matrix_unchanged = (same_projection and old_centers.index.equals(new_centers.index)
                            and old_centers.geom_equals_exact(new_centers, tolerance=1e-6).all())
        attributes = ['id', 'id_grp_stp', 'id_cluster']
        old_sorted = previous.sort_values('id').reset_index(drop=True)
        new_sorted = cluster_frame.sort_values('id').reset_index(drop=True)
        structure_unchanged = (old_sorted[attributes].equals(new_sorted[attributes])
            and old_sorted.geometry.geom_equals_exact(new_sorted.geometry, tolerance=1e-6).all())
        if (structure_unchanged and os.path.isfile(matrix_path)
                and os.path.isfile(centers_path)
                and list(pd.read_csv(centers_path, sep=';', nrows=0).columns) == ['id_cluster', 'wkt']
                and (district_layer is None or os.path.isfile(ref_stops_path))):
            # Правка подписи не требует пересчета уже существующих производных.
            return [(cluster_output, cluster_target,
                     {'sep': ';', 'index': False, 'encoding': 'utf-8'})], []
    writes = [
        (
            cluster_output,
            cluster_target,
            {'sep': ';', 'index': False, 'encoding': 'utf-8'},
        ),
        (
            _build_geom_clusters(cluster_frame),
            centers_path,
            {'sep': ';', 'encoding': 'utf-8'},
        ),
    ]
    if not matrix_unchanged or not os.path.isfile(matrix_path):
        writes.append((_stop_distance_matrix_writer(
            cluster_frame, k_walk, metric_crs=cluster_frame.crs), matrix_path, {}))
        settings = _read_calculation_settings(path_src)
        settings.update(k_walk=k_walk, distance_unit='km', metric_crs=str(cluster_frame.crs))
        writes.append((settings, os.path.join(path_src, 'geo_calculation_settings.json'), {}))
    deletes = []
    if district_layer is not None:
        writes.append(
            (
                _build_ref_stops(cluster_frame, district_frame, stop_frame),
                ref_stops_path,
                {'sep': ';', 'index': False, 'encoding': 'utf-8'},
            )
        )
    else:
        deletes.append(ref_stops_path)
    return writes, deletes


def _invalidate_graph_routes(proj_name, file_path):
    """Возвращает (writes, deletes, count) для очистки маршрутов старого графа.

    Рассчитанные ID определяются по rnd в routes_elements; ручные и загруженные
    маршруты без таких сегментов сохраняются. Пустой слой маршрутов ставится
    на удаление. count — число уникальных рассчитанных ID. Файлы не меняются
    до передачи заданий в _write_geo_files.
    """
    elements_path = os.path.join(file_path, f'routes_elements_{proj_name}.csv')
    if not os.path.isfile(elements_path):
        return [], [], 0
    route_ids = set(_load_route_elements(elements_path)['rnd'])
    writes, deletes = [], [elements_path]
    routes_path = os.path.join(file_path, f'routes_{proj_name}.csv')
    if route_ids and os.path.isfile(routes_path):
        routes = pd.read_csv(routes_path, sep=';', dtype={'id': str}, keep_default_na=False)
        routes = routes.loc[~routes['id'].isin(route_ids)].copy()
        if routes.empty:
            deletes.append(routes_path)
        else:
            writes.append((routes, routes_path, {'sep': ';', 'encoding': 'utf-8', 'index': False}))
    return writes, deletes, len(route_ids)


def get_osm_G(log_username, file_path, proj_name, area_name):
    """Загружает сеть OSM типа drive для area_name и сохраняет g.graphml.

    После установки графа удаляются его прежняя подготовленная основа
    и рассчитанные маршруты по заданиям _invalidate_graph_routes.
    log_username/proj_name используются для журналирования.
    Возвращает (True, сообщение, graph), при исключении —
    (False, безопасное сообщение, None). Ошибка логируется с traceback;
    сохранение использует правила частичной установки _write_geo_files.
    """
    try:
        graph = _download_osm_graph(area_name)
        writes, deletes, removed_count = _invalidate_graph_routes(proj_name, file_path)
        writes.insert(0, (lambda target: ox.save_graphml(graph, target),
                          os.path.join(file_path, 'g.graphml'), {}))
        # Прежняя основа удаляется после успешной установки новых файлов;
        # новая будет подготовлена при первом запросе маршрута.
        deletes.append(os.path.join(file_path, PREPARED_ROUTE_GRAPH_FILENAME))
        _write_geo_files(writes, deletes)
        message = 'Граф успешно загружен'
        if removed_count:
            message += f'. Удалено рассчитанных маршрутов: {removed_count}; создайте их заново.'
        return True, message, graph
    except _OverpassUnavailable:
        app.logger.exception('%s - geo project %s all Overpass servers failed', log_username, proj_name)
        return False, 'Серверы загрузки дорожной сети недоступны. Попробуйте позднее.', None
    except Exception:
        app.logger.exception('%s - geo project %s graph load error', log_username, proj_name)
        return False, 'Проблема загрузки графа, попробуйте уточнить адрес', None

def get_cluster(log_username, proj_name, path_src, path_out, eps_val1, eps_val2, eps_val3, use_draw_districts, use_gen_districts, analitic_cluster_name, k_walk=None, boundary_buffer_m=100.0):
    """Кластеризует остановки и сохраняет слои, справочники, матрицу и настройки.

    eps_val1/eps_val2 — пороги соседства DBSCAN для групп и аналитических
    кластеров в метрах (0 < eps_val1 <= eps_val2), не предельные диаметры.
    eps_val3 — порог группировки для генерации районов в метрах. use_gen_districts имеет
    приоритет над use_draw_districts; без обоих режимов ref_stops удаляется.
    Граница Вороного — оболочка всех остановок с boundary_buffer_m метров буфера.

    k_walk берется из аргумента либо сохраненных настроек/значения 1.15.
    analitic_cluster_name — имя CSV центров в path_out. path_src/path_out
    должны быть строками с завершающим разделителем пути.
    Возвращает (успех, сообщение), логируя ошибки. Формирует весь комплект
    до установки; ее атомарность и ограничения заданы _write_geo_files.
    """
    try:
        eps_val1 = _positive_distance(eps_val1, 'Порог объединения остановок в группы')
        eps_val2 = _positive_distance(eps_val2, 'Порог объединения остановок в аналитические кластеры')
        if eps_val1 > eps_val2:
            raise GeoValidationError('Порог группы остановок не должен превышать порог аналитического кластера')
        if use_gen_districts:
            eps_val3 = _positive_distance(eps_val3, 'Порог генерации районов')
        geo_stop_name = f'stops_{proj_name}.csv'
        geo_stop = _load_required_geo_layer(path_src + geo_stop_name, 'слой остановок')
        if geo_stop.empty or not geo_stop.geometry.geom_type.eq('Point').all():
            raise GeoValidationError('Для кластеризации нужен непустой слой точечных остановок')
        geo_stop['id'] = _validate_id_column(geo_stop, 'id', 'остановок', unique=True)
        geo_stop = _metric_frame(geo_stop)
        geo_stop.drop(columns=['wkt', 'name'], errors='ignore', inplace=True)
        # Справочник строится по исходным точкам, а не по центрам групп:
        # объединение остановок не должно сокращать число id_stop.
        original_stops = geo_stop.copy()
        coordinates = np.column_stack((geo_stop.geometry.x, geo_stop.geometry.y))
        # Пороги соседства DBSCAN выражены в метрах, диаметр кластера не ограничен.
        geo_stop = geo_stop.assign(
            id_grp_stp=skc.DBSCAN(eps=eps_val1, min_samples=1).fit_predict(coordinates))
        # Больший порог гарантирует вложенность групп в аналитические кластеры.
        geo_stop = geo_stop.assign(
            id_cluster=skc.DBSCAN(eps=eps_val2, min_samples=1).fit_predict(coordinates))
        geo_stop = _prepare_cluster_frame(geo_stop, metric_crs=geo_stop.crs)
        cluster_layer = geo_stop.to_crs('epsg:4326')
        cluster_layer_name = f'clusters_{proj_name}.csv'
        cluster_layer = cluster_layer.rename_geometry('wkt')
        # Слой для карты хранит все остановки, отдельный справочник — центры.
        geom_clusters_output = _build_geom_clusters(geo_stop)
        csv_writes = [
            (
                cluster_layer,
                path_src + cluster_layer_name,
                {'sep': ';', 'index': False, 'encoding': 'utf-8'},
            ),
            (
                geom_clusters_output,
                path_out + analitic_cluster_name,
                {'sep': ';', 'encoding': 'utf-8'},
            ),
        ]
        msg = "Кластеризация ОП успешно завершена"
        # Сначала выбираем источник районов: генерация имеет приоритет.
        do_ref = 0
        voronoi_settings = None
        if use_gen_districts:
            geo_distr, border, diagnostics = _build_voronoi_districts(
                original_stops, eps_val3, buffer_m=boundary_buffer_m,
            )
            gen_districts_name = f'districtsGEN_{proj_name}.csv'
            generated_districts_output = geo_distr.to_crs(4326).rename_geometry('wkt')
            csv_writes.append(
                (
                    generated_districts_output,
                    path_src + gen_districts_name,
                    {'sep': ';', 'encoding': 'utf-8', 'index': False},
                )
            )
            boundary_output = gpd.GeoDataFrame({'id': [1]}, geometry=[border], crs=geo_stop.crs)
            csv_writes.append((boundary_output.to_crs(4326).rename_geometry('wkt'),
                               path_out + f'districts_boundary_{proj_name}.csv',
                               {'sep': ';', 'encoding': 'utf-8', 'index': False}))
            voronoi_settings = {'algorithm': 'shapely_voronoi',
                                'buffer_m': float(boundary_buffer_m), 'radius_m': float(eps_val3),
                                'diagnostics': diagnostics, 'id_scheme': 'sequential'}
            do_ref = 1
        else:
            # Без генерации можно использовать загруженные/нарисованные районы.
            if use_draw_districts:
                geo_distr_name = f'districts_{proj_name}.csv'
                geo_distr = _load_required_geo_layer(
                    path_src + geo_distr_name, 'слой транспортных районов: districts')
                do_ref =1
        # Район назначается каждой исходной остановке пространственным пересечением.
        if do_ref == 1:
            df = _build_ref_stops(geo_stop, geo_distr, original_stops)
            ref_stop_out = f'ref_stops_{proj_name}.csv'
            csv_writes.append(
                (
                    df,
                    path_out + ref_stop_out,
                    {'sep': ';', 'index': False, 'encoding': 'utf-8'},
                )
            )
            # Внешние ОП допустимы; пропуск района внутри сгенерированной границы — ошибка.
            not_covered_stops = int(df['tr_district'].isna().sum())
            if use_gen_districts:
                unassigned_inside = not_covered_stops - diagnostics['outside_boundary']
                if unassigned_inside:
                    raise GeoValidationError('Справочник содержит остановки внутри границы без района')
                msg = (f'Кластеризация и генерация районов завершены. '
                       f'Вне границы территории: {diagnostics["outside_boundary"]} ОП; '
                       f'внутри границы без района: {unassigned_inside} ОП.')
            elif not_covered_stops > 0:
                msg = (f'Кластеризация ОП завершена. Требуется проверка слоя транспортных районов: '
                       f'без района осталось {not_covered_stops} ОП.')
        # Старые производные не должны пережить смену режима расчета.
        deletes = [] if do_ref else [path_out + f'ref_stops_{proj_name}.csv']
        if not use_gen_districts:
            deletes.append(path_src + f'districtsGEN_{proj_name}.csv')
            deletes.append(path_out + f'districts_boundary_{proj_name}.csv')
        # Матрица и ее коэффициент публикуются в одном подготовленном комплекте.
        calculation_settings = _read_calculation_settings(path_src)
        if k_walk is None:
            k_walk, _ = _load_geo_calculation_settings(path_src, proj_name, calculation_settings)
        k_walk = _positive_distance(k_walk, 'Коэффициент непрямолинейности')
        selected_district = 'districtsGEN' if use_gen_districts else ('districts' if use_draw_districts else None)
        calculation_settings.update(k_walk=float(k_walk), district_layer=selected_district,
                                    distance_unit='km', metric_crs=str(geo_stop.crs),
                                    cluster_parameters={'stop_group_radius_m': float(eps_val1),
                                                        'analytic_radius_m': float(eps_val2)})
        if voronoi_settings is not None:
            calculation_settings['voronoi'] = dict(voronoi_settings, enabled=True)
        elif 'voronoi' in calculation_settings:
            calculation_settings['voronoi']['enabled'] = False
        csv_writes.extend([
            (_stop_distance_matrix_writer(geo_stop, k_walk, metric_crs=geo_stop.crs),
             path_out + f'matrix_dist_stops_{proj_name}.csv',
             {'sep': ';', 'index': False, 'encoding': 'utf-8'}),
            (calculation_settings, path_src + 'geo_calculation_settings.json', {}),
        ])
        _write_geo_files(csv_writes, deletes)
        return True, msg
    except GeoValidationError as app_error:
        app.logger.warning('%s - geo project %s validation error: %s', log_username, proj_name, app_error)
        return False, f'Ошибка генерации/расчета: {app_error}'
    except Exception:
        app.logger.exception('%s - geo project %s cluster calc error', log_username, proj_name)
        return False, 'Проблема кластеризации остановок, обратитесь в техподдержку'

def get_matrix(proj_name, path_src, path_out, k_walk, matrix_name_out):
    """Пересчитывает полную матрицу расстояний между центрами групп остановок.

    Читает clusters_<proj_name>.csv, пишет matrix_name_out в path_out
    и обновляет k_walk, единицы 'km' и CRS в настройках. Все направленные
    пары без диагонали записываются блоками. path_src/path_out — строки
    с завершающим разделителем. Возвращает сообщение об успехе;
    исключения валидации и сохранения передаются вызывающему коду.
    """
    geo_stop_name = f'clusters_{proj_name}.csv'
    geo_stop = _load_required_geo_layer(path_src + geo_stop_name, 'слой кластеров')
    geo_stop = _prepare_cluster_frame(geo_stop)
    matrix_main = _stop_distance_matrix_writer(geo_stop, k_walk, metric_crs=geo_stop.crs)
    settings = _read_calculation_settings(path_src)
    settings.update(k_walk=float(k_walk), distance_unit='km', metric_crs=str(geo_stop.crs))
    _write_geo_files([
        (
            matrix_main,
            path_out + matrix_name_out,
            {'sep': ';', 'index': False, 'encoding': 'utf-8'},
        ),
        (settings, os.path.join(path_src, 'geo_calculation_settings.json'), {}),
    ])
    return 'Рассчитана матрица межостановочных расстояний'


def _clear_active_district(settings):
    """Сбрасывает выбор районов и флаг генерации в dict settings на месте.

    Остальные параметры Вороного сохраняются для следующего расчета;
    JSON и производные файлы эта функция не изменяет.
    """
    settings['district_layer'] = None
    settings['voronoi'] = dict(settings.get('voronoi', {}), enabled=False)


def _invalidate_source_dependents(proj_name, path_src, path_out, layer, changed_ids,
                                  district_frame=None, removed_layer=False):
    """Возвращает (writes, deletes) для зависимых данных измененного источника.

    changed_ids — строковые ID объектов; пустой набор без removed_layer
    означает отсутствие значимого изменения. Изменение выбранной границы сбрасывает генерацию,
    изменение stops инвалидирует кластеры/матрицу и затронутые маршруты.
    Для активных районов district_frame позволяет пересобрать ref_stops;
    removed_layer=True означает удаление всего слоя. Работа на диске
    ограничивается чтением: запись/удаление выполняет _write_geo_files.
    """
    writes, deletes = [], []
    if not changed_ids and not removed_layer:
        return writes, deletes
    settings = _read_calculation_settings(path_src)
    if layer == 'stops':
        deletes.append(os.path.join(path_out, f'ref_stops_{proj_name}.csv'))
    if layer in ('districts', 'districtsGEN'):
        # Правка неактивных районов не меняет справочник другого выбранного слоя.
        _, active_district = _load_geo_calculation_settings(path_src, proj_name, settings)
        ref_path = os.path.join(path_out, f'ref_stops_{proj_name}.csv')
        clusters_path = os.path.join(path_src, f'clusters_{proj_name}.csv')
        stops_path = os.path.join(path_src, f'stops_{proj_name}.csv')
        if active_district == layer:
            if (not removed_layer and district_frame is not None
                    and os.path.isfile(clusters_path) and os.path.isfile(stops_path)):
                clusters = _prepare_cluster_frame(_load_geo_layer(clusters_path))
                ref = _build_ref_stops(clusters, district_frame, _load_geo_layer(stops_path))
                writes.append((ref, ref_path, {'sep': ';', 'index': False, 'encoding': 'utf-8'}))
            else:
                deletes.append(ref_path)
        elif active_district is None:
            deletes.append(ref_path)
    if layer == 'stops':
        if settings.get('district_layer') == 'districtsGEN':
            _clear_active_district(settings)
            writes.append((settings, os.path.join(path_src, 'geo_calculation_settings.json'), {}))
        deletes.extend([
            os.path.join(path_src, f'clusters_{proj_name}.csv'),
            os.path.join(path_src, f'districtsGEN_{proj_name}.csv'),
            os.path.join(path_out, f'geom_clusters_{proj_name}.csv'),
            os.path.join(path_out, f'matrix_dist_stops_{proj_name}.csv'),
            os.path.join(path_out, f'districts_boundary_{proj_name}.csv'),
        ])
        elements_path = os.path.join(path_src, f'routes_elements_{proj_name}.csv')
        if removed_layer:
            route_writes, route_deletes, _ = _invalidate_graph_routes(proj_name, path_src)
            return writes + route_writes, deletes + route_deletes
        if os.path.isfile(elements_path):
            # Удаляем весь маршрут, содержащий измененную ОП, вместе со всеми сегментами.
            elements = _load_route_elements(elements_path)
            affected = elements['source'].isin(changed_ids) | elements['target'].isin(changed_ids)
            route_ids = elements.loc[affected, 'rnd']
            if not route_ids.empty:
                elements = elements.loc[~elements['rnd'].isin(route_ids)].copy()
                if elements.empty:
                    deletes.append(elements_path)
                else:
                    writes.append((elements, elements_path, {'sep': ';', 'index': False, 'encoding': 'utf-8'}))
                routes_path = os.path.join(path_src, f'routes_{proj_name}.csv')
                if os.path.isfile(routes_path):
                    routes = pd.read_csv(routes_path, sep=';', dtype={'id': str}, keep_default_na=False)
                    routes = routes.loc[~routes['id'].isin(route_ids)].copy()
                    if routes.empty:
                        deletes.append(routes_path)
                    else:
                        writes.append((routes, routes_path, {'sep': ';', 'index': False, 'encoding': 'utf-8'}))
    return writes, deletes


def delete_geo_layer(proj_name, path_src, path_out, layer):
    """Удаляет файл слоя и обнаруженные зависимые данные проекта.

    layer — безопасное имя без суффикса проекта. Зависимости определяются
    типом слоя и текущими настройками. Для активного
    слоя районов сбрасывается выбор; для routes удаляется таблица сегментов.
    Возвращает None, исключения чтения/сохранения не перехватываются.
    """
    _validate_layer_name(layer)
    layer_path = os.path.join(path_src, f'{layer}_{proj_name}.csv')
    changed_ids = set()
    writes, deletes = _invalidate_source_dependents(
        proj_name, path_src, path_out, layer, changed_ids, removed_layer=True,
    )
    deletes.append(layer_path)
    if layer == 'clusters':
        deletes.extend(os.path.join(path_out, f'{prefix}_{proj_name}.csv')
                       for prefix in ('geom_clusters', 'matrix_dist_stops', 'ref_stops'))
    if layer == 'routes':
        deletes.append(os.path.join(path_src, f'routes_elements_{proj_name}.csv'))
    if layer in ('districts', 'districtsGEN'):
        k_walk, active = _load_geo_calculation_settings(path_src, proj_name)
        if active == layer:
            settings = _read_calculation_settings(path_src)
            settings.update(k_walk=k_walk, district_layer=None)
            if layer == 'districtsGEN':
                _clear_active_district(settings)
            writes.append((settings,
                           os.path.join(path_src, 'geo_calculation_settings.json'), {}))
        if layer == 'districtsGEN':
            deletes.append(os.path.join(path_out, f'districts_boundary_{proj_name}.csv'))
    _write_geo_files(writes, list(dict.fromkeys(deletes)))


def _validated_geo_edit(function):
    """Оборачивает CRUD-функцию, возвращая безопасный отказ вместо исключения.

    GeoValidationError дает (False, {'err_info': текст}); внутренние ошибки
    логируются с traceback и заменяются общим сообщением. wraps сохраняет
    метаданные исходной функции; успешный результат передается без изменений.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        """Выполняет операцию и преобразует исключение в ответ изменения геоданных."""
        try:
            return function(*args, **kwargs)
        except GeoValidationError as error:
            app.logger.warning('Ошибка изменения геоданных: %s', error)
            return False, {'err_info': str(error)}
        except Exception:
            app.logger.exception('Внутренняя ошибка изменения геоданных')
            return False, {'err_info': 'Не удалось изменить геоданные. Обратитесь в техподдержку.'}
    return wrapped


def _valid_geo_id(value):
    """Проверяет скалярный ID: непустой str, int или конечный float, кроме bool.

    Возвращает bool без изменения значения; преобразование принятого ID
    в строку выполняется при обработке операции.
    """
    return (isinstance(value, (str, int, float)) and not isinstance(value, bool)
            and bool(str(value).strip())
            and (not isinstance(value, float) or np.isfinite(value)))


def _validate_geo_payload(layer_name, action_type, obj_data, expected_geometry):
    """Проверяет запрос одной CRUD-операции, выдавая GeoValidationError при отказе.

    createSM принимает id маршрута и минимум два ID остановок в elements.
    Остальные действия требуют ровно один Feature с properties.id;
    delete не требует геометрии. createDM/modify используют GeoJSON в 3857.
    Семейство геометрии задает expected_geometry либо тип основного слоя;
    LineString/Polygon допускают соответствующий Multi-тип. Файлы не читаются.
    """
    if action_type not in ('createDM', 'createSM', 'modify', 'delete'):
        raise GeoValidationError('Неизвестное действие')
    if not isinstance(obj_data, dict):
        raise GeoValidationError('Ожидается объект GeoJSON')
    if action_type == 'createSM':
        elements = obj_data.get('elements')
        if (layer_name != 'routes' or not _valid_geo_id(obj_data.get('id'))
                or not isinstance(elements, list) or len(elements) < 2
                or not all(_valid_geo_id(value) for value in elements)):
            raise GeoValidationError('Маршрут требует id и минимум две остановки с непустыми id')
        return

    features = obj_data.get('features')
    if not isinstance(features, list) or len(features) != 1:
        raise GeoValidationError('За один запрос изменяется ровно один объект')
    feature = features[0]
    if (not isinstance(feature, dict) or not isinstance(feature.get('properties'), dict)
            or not _valid_geo_id(feature['properties'].get('id'))):
        raise GeoValidationError('Объект требует непустой скалярный id')
    if action_type == 'delete':
        return
    if not isinstance(feature.get('geometry'), dict):
        raise GeoValidationError('Объект требует геометрию GeoJSON')
    try:
        geometry = shapely.geometry.shape(feature['geometry'])
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, shapely.errors.ShapelyError) as error:
        raise GeoValidationError('Некорректная геометрия GeoJSON') from error
    family = (expected_geometry if expected_geometry != 'none' else None) or {
        'stops': 'Point', 'clusters': 'Point', 'routes': 'LineString',
        'districts': 'Polygon', 'districtsGEN': 'Polygon',
    }.get(layer_name)
    allowed = {'Point': {'Point'}, 'LineString': {'LineString', 'MultiLineString'},
               'Polygon': {'Polygon', 'MultiPolygon'}}.get(family)
    if (geometry.is_empty or not geometry.is_valid
            or not np.isfinite(shapely.get_coordinates(geometry)).all()
            or (allowed is not None and geometry.geom_type not in allowed)):
        raise GeoValidationError('Некорректная геометрия или тип геометрии слоя')
    error = geo_geometry_error(gpd.GeoDataFrame(geometry=[geometry], crs=3857))
    if error:
        raise GeoValidationError(error)


def _geo_layer_change_writes(proj_name, file_path, file_name, geo_stop_name,
                            layer_name, frame, output_path, changed_ids):
    """Возвращает (writes, deletes) для измененного слоя и его зависимостей.

    frame уже подготовлен для CSV/WKT. Для clusters вызывается пересчет
    производных; для других слоев добавляется запись самого слоя и задания
    инвалидации по changed_ids. Последний кластер удаляется вместе с
    производными без попытки расчёта пустого слоя; пустые routes удаляются.
    Файлы здесь не записываются.
    """
    if layer_name == 'clusters':
        if frame.empty:
            return [], [os.path.join(file_path, file_name), *[
                os.path.join(output_path, f'{prefix}_{proj_name}.csv')
                for prefix in ('geom_clusters', 'matrix_dist_stops', 'ref_stops')]]
        return _cluster_bundle_writes(
            proj_name, file_path, output_path, frame,
            os.path.join(file_path, file_name), os.path.join(file_path, geo_stop_name),
        )
    writes, deletes = _invalidate_source_dependents(
        proj_name, file_path, output_path, layer_name, changed_ids,
        district_frame=frame if layer_name in ('districts', 'districtsGEN') else None,
    )
    if layer_name == 'routes' and frame.empty:
        deletes.append(os.path.join(file_path, file_name))
    else:
        writes.insert(0, (frame, os.path.join(file_path, file_name),
                          {'sep': ';', 'encoding': 'utf-8', 'index': False}))
    return writes, deletes


def _upsert_geo_feature(proj_name, file_path, file_name, geo_stop_name,
                        layer_name, action_type, obj_data, output_path):
    """Создает или полностью заменяет один Feature по ID и сохраняет зависимости.

    Принимает проверенный GeoJSON в 3857 для createDM/modify. ID должен быть
    уникальным при создании и найден ровно один раз при замене. Результат
    сохраняется в CSV/WKT в 4326. Смена геометрии рассчитанного маршрута
    запрещена; правка атрибутов остановки при прежней геометрии сохраняет
    ее расчеты. Преобразуются только выбранный старый и входящий объект;
    замена сохраняет позицию строки, создание добавляет строку в конец.
    Возвращает (True, None), ошибки передает внешнему декоратору.
    """
    target = os.path.join(file_path, file_name)
    original_feature = None
    original_position = None
    existing = None
    if os.path.isfile(target):
        existing = _load_geo_layer(target).drop(columns='wkt')
        if action_type == 'modify':
            object_id = str(obj_data['features'][0]['properties']['id'])
            matching = existing['id'].eq(object_id)
            if matching.sum() != 1:
                raise GeoValidationError(f'Не найден единственный объект с id {object_id}')
            original_position = int(np.flatnonzero(matching.to_numpy())[0])
            original_feature = existing.loc[matching].to_crs(3857)
    elif action_type == 'modify':
        raise GeoValidationError('Нельзя изменить объект: слой отсутствует')

    incoming = gpd.GeoDataFrame.from_features(obj_data['features'], crs=3857)
    incoming['id'] = incoming['id'].astype(str)
    if layer_name == 'stops' and 'name' not in incoming.columns:
        incoming['name'] = ''
    if (action_type == 'createDM' and existing is not None
            and incoming['id'].isin(existing['id']).any()):
        raise GeoValidationError('Объект с таким id уже существует')

    # Допуск в единицах 3857 учитывает дрейф координат при обмене с редактором.
    geometry_changed = (original_feature is None
                        or not original_feature.geometry.iloc[0].equals_exact(
                            incoming.geometry.iloc[0], tolerance=0.05))
    elements_path = os.path.join(file_path, f'routes_elements_{proj_name}.csv')
    if (layer_name == 'routes' and original_feature is not None and geometry_changed
            and os.path.isfile(elements_path)):
        route_id = incoming['id'].iloc[0]
        if _load_route_elements(elements_path)['rnd'].eq(route_id).any():
            raise GeoValidationError('Геометрию рассчитанного маршрута нельзя менять напрямую. '
                             'Удалите маршрут и создайте его заново по остановкам.')

    incoming = incoming.to_crs(4326)
    error = geo_geometry_error(incoming)
    if error:
        raise GeoValidationError(error)
    if existing is None:
        combined = incoming
    elif original_position is None:
        combined = _concat_geo_frames([existing, incoming])
    else:
        combined = _concat_geo_frames([
            existing.iloc[:original_position], incoming, existing.iloc[original_position + 1:]])
    output = combined.rename_geometry('wkt')
    changed_ids = set(incoming['id'])
    if layer_name == 'stops' and not geometry_changed:
        # Подпись остановки не влияет на маршруты и расстояния.
        changed_ids = set()
    writes, deletes = _geo_layer_change_writes(
        proj_name, file_path, file_name, geo_stop_name,
        layer_name, output, output_path, changed_ids,
    )
    _write_geo_files(writes, deletes)
    return True, None


def _delete_geo_feature(proj_name, file_path, file_name, geo_stop_name,
                       layer_name, obj_data, output_path):
    """Удаляет один объект по properties.id и сохраняет изменения зависимостей.

    В слое должен существовать ровно один такой ID. При удалении маршрута
    также исключаются все его строки routes_elements; пустые файлы routes
    и routes_elements удаляются. Остальные обычные слои сохраняют заголовок.
    Последний кластер удаляется с производными. Возвращает
    (True, None); ошибки передаются декоратору _validated_geo_edit.
    """
    object_id = str(obj_data['features'][0]['properties']['id'])
    frame = _load_geo_layer(os.path.join(file_path, file_name))
    matching = frame['id'].eq(object_id)
    if matching.sum() != 1:
        raise GeoValidationError(f'Не найден единственный объект с id {object_id}')
    frame = frame.loc[~matching].drop(columns='wkt').to_crs(4326).rename_geometry('wkt')
    writes, deletes = _geo_layer_change_writes(
        proj_name, file_path, file_name, geo_stop_name,
        layer_name, frame, output_path, {object_id},
    )
    if layer_name == 'routes':
        elements_path = os.path.join(file_path, f'routes_elements_{proj_name}.csv')
        if frame.empty:
            deletes.append(elements_path)
        elif os.path.isfile(elements_path):
            elements = _load_route_elements(elements_path)
            elements = elements.loc[~elements['rnd'].eq(object_id)].copy()
            if elements.empty:
                deletes.append(elements_path)
            else:
                writes.append((elements, elements_path,
                               {'sep': ';', 'encoding': 'utf-8', 'index': False}))
    _write_geo_files(writes, deletes)
    return True, None


def _route_graph_fingerprint(filename):
    """Возвращает SHA-256, размер и mtime_ns исходного GraphML.

    Файл читается блоками по 1 МиБ; ошибки доступа к файлу передаются вызывающему коду.
    Отпечаток определяет актуальность подготовленной основы на диске.
    """
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = os.stat(filename)
    return {'sha256': digest.hexdigest(), 'size': after.st_size,
            'mtime_ns': after.st_mtime_ns}


def _write_prepared_route_graph(filename, edges, nodes, fingerprint):
    """Записывает подготовленные edges/nodes и метаданные в новый файл SQLite.

    Оба GeoDataFrame имеют общую метрическую CRS; ID узлов целочисленные.
    Геометрия записывается как WKB, веса — как length в метрах.
    Метаданные содержат fingerprint исходника, версию подготовки и CRS.
    position сохраняет порядок строк. Файл должен быть пустым/новым;
    вызывающий _write_geo_files обеспечивает временный путь и атомарную замену.
    """
    metadata = {'version': ROUTE_GRAPH_PREPARATION_VERSION,
                'source': fingerprint, 'crs': nodes.crs.to_wkt(),
                'node_index_name': nodes.index.name}
    connection = sqlite3.connect(filename)
    try:
        with connection:
            connection.execute('CREATE TABLE metadata (value TEXT NOT NULL)')
            connection.execute('INSERT INTO metadata VALUES (?)',
                               (json.dumps(metadata),))
            connection.execute('CREATE TABLE nodes '
                               '(position INTEGER, id INTEGER PRIMARY KEY, wkb BLOB NOT NULL)')
            connection.execute('CREATE TABLE edges '
                               '(position INTEGER, u INTEGER, v INTEGER, '
                               'length REAL NOT NULL, wkb BLOB NOT NULL, PRIMARY KEY (u, v))')
            # Пишем строки через генераторы, без отдельного списка всех SQL-записей.
            connection.executemany('INSERT INTO nodes VALUES (?, ?, ?)',
                ((i, int(node_id), geometry) for i, (node_id, geometry) in
                 enumerate(zip(nodes.index, nodes.geometry.to_wkb()))))
            connection.executemany('INSERT INTO edges VALUES (?, ?, ?, ?, ?)',
                ((i, int(u), int(v), float(length), geometry)
                 for i, ((u, v), length, geometry) in enumerate(zip(
                     edges.index, edges['length'], edges.geometry.to_wkb()))))
    finally:
        connection.close()


def _validate_route_graph_base(edges, nodes):
    """Проверяет основу до сохранения или использования в маршрутизации."""
    validate_simple_edges(edges)
    if nodes.empty or edges.empty or not nodes.index.is_unique or nodes.index.hasnans:
        raise GeoValidationError('Основа дорожного графа пуста или содержит неверные ID узлов')
    if (nodes.crs is None or edges.crs != nodes.crs or not nodes.crs.is_projected
            or any(axis.unit_conversion_factor != 1 for axis in nodes.crs.axis_info)):
        raise GeoValidationError('Основа дорожного графа должна иметь общую метрическую CRS')
    weights = pd.to_numeric(edges['length'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise GeoValidationError('Вес дорожного ребра должен быть конечным неотрицательным числом')
    for frame, geometry_type in ((nodes, 'Point'), (edges, 'LineString')):
        geometry = frame.geometry
        if (geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all()
                or not geometry.geom_type.eq(geometry_type).all()
                or not np.isfinite(shapely.get_coordinates(geometry)).all()):
            raise GeoValidationError('Геометрия основы дорожного графа повреждена')
    if any(not edges.index.get_level_values(level).isin(nodes.index).all() for level in (0, 1)):
        raise GeoValidationError('Конечный узел дорожного ребра отсутствует в основе')


def _read_prepared_route_graph(filename, fingerprint):
    """Возвращает (edges, nodes) из актуальной основы либо None при устаревании.

    SQLite открывается только для чтения. До загрузки таблиц сравниваются
    версия подготовки и fingerprint исходного GraphML. Восстанавливаются
    геометрия WKB, CRS и индексы; проверяются веса, геометрия и конечные узлы.
    Ошибки SQLite, WKB и CRS передаются _load_route_graph для перестроения.
    Каждый вызов создает новые объекты в памяти; постоянного кеша нет.
    """
    # URI сохраняет пробелы/кириллицу в пути; mode=ro не создает отсутствующий файл.
    connection = sqlite3.connect(Path(filename).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        row = connection.execute('SELECT value FROM metadata').fetchone()
        metadata = json.loads(row[0]) if row else {}
        if (metadata.get('version') != ROUTE_GRAPH_PREPARATION_VERSION
                or metadata.get('source') != fingerprint):
            return None
        # Устаревшая основа отбрасывается по метаданным до чтения тяжелых таблиц.
        nodes = pd.read_sql_query('SELECT id, wkb FROM nodes ORDER BY position', connection)
        edges = pd.read_sql_query('SELECT u, v, length, wkb FROM edges ORDER BY position', connection)
    finally:
        connection.close()
    crs = metadata['crs']
    nodes = nodes.set_index('id')
    nodes.index.name = metadata['node_index_name']
    nodes = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkb(nodes.pop('wkb'), crs=crs), crs=crs)
    edges = edges.set_index(['u', 'v'])
    geometry = gpd.GeoSeries.from_wkb(edges.pop('wkb'), crs=crs)
    edges = gpd.GeoDataFrame(edges, geometry=geometry, crs=crs)
    _validate_route_graph_base(edges, nodes)
    return edges, nodes


def _prepare_route_graph_base(graph_filename, proj_name):
    """Возвращает (edges, nodes) основы для маршрутизации из исходного GraphML.

    Подбирает UTM по узлам, переводит ребра в ту же CRS и преобразует
    мультиграф в простой, сохраняя параллельные дороги через промежуточные узлы.
    edges имеет индекс (u, v), length/geometry; nodes — индекс ID
    и geometry. Привязка остановок и сохранение основы выполняются отдельно.
    """
    graph = ox.load_graphml(graph_filename)
    nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False, node_geometry=True,
                             fill_edge_geometry=True)
    nodes = _metric_frame(nodes)[['geometry']]
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, node_geometry=True,
                             fill_edge_geometry=True).to_crs(nodes.crs)[['length', 'geometry']]
    edge_count_before = len(edges)
    node_count_before = len(nodes)
    edges, nodes = simplify_graph_edges(edges, nodes)
    edges = edges.drop(columns='key')
    _validate_route_graph_base(edges, nodes)
    app.logger.info('geo project %s graph simplified: edges_before=%s, edges_after=%s, '
                    'parallel_edges_split=%s, added_nodes=%s', proj_name, edge_count_before,
                    len(edges), len(edges) - edge_count_before, len(nodes) - node_count_before)
    return edges, nodes


def _load_route_graph(file_path, proj_name):
    """Возвращает рабочие (edges, nodes), используя основу g_prepared.sqlite.

    Проверяет отпечаток обязательного g.graphml и версию сохраненной основы.
    Отсутствующую, устаревшую или нечитаемую основу готовит заново и сохраняет
    атомарно. Ошибки подготовки/записи
    передаются вызывающему коду. Объекты не удерживаются в постоянном кеше.
    """
    graph_filename = os.path.join(file_path, 'g.graphml')
    prepared_filename = os.path.join(file_path, PREPARED_ROUTE_GRAPH_FILENAME)
    fingerprint = _route_graph_fingerprint(graph_filename)
    if os.path.isfile(prepared_filename):
        try:
            prepared = _read_prepared_route_graph(prepared_filename, fingerprint)
            if prepared is not None:
                return prepared
        except (OSError, sqlite3.Error, pd.errors.DatabaseError, CRSError, ValueError, KeyError,
                TypeError, AttributeError, shapely.errors.ShapelyError):
            app.logger.warning('geo project %s prepared graph is unreadable; rebuilding', proj_name)
    edges, nodes = _prepare_route_graph_base(graph_filename, proj_name)
    _write_geo_files([(lambda filename: _write_prepared_route_graph(
        filename, edges, nodes, fingerprint), prepared_filename, {})])
    app.logger.info('geo project %s prepared graph saved to disk', proj_name)
    return edges, nodes


def _split_edge_at_positions(edge, edge_id, geometry, positions, node_ids):
    """Возвращает участки ребра как список ((source, target), attributes).

    geometry — непустая LineString ненулевой длины, ориентированная от u к v.
    positions — отсортированные внутренние расстояния вдоль линии в метрах;
    node_ids — соответствующие ID точек разреза. Геометрия сохраняет изгибы,
    вес length распределяется пропорционально геометрической длине частей.
    Проверка этих предусловий возложена на вызывающий код.
    """
    cuts = [0.0, *positions, geometry.length]
    sequence = [edge_id[0], *node_ids, edge_id[1]]
    lengths = float(edge['length']) * np.diff(cuts) / geometry.length
    # Поправка последней части сохраняет исходный вес несмотря на округление float.
    lengths[-1] += float(edge['length']) - lengths.sum()
    rows = []
    for i, length in enumerate(lengths):
        row = edge.to_dict()
        row.update(length=length, geometry=substring(geometry, cuts[i], cuts[i + 1]))
        rows.append(((sequence[i], sequence[i + 1]), row))
    return rows


def _prepare_route_graph(edges, nodes, stop_points):
    """Привязывает точки остановок к ребрам, возвращая рабочий граф и назначения.

    edges — простой ориентированный граф с индексом (u, v), length/geometry;
    nodes и stop_points заданы в той же метрической CRS. Результат —
    (edges, nodes, route_nodes, distances), где последние два объекта
    соответствуют порядку stop_points; distances — расстояния до ребер в метрах.

    Разрезы создаются внутри выбранных ребер; проекции в концах используют
    исходные узлы. Совпадающие встречные дороги получают общие узлы, а
    пересечения разных дорог не создают соединений. Исходные таблицы
    не изменяются; проверка максимального расстояния выполняется вызывающим кодом.
    """
    validate_simple_edges(edges)
    if edges.empty or nodes.empty:
        raise GeoValidationError('Дорожный граф пуст')
    if not edges.geometry.geom_type.eq('LineString').all():
        raise GeoValidationError('Ребра дорожного графа должны быть линиями')
    if (edges.geometry.is_empty.any() or not edges.geometry.is_valid.all()
            or not np.isfinite(shapely.get_coordinates(edges.geometry.array)).all()):
        raise GeoValidationError('Граф содержит некорректные геометрии ребер')
    # Ищем ближайшую геометрию дороги, чтобы длинное ребро не проиграло чужому узлу.
    pairs, nearest_distances = edges.sindex.nearest(
        stop_points, return_all=True, return_distance=True)
    # У пространственного индекса порядок равных кандидатов не гарантирован.
    candidates = {}
    for stop_index, edge_index, distance in zip(pairs[0], pairs[1], nearest_distances):
        edge_id = edges.index[edge_index]
        choice = (float(distance), edge_id)
        if stop_index not in candidates or choice < candidates[stop_index]:
            candidates[stop_index] = choice
    route_nodes = [None] * len(stop_points)
    distances = np.empty(len(stop_points), dtype=float)
    groups, paired_edges = {}, {}
    # Совпадающую встречную пару обрабатываем вместе, сохраняя движение в обе стороны.
    for i, point in enumerate(stop_points):
        if i not in candidates:
            raise GeoValidationError('Не удалось найти дорожное ребро для остановки')
        distances[i], edge_id = candidates[i]
        if edge_id not in paired_edges:
            reverse_id = (edge_id[1], edge_id[0])
            shared = (reverse_id != edge_id and reverse_id in edges.index
                      and edges.loc[edge_id].geometry.equals(edges.loc[reverse_id].geometry))
            canonical = min(edge_id, reverse_id) if shared else edge_id
            paired_edges[edge_id] = canonical
            if shared:
                paired_edges[reverse_id] = canonical
        groups.setdefault(paired_edges[edge_id], []).append((i, point))

    next_node_id = int(nodes.index.max()) + 1
    new_nodes, new_edges, removed_edges = [], [], []
    tolerance = 1e-6  # Только погрешность вычислений, в метрах.
    for edge_id, selected in sorted(groups.items()):
        edge = edges.loc[edge_id]
        geometry = shapely.LineString(_orient_edge_coordinates(
            shapely.get_coordinates(edge.geometry), nodes.loc[edge_id[0]].geometry))
        projections = sorted((geometry.project(point), i) for i, point in selected)
        # Совпавшие проекции разделяют один узел; концы не создают нулевые части.
        positions, node_ids = [], []
        for position, i in projections:
            if position <= tolerance:
                route_nodes[i] = edge_id[0]
            elif geometry.length - position <= tolerance:
                route_nodes[i] = edge_id[1]
            else:
                if not positions or position - positions[-1] > tolerance:
                    positions.append(position)
                    node_ids.append(next_node_id)
                    new_nodes.append((next_node_id, geometry.interpolate(position)))
                    next_node_id += 1
                route_nodes[i] = node_ids[-1]
        if not positions:
            continue
        new_edges.extend(_split_edge_at_positions(edge, edge_id, geometry, positions, node_ids))
        removed_edges.append(edge_id)
        reverse_id = (edge_id[1], edge_id[0])
        if reverse_id != edge_id and paired_edges.get(reverse_id) == edge_id:
            # В обратном направлении меняются порядок узлов и отсчет расстояний.
            reverse_edge = edges.loc[reverse_id]
            reverse_geometry = shapely.LineString(_orient_edge_coordinates(
                shapely.get_coordinates(reverse_edge.geometry), nodes.loc[reverse_id[0]].geometry))
            reverse_positions = [reverse_geometry.length - p for p in reversed(positions)]
            new_edges.extend(_split_edge_at_positions(
                reverse_edge, reverse_id, reverse_geometry, reverse_positions, list(reversed(node_ids))))
            removed_edges.append(reverse_id)
    if new_edges:
        # Объединяем изменения один раз; постоянная дисковая основа остается прежней.
        additions = gpd.GeoDataFrame([row for _, row in new_edges],
            index=pd.MultiIndex.from_tuples([key for key, _ in new_edges], names=['u', 'v']),
            geometry='geometry', crs=edges.crs)
        edges = gpd.GeoDataFrame(pd.concat([edges.drop(index=removed_edges), additions]),
                                 geometry='geometry', crs=edges.crs)
        additions = gpd.GeoDataFrame({'geometry': [point for _, point in new_nodes]},
            index=pd.Index([key for key, _ in new_nodes], name=nodes.index.name), crs=nodes.crs)
        nodes = gpd.GeoDataFrame(pd.concat([nodes, additions]), geometry='geometry', crs=nodes.crs)
    validate_simple_edges(edges)
    return edges, nodes, route_nodes, distances


def _calculate_route(edges, nodes, stops, route_id, stop_ids, max_snap_distance):
    """Возвращает (geometry, elements) маршрута по упорядоченным ID остановок.

    edges/nodes/stops заданы в общей метрической CRS. Каждому строковому ID
    в stop_ids должна соответствовать одна остановка. Привязка ограничена
    max_snap_distance метрами; для каждой соседней пары ищется направленный
    кратчайший путь по весам OSM. elements содержит rnd/source/target/length
    с длиной в метрах, округленной до 0,001.

    LineString/MultiLineString сохраняет порядок и повторные проходы ребер.
    Нулевые сегменты допустимы, но полностью нулевой маршрут отвергается.
    Недоступные пути и неверные входы дают GeoValidationError; файлы не меняются.
    """
    stop_points = []
    for stop_id in stop_ids:
        matching = stops.loc[stops['id'].eq(stop_id)]
        if len(matching) != 1:
            raise GeoValidationError(f'Не найдена единственная остановка с id {stop_id}')
        stop_points.append(matching.geometry.iloc[0])

    edges, nodes, route_nodes, distances = _prepare_route_graph(edges, nodes, stop_points)
    graph = get_G(edges)
    if not np.isfinite(max_snap_distance) or max_snap_distance <= 0:
        raise GeoValidationError('Порог привязки остановок должен быть положительным числом')
    too_far = np.flatnonzero(distances > max_snap_distance)
    if len(too_far):
        details = ', '.join(f'{stop_ids[i]}: {distances[i]:.1f} м' for i in too_far)
        raise GeoValidationError(f'Остановки слишком далеко от дорожных рёбер '
                         f'(порог {max_snap_distance:g} м): {details}')

    route_frames, segments = [], []
    # Порядок пар задает порядок сегментов, включая возвраты и повторные остановки.
    for i, (source, target) in enumerate(zip(route_nodes[:-1], route_nodes[1:])):
        predecessors, _ = deijkstra(graph, source, target)
        path = pathfinder(predecessors, source, target, max(1, len(predecessors)))
        if not path:
            raise GeoValidationError(f'Не найден дорожный путь между остановками {stop_ids[i]} и {stop_ids[i + 1]}')
        edge_keys = list(zip(path[:-1], path[1:]))
        length = 0.0
        if edge_keys:
            route_frame = edges.loc[edge_keys]
            length = route_frame['length'].sum()
            route_frames.append(route_frame)
        segments.append([route_id, stop_ids[i], stop_ids[i + 1], round(length, 3)])
    if not route_frames:
        raise GeoValidationError('Все остановки маршрута привязаны к одному дорожному узлу; маршрут не образует линию')
    parts = []
    # Не объединяем участки геометрическим union: он потерял бы повторные проходы.
    for frame in route_frames:
        for (source, _), edge in frame.iterrows():
            coordinates = _orient_edge_coordinates(
                shapely.get_coordinates(edge.geometry), nodes.loc[source].geometry)
            parts.append(shapely.LineString(coordinates))
    geometry = parts[0] if len(parts) == 1 else shapely.MultiLineString(parts)
    elements = pd.DataFrame(segments, columns=['rnd', 'source', 'target', 'length'])
    return geometry, elements


def _create_geo_route(proj_name, file_path, file_name, geo_stop_name, obj_data):
    """Добавляет маршрут по obj_data['elements'] и пересохраняет оба CSV проекта.

    Проверяет уникальность id в слое маршрутов и таблице сегментов,
    загружает основу/остановки и рассчитывает путь. Порог привязки берется
    из GEO_MAX_STOP_SNAP_DISTANCE_M (по умолчанию 100 м). Геометрия сохраняется
    в 4326, ответ (True, GeoJSON) содержит только новый объект в 3857.
    Ошибки передаются внешнему декоратору; запись основы графа при ее
    первом использовании может предшествовать сохранению маршрута.
    """
    route_id = str(obj_data['id'])
    route_path = os.path.join(file_path, file_name)
    elements_path = os.path.join(file_path, f'routes_elements_{proj_name}.csv')
    existing_routes = None
    existing_elements = None
    if os.path.isfile(route_path):
        existing_routes = _load_geo_layer(route_path).drop(columns='wkt')
        if existing_routes['id'].eq(route_id).any():
            raise GeoValidationError(f'Маршрут с id {route_id} уже существует')
    if os.path.isfile(elements_path):
        existing_elements = _load_route_elements(elements_path)
        if existing_elements['rnd'].eq(route_id).any():
            raise GeoValidationError(f'Сегменты маршрута с id {route_id} уже существуют')
    stops_path = os.path.join(file_path, geo_stop_name)
    stops = _load_required_geo_layer(stops_path, 'слой остановок')
    edges, nodes = _load_route_graph(file_path, proj_name)
    stops = stops.to_crs(nodes.crs)
    stop_ids = [str(value) for value in obj_data['elements']]
    geometry, elements = _calculate_route(
        edges, nodes, stops, route_id, stop_ids,
        float(app.config.get('GEO_MAX_STOP_SNAP_DISTANCE_M', 100.0)),
    )
    route = _build_route_feature(geometry, obj_data, nodes.crs)
    response = json.loads(route.to_crs(3857).to_json(drop_id=True))
    if existing_routes is not None:
        route = _concat_geo_frames([existing_routes, route])
    if existing_elements is not None:
        elements = pd.concat([existing_elements, elements], ignore_index=True)
    options = {'sep': ';', 'encoding': 'utf-8', 'index': False}
    _write_geo_files([
        (route.rename_geometry('wkt'), route_path, options),
        (elements, elements_path, options),
    ])
    return True, response


@_validated_geo_edit
def geo_data_modify(
        proj_name, file_path, file_name, geo_stop_name, layer_name,
        action_type, objData, output_path=None, expected_geometry=None):
    """Выполняет одну CRUD-операцию и возвращает (успех, результат).

    createDM/modify принимают objData с одним GeoJSON Feature в 3857;
    delete достаточно одного Feature с properties.id без геометрии.
    createSM — id маршрута и упорядоченный список ID остановок в elements.
    file_path/file_name задают изменяемый CSV, geo_stop_name — файл остановок.
    output_path задает каталог производных (по умолчанию file_path),
    expected_geometry — семейство геометрии из описания слоя.

    При обычном успехе результат None, при createSM — GeoJSON нового маршрута;
    декоратор возвращает {'err_info': сообщение} при ошибке. Служебные
    properties удаляются из objData на месте перед обработкой.
    """
    _validate_layer_name(layer_name)
    if action_type == 'createDM' and layer_name in ('clusters', 'districtsGEN'):
        raise GeoValidationError('Создание объектов генерируемого слоя запрещено')
    _validate_geo_payload(layer_name, action_type, objData, expected_geometry)
    _remove_geo_technical_properties(objData)
    if action_type == 'createSM':
        return _create_geo_route(proj_name, file_path, file_name, geo_stop_name, objData)
    output_path = output_path or file_path
    if action_type == 'delete':
        return _delete_geo_feature(proj_name, file_path, file_name, geo_stop_name,
                                   layer_name, objData, output_path)
    return _upsert_geo_feature(proj_name, file_path, file_name, geo_stop_name,
                               layer_name, action_type, objData, output_path)


def prepare_data_to_push(path_src, filename):
    """Возвращает {'data': FeatureCollection} слоя в EPSG:3857 для редактора.

    Читает CSV/WKT в 4326, исключает служебный атрибут wkt и добавляет
    описание CRS. ID объектов остаются в properties; индекс DataFrame
    не становится GeoJSON id. path_src — строка с завершающим разделителем.
    Ошибки чтения/геометрии передаются вызывающему коду.
    """
    geo_df = _load_geo_layer(path_src + filename)
    geo_df.drop(columns=['wkt'], inplace=True)
    geo_df = geo_df.to_crs('epsg:3857')
    json_data = json.loads(geo_df.to_json(drop_id=True))
    json_data["crs"] = {"type": "name", "properties": {"name": "epsg:3857"}}          
    json_full = { 
    "data": json_data
    }
    return json_full


def validate_simple_edges(g_edges):
    """Проверяет уникальный двухуровневый MultiIndex с именами u/v.

    Возвращает None либо выдает GeoValidationError. Проверяет структуру
    индекса, а не наличие столбцов, корректность геометрии или весов.
    """
    if not isinstance(g_edges.index, pd.MultiIndex):
        raise GeoValidationError('Ожидается MultiIndex ребер с уровнями (u, v)')
    if g_edges.index.nlevels != 2 or list(g_edges.index.names) != ['u', 'v']:
        raise GeoValidationError('Ожидается индекс ребер с уровнями (u, v)')
    if not g_edges.index.is_unique:
        duplicate_count = g_edges.index[g_edges.index.duplicated()].nunique()
        raise GeoValidationError(
            f'Граф содержит повторяющиеся пары (u, v): {duplicate_count}'
        )


def simplify_graph_edges(g_edges, g_nodes):
    """Возвращает (edges, nodes) простого графа, сохраняя дороги мультиграфа OSM.

    g_edges имеет индекс (u, v, key), length в метрах и geometry;
    g_nodes содержит геометрии узлов в той же метрической CRS.
    Параллельные дороги сохраняются как цепочки (u, midpoint, v).
    Совпадающие встречные дороги используют общий промежуточный узел;
    остальные дороги соединяются только в исходных узлах. Длины и изгибы
    сохраняются, ID новых узлов выбираются выше максимального исходного ID.
    Результат имеет уникальный индекс (u, v) и сохраняет столбец key.
    Исходные таблицы не изменяются; при отсутствии разрезов g_nodes
    возвращается как есть. Ошибки индекса/весов/разрезаемых линий дают отказ.
    """
    if not isinstance(g_edges.index, pd.MultiIndex):
        raise GeoValidationError('Ожидается MultiIndex ребер OSMnx с уровнями (u, v, key)')
    if g_edges.index.nlevels != 3 or list(g_edges.index.names) != ['u', 'v', 'key']:
        raise GeoValidationError('Ожидается индекс ребер OSMnx с уровнями (u, v, key)')

    required_columns = {'length', 'geometry'}
    missing_columns = required_columns.difference(g_edges.columns)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise GeoValidationError(f'В таблице ребер отсутствуют поля: {missing}')

    edges = g_edges.reset_index()
    edges['length'] = pd.to_numeric(edges['length'], errors='raise')
    if not np.isfinite(edges['length']).all() or edges['length'].lt(0).any():
        raise GeoValidationError('Длины ребер должны быть конечными неотрицательными числами')

    edges = edges.sort_values(['u', 'v', 'length', 'key'], kind='stable').set_index(['u', 'v'])
    edges = gpd.GeoDataFrame(edges, geometry='geometry', crs=g_edges.crs)
    parallel_pairs = edges.index[edges.index.duplicated(keep=False)].unique()
    # Разрезаем также встречные ребра этих пар, чтобы сохранить движение
    # в обе стороны по одной дороге после привязки остановок.
    affected_pairs = set(parallel_pairs) | {(v, u) for u, v in parallel_pairs}
    split_mask = edges.index.isin(affected_pairs)
    if not split_mask.any():
        validate_simple_edges(edges)
        return edges, g_nodes

    to_split = edges.loc[split_mask]
    if (not to_split.geometry.geom_type.eq('LineString').all()
            or to_split.geometry.is_empty.any() or not to_split.geometry.is_valid.all()
            or not np.isfinite(shapely.get_coordinates(to_split.geometry.array)).all()):
        raise GeoValidationError('Для дробления параллельных дорог нужны корректные линии')

    next_node_id = int(g_nodes.index.max()) + 1
    new_nodes, new_edges, unpaired = [], [], {}
    # Каждой дороге сопоставляем не более одного встречного ребра той же геометрии.
    for (source, target), edge in to_split.iterrows():
        geometry = shapely.LineString(_orient_edge_coordinates(
            shapely.get_coordinates(edge.geometry), g_nodes.loc[source].geometry))
        reverse_candidates = unpaired.get((target, source), []) if source != target else []
        match = next((i for i, (other, _, _) in enumerate(reverse_candidates)
                      if geometry.equals(other)), None)
        if match is None:
            midpoint_id = next_node_id
            next_node_id += 1
            new_nodes.append((midpoint_id, geometry.interpolate(geometry.length / 2)))
            rows = _split_edge_at_positions(
                edge, (source, target), geometry, [geometry.length / 2], [midpoint_id])
            unpaired.setdefault((source, target), []).append(
                (geometry, midpoint_id, [row['geometry'] for _, row in rows]))
        else:
            other, midpoint_id, parts = reverse_candidates.pop(match)
            rows = _split_edge_at_positions(
                edge, (source, target), shapely.reverse(other),
                [other.length / 2], [midpoint_id])
            # Общие части в обратном порядке исключают расхождение координат
            # точки разреза из-за округления при интерполяции встречной линии.
            for (_, row), part in zip(rows, reversed(parts)):
                row['geometry'] = shapely.reverse(part)
        new_edges.extend(rows)

    additions = gpd.GeoDataFrame([row for _, row in new_edges],
        index=pd.MultiIndex.from_tuples([pair for pair, _ in new_edges], names=['u', 'v']),
        geometry='geometry', crs=edges.crs)
    edges = gpd.GeoDataFrame(pd.concat([edges.loc[~split_mask], additions]),
                             geometry='geometry', crs=edges.crs)
    additions = gpd.GeoDataFrame(geometry=[point for _, point in new_nodes],
        index=pd.Index([node for node, _ in new_nodes], name=g_nodes.index.name), crs=g_nodes.crs)
    nodes = gpd.GeoDataFrame(pd.concat([g_nodes, additions]),
                             geometry='geometry', crs=g_nodes.crs)
    validate_simple_edges(edges)
    return edges, nodes


def _orient_edge_coordinates(coordinates, source_geometry):
    """Возвращает массив координат в направлении от source_geometry.

    Требуются минимум две координаты. Из двух концов первым становится
    ближайший к исходному узлу; при равенстве сохраняется исходный порядок.
    При развороте создается копия массива; CRS здесь не преобразуется.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise GeoValidationError('Для дробления ребра требуется минимум две координаты')

    first_distance = shapely.Point(coordinates[0]).distance(source_geometry)
    last_distance = shapely.Point(coordinates[-1]).distance(source_geometry)
    if last_distance < first_distance:
        return coordinates[::-1].copy()
    return coordinates


def add_edge(G, source, target, weight):
    """Добавляет направленное ребро в dict G на месте, возвращая None.

    Формат G: {source: {target: weight}}. Вес должен быть конечным
    неотрицательным числом; повторная пара (source, target) запрещена
    и вместе с неверным весом дает GeoValidationError.
    """
    if not np.isfinite(weight) or weight < 0:
        raise GeoValidationError('Вес ребра должен быть конечным неотрицательным числом')
    if source in G and target in G[source]:
        raise GeoValidationError(f'Повторяющееся ребро простого графа: ({source}, {target})')
    G.setdefault(source, {})[target] = weight


def get_G(g_edges):
    """Возвращает словарь смежности {u: {v: length}} из таблицы простого графа.

    Проверяет индекс (u, v); add_edge проверяет веса. Узел без исходящих
    ребер может отсутствовать как ключ — deijkstra учитывает такой случай.
    Геометрия не переносится в словарь; она остается в исходной таблице.
    """
    validate_simple_edges(g_edges)
    G = {}
    for i in g_edges.itertuples():
        add_edge(G, i.Index[0], i.Index[1], i.length)
    return G


def deijkstra(G, start, target=None):
    """Возвращает (predecessors, distances) кратчайших путей от start.

    G — словарь {u: {v: неотрицательный вес}}. Проверку весов выполняет
    создатель графа. predecessors отображает узел в предшественника,
    distances содержит длины без округления; расстояние до start равно 0.
    Без target вычисляются все достижимые узлы. С target поиск заканчивается
    при извлечении цели из очереди; расстояния до остальных обнаруженных
    узлов при этом могут быть предварительными. Недостижимые узлы отсутствуют.
    """
    queue = [(0.0, start)]
    distances = {start: 0.0}
    predecessors = {}

    while queue:
        current_distance, node = heappop(queue)
        # Улучшенный путь оставляет в куче старую запись; ее повторно не разворачиваем.
        if current_distance != distances.get(node):
            continue
        if target is not None and node == target:
            break

        for neighbor, weight in G.get(node, {}).items():
            candidate_distance = current_distance + weight
            if candidate_distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = candidate_distance
                predecessors[neighbor] = node
                heappush(queue, (candidate_distance, neighbor))

    return predecessors, distances


def pathfinder(links, start, end, search_limit):
    """Восстанавливает список узлов start→end по словарю предшественников links.

    Оба конца включены; start == end дает [start]. Отсутствующий предшественник
    или превышение search_limit переходов дает []. Лимит ограничивает число
    ребер при восстановлении, а не физическую длину, и защищает от циклов.
    Для штатного использования search_limit должен быть положительным.
    """
    path = deque([end])
    cur_v = end
    count_br = 0
    while cur_v != start:
        if cur_v not in links:
            return []
        cur_v = links[cur_v]
        path.appendleft(cur_v)
        count_br += 1
        if count_br >= search_limit and cur_v != start:
            app.logger.warning('pathfinder limit reached: start=%s, end=%s, limit=%s',
                               start, end, search_limit)
            return []
    return list(path)
