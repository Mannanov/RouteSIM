import os
from collections import deque
import json
import pandas as pd
import seaborn as sns
import csv
import geopandas as gpd
from serv_funcs import path_gen, load_geo_csv, load_geo_df, select_utm_crs
from routes import session
from rsim import app
import folium
from shapely import concave_hull
from math import exp
import numpy as np


LOS_BASEMAP_TILES = 'OpenStreetMap'
LOS_BASEMAP_STYLE = '''
<style>
.leaflet-tile-pane img.leaflet-tile {
    filter: grayscale(100%) brightness(108%) contrast(85%);
}
</style>
'''


def _make_los_map(location, zoom_start=12, min_zoom=None):
    '''Инициируем карту Фолиум'''
    map_options = {
        'location': location,
        'zoom_start': zoom_start,
        'tiles': LOS_BASEMAP_TILES,
    }
    if min_zoom is not None:
        map_options['min_zoom'] = min_zoom
    los_map = folium.Map(**map_options)
    los_map.get_root().header.add_child(folium.Element(LOS_BASEMAP_STYLE))
    return los_map



def rnd_combi(ver):
    '''
    ver - код проекта
    RND COMBI - пишем в файл комбинации всех возможных связей на маршруте
    массив должен быть отсортирован по маршруртам и последовательности остановок СТРОГО -
    без повторений номеров маршрутов (желательно использовать именно мвн, а не id), без хвостовых заглушек,
    rnd / order / source / target / distance / tr_time / interval / carry / tarif
    Кодировка: UTF-8
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_in = 'rnd' + ver + '.csv'
    name_out = 'rnd_combi' + ver + '.csv'
        
    prebase = {}
    count = 0
    with open(path_src + name_in, 'r', encoding='utf-8', newline='') as f_in:
        reader = csv.reader(f_in, delimiter=";")
        for row in reader:
            if count == 0:
                count += 1
                continue
            prebase[count] = {}
            prebase[count]['rnd'] = row[0]
            prebase[count]['source'] = row[2]
            prebase[count]['target'] = row[3]
            prebase[count]['dist'] = row[4]
            prebase[count]['tr_time'] = row[5]
            prebase[count]['interval'] = row[6]
            prebase[count]['carry'] = row[7]
            prebase[count]['tarif'] = row[8]
            count += 1
            
    with open(path_out + name_out, 'w', encoding='utf-8', newline='') as f:
        f.write('rnd'+";"+'link'+";"+'distance'+";"+'tr_time'+";"+'interval'+";"+'carry'+";"+'tarif'+'\n')
        for i in prebase:
                route = prebase[i]['rnd']
                start_point = prebase[i]['source']
                position = int(i)
                sum_length = 0
                sum_time = 0
                for j in prebase:
                    if route == prebase[j]['rnd']:
                        if int(j) >= position:
                            sum_length += float(prebase[j]['dist'])
                            sum_time += float(prebase[j]['tr_time'])
                            interval = float(prebase[j]['interval'])
                            f.write(route+";"+start_point+">"+prebase[j]['target']+";"+str(round(sum_length, 3))+
                                    ";"+str(round(sum_time, 5))+";"+prebase[j]['interval']+";"+prebase[j]['carry']+";"+prebase[j]['tarif']+'\n')
        f.close()
    
    rnd_combi = pd.read_csv(path_out + name_out, delimiter=';', encoding='utf-8')
    rnd_combi.sort_values(by=['link', 'tr_time'], inplace=True)
    rnd_combi.to_csv(path_out + name_out, sep=';', index=False, encoding='utf-8')


def get_vectors(ver, interval_tresh=25, interval_denom=2, thresh_best_links=0.7, use_reducer=False, reducer_grade=1):
    '''
    Делаем вектора, взвешиваем время в пути на них относительно провозной.
    Добавляем в вектора среднее время ожидания на основе сетевого интервала!
    - массив rnd_combi предварительно сортируем по линкам
    - делаем вектора из rnd_combi: группируем, взвешиваем время в пути на них относительно провозной
    - interval_denom - делитель для сетевого интервала
    - interval_tresh - для больших интервалов, считаем, что пользователи пользуются расписанием, 
    - тогда учитываем отклонение от расписания и выдаем время ожидания = interval_tresh/2 (минуты)
    - thresh_best_links - отбираем лучшие связи на основе порога, например - самый быстрый вариант/рассматриваемый >= 0.7
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_in = 'rnd_combi' + ver + '.csv'
    name_out = 'vectors' + ver + '.csv'
    name_out_short = 'vectors_short' + ver 
    
    with open(path_out + name_in, newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, delimiter=';')
        prev = ''
        base = {}
        count = 0
        labels = 0
        for row in reader:
            if labels == 0:
                labels = 1
                continue
            vector = row[1]
            if vector != prev:
                count = 1
                base[vector] = {}
                base[vector][count] = {} 
                base[vector][count]['rnd'] = row[0] 
                base[vector][count]['data'] = [row[1], row[2], row[3], row[4], row[5], row[6]] 
                prev = vector
                best_time = float(row[3])
            else:
                count += 1
                base[vector][count] = {}
                base[vector][count]['rnd'] = row[0]
                base[vector][count]['data'] = [row[1], row[2], row[3], row[4], row[5], row[6]]
                if best_time > float(row[3]):
                    best_time = row[3]
            base[vector]['best_time'] = best_time

    with open(path_out + name_out, 'w', encoding='utf-8') as f:
        f.write('link'+";"+'tr_time'+";"+'aver_dist'+";"+'net_interval'+";"+'sum_carry'+";"+'min_tarif'+";"+'RNDs'+
                ";"+'source'+";"+'target'+";"+'atr_carrs'+'\n')
        vbase = {}
        for i in base:
            attract = []
            carrs = []
            intervals = []
            tr_times = []
            distances = []
            rnds = []
            best_time = float(base[i]['best_time'])
            min_tarif = 99999
            for j in base[i]:
                if j != 'best_time':
                    link_name = base[i][j]['data'][0]
                    if best_time/float(base[i][j]['data'][2]) >= thresh_best_links:
                        ## if rnd not in rnds - можно добавить условие, чтоб исключить повтор расчета для кластеров,
                        ## когда два и более ОП обслуживаются одним маршрутом 
                        if base[i][j]['rnd'] in rnds: continue # ПРОВЕРИТЬ РАБОТУ!!!
                        rnds.append(base[i][j]['rnd'])
                        if use_reducer == 'Экспоненциальная':
                            # var 1 - нижняя точка примерно равна обычной пропорции, похоже на логарифм угасание
                            # attr_ratio = best_time/float(base[i][j]['data'][2])
                            # a_k = 2.172 + 0.00422*reducer_grade
                            # attr_calc = round(a_k*(attr_ratio**3) - 2.795*attr_ratio + 1.584, 3)
                            # var 2 - сильнее редуцируем плохие
                            # attr_ratio = best_time/float(base[i][j]['data'][2])
                            # a_k = -8.065 + 0.005*reducer_grade
                            # attr_calc = round(a_k*(attr_ratio**3) + 15.85*attr_ratio - 6.83, 3)
                            # attract.append(attr_calc)
                            # var 3
                            reducer_grade_dict = {
                                3: [0.14, 1.96599],
                                2: [0.18, 1.7148],
                                1: [0.25, 1.3863]
                            }
                            attr_ratio = best_time/float(base[i][j]['data'][2])
                            attr_calc = round(reducer_grade_dict[reducer_grade][0]*exp(reducer_grade_dict[reducer_grade][1]*attr_ratio), 3)
                            attract.append(attr_calc)
                        else:
                            attract.append(best_time/float(base[i][j]['data'][2]))
                        carrs.append(float(base[i][j]['data'][4]))
                        intervals.append(float(base[i][j]['data'][3]))
                        tr_times.append(float(base[i][j]['data'][2]))
                        tarif = float(base[i][j]['data'][5])
                        distances.append(float(base[i][j]['data'][1]))

                        if tarif < min_tarif:
                            min_tarif = tarif

            net_interval = 60/sum([60/i for i in intervals])            
            spred = [round(a*c, 1) for a, c in zip(attract, carrs)]
            w_spred = [round(s*t, 3) for s, t in zip(spred, tr_times)]
            d_spred = [round(s*d, 3) for s, d in zip(spred, distances)]
            if net_interval > interval_tresh:
                net_interval = interval_tresh
            fin_tr_time = sum(w_spred)/sum(spred) + (net_interval/interval_denom)/60 
            aver_dist = sum(d_spred)/sum(spred)
            source = int(link_name.split('>')[0])
            target = int(link_name.split('>')[1])
            vbase[link_name] = [rnds, spred]
            f.write(link_name+";"+str(round(fin_tr_time, 3))+";"+str(round(aver_dist, 2))+";"+str(round(net_interval, 2))+
                    ";"+str(sum(spred))+";"+str(min_tarif)+";"+str(rnds)+";"+str(source)+";"+str(target)+";"+str(spred)+'\n')
        f.close()

    with open(path_out + name_out_short, "w", encoding='utf-8') as outfile:
        json.dump(vbase, outfile, ensure_ascii=False)


def add_walk_links_to_vec(ver, dist_trash=0.9, ped_speed=4.5):
    '''
    Добавляем в вектора пешие подходы между ближайшими остановками
    dist_trash - предельное расстояние для пешего подхода к ОП (км) / treshhold for max acceptable walk distance (km)
    ped_speed - средняя скорость пешехода / pedestrian speed (km/h)

    Для каждой связи source-target сохраняем вариант с минимальным временем.
    При равном времени пешая связь имеет приоритет перед транспортной.
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_in_matrix = 'matrix_dist_stops' + ver + '.csv'
    name_in_vectors = 'vectors' + ver + '.csv'
    name_out_vectors = 'vectors_w' + ver + '.csv'
               
    matrix_walk_stops = pd.read_csv(path_src + name_in_matrix, delimiter=';', encoding='utf-8')
    matrix_walk_stops = matrix_walk_stops[matrix_walk_stops.distance <= dist_trash]
    
    matrix_walk_stops['distance'] = np.where(matrix_walk_stops['distance'] == 0, 0.01, matrix_walk_stops['distance'])
    matrix_walk_stops['tr_time'] = round( matrix_walk_stops['distance']/ped_speed, 5)
    matrix_walk_stops['net_interval'] = 0
    matrix_walk_stops['sum_carry'] = 0
    matrix_walk_stops['min_tarif'] = 0
    matrix_walk_stops['atr_carrs'] = '[0]'
    matrix_walk_stops['RNDs'] = 'W'
    matrix_walk_stops = matrix_walk_stops.rename(columns={"distance": "aver_dist"})
    
    vectors = pd.read_csv(path_out + name_in_vectors, delimiter=';', encoding='utf-8')
    vectors = pd.concat([vectors, matrix_walk_stops], ignore_index=True)
    vectors['_walk_priority'] = np.where(vectors['RNDs'] == 'W', 0, 1)
    result = (vectors
              .sort_values(['link', 'tr_time', '_walk_priority'], kind='stable')
              .drop_duplicates(subset=['link'], keep='first')
              .drop(columns=['_walk_priority']))
    result.to_csv(path_out + name_out_vectors, sep=';', index=False, encoding='utf-8') #compression='zip'


def get_G_vec(ver):
    grf = {}
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_vectors = 'vectors_w' + ver + '.csv'
    vectors = pd.read_csv(path_out + name_vectors, delimiter=';', encoding='utf-8')
    def add_edge(source, target):
        if source not in grf:
            grf[source] = [target]
        else:
            grf[source].append(target)
    for i in vectors.itertuples():               
        add_edge(i.source, i.target)
    return grf


def bfs_pathfinder(graph, start, target, deep_limit):
    result = []

    def dfs(current, path):
        if current == target:
            result.append(path[:])
            return

        if len(path) - 1 >= deep_limit:
            return

        for neighbor in graph.get(current, []):
            if neighbor == start:
                continue

            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()

    dfs(start, [start])
    return result


def vec_w_to_dict(ver):
    v_dict = {}
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_vectors = 'vectors_w' + ver + '.csv'
    vectors = pd.read_csv(path_out + name_vectors, delimiter=';', encoding='utf-8')
    
    for i in vectors.itertuples():
        v_dict[i.link] = [i.tr_time, i.aver_dist, i.sum_carry, i.RNDs, i.atr_carrs]
    return v_dict


def los_computate(v_dict, all_paths, change_penalty=3, best_tresh=0.85, change_tolerance=0.1):
    '''
    Рассчитываем итоговый ЛОС
    change_penalty - штраф за пересадку (минуты), по умолчанию 3, рекомендуемый диапазон 1 - 5 минут, допустимо до 10 минут
    best_tresh - порог отсечки худших связей (по умолчанию не более 20% от лучшей)
    change_tolerance - пока по умолчанию 0.1, хз так подобралось..
    тариф пока не отрабатываем, нужен ли, вопрос?
    v_dict = [0 - i.tr_time, 1 - i.aver_dist, 2 - i.sum_carry, 3 - i.RNDs, 4 - i.atr_carrs]
    los_carry - максимальная провозная способность одного варианта пути
    '''
    change_penalty = change_penalty/60
    
    all_vars = []
    for path in all_paths:
        chain = []
        for i in range(len(path)):
            if i+1 < len(path):
                chain.append(str(path[i]) + '>' + str(path[i+1]))
        all_vars.append(chain)
    
    tr_times = []
    carrs = []
    distances = []
    best_time = 9999
    route_changes = []
    for i in all_vars:
        var_tr_time = 0
        var_distance = 0
        min_carry = 0
        count_changes = -1
        # link_with_change = 1
        for j in i:
            var_tr_time += v_dict[j][0]
            var_distance += v_dict[j][1]
            if min_carry == 0:
                min_carry = v_dict[j][2] 
            else:
                if min_carry > v_dict[j][2] and v_dict[j][2] != 0:
                    min_carry = v_dict[j][2]
            if v_dict[j][3] != 'W':
                count_changes += 1
        if count_changes == -1:
            count_changes = 0
        var_tr_time += count_changes*change_penalty
        tr_times.append(var_tr_time)
        distances.append(var_distance)
        carrs.append(min_carry)
        route_changes.append(count_changes)
        
        if var_tr_time < best_time: best_time = var_tr_time
        
    var_quality = [best_time/t for t in tr_times]
    bad_vars = []
    for idx, val in enumerate(var_quality):
        if val < best_tresh:
            bad_vars.append(idx)
    count = 0
    for idx in bad_vars:
        del all_vars[idx-count]
        del tr_times[idx-count]
        del carrs[idx-count]
        del distances[idx-count]
        del route_changes[idx-count]
        count += 1
      
    if sum(carrs) != 0:
        los = round(sum([c*t for c, t in zip(carrs, tr_times)])/sum(carrs), 3)
        los_distance = round(sum([c*d for c, d in zip(carrs, distances)])/sum(carrs), 3)
        los_speed = round(los_distance/los, 3)
        los_carry = max(carrs)
        # fin_changes = sum(route_changes)/len(route_changes)
        fin_changes = round(sum([c*rc for c, rc in zip(carrs, route_changes)])/sum(carrs), 3)
        if fin_changes < change_tolerance:
            fin_changes = 0
            
        return (all_vars, los, round(best_time, 3), los_speed, los_distance, los_carry, carrs, fin_changes)       
    else:
        # Полностью пешая связь не является поездкой на пассажирском транспорте
        # и не участвует в транспортной оценке LOS.
        return ('W', 0, 0, 0, min(distances), 0, [0], 0)


def all_links_los(grf, v_dict, ver='', best_tresh=0.85, deep_limit=2, change_penalty=3):
    '''
    Расчет ЛОС для всех возможных комбинаций между ОП/кластерами
    deep_limit - кол-во пересадок / number of changes
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_matrix = 'matrix_dist_stops' + ver + '.csv'
    name_los_out = 'los' + ver + '.csv'
    name_los_short = 'los_short' + ver
        
    with open(path_out + name_los_out, 'w', encoding='utf-8', newline='') as f:
        f.write('link'+";"+"source"+";"+"target"+";"+'all_vars'+";"+'los'+";"+
                'best_time'+";"+'los_speed'+";"+'los_distance'+";"+'los_carry'+";"+'all_carrs'+";"+'aver_changes'+'\n')
        
        matrix_stops = pd.read_csv(path_src + name_matrix, delimiter=';', encoding='utf-8')        
        # Для прогресс бара
        # delim = round(matrix_stops.shape[0]/100, 0) - 1
        # idx_steps = []
        # percent_steps = []
        
        # for i in range(99):
        #     percent_steps.append(i)
        #     idx_steps.append((i+1)*delim)
            
        los_short = {}
        for l in matrix_stops.itertuples():
            all_paths = bfs_pathfinder(grf, l.source, l.target, deep_limit)
            if all_paths != []:
                los_descr = los_computate(v_dict, all_paths, best_tresh=best_tresh, change_penalty=change_penalty)
              
                f.write(l.link+";"+str(l.source)+";"+str(l.target)+";"+str(los_descr[0])+";"+str(los_descr[1])+";"+
                        str(los_descr[2])+";"+str(los_descr[3])+";"+str(los_descr[4])+";"+str(los_descr[5])+";"
                        +str(los_descr[6])+";"+str(los_descr[7])+'\n')
                los_short[l.link] = [los_descr[0], los_descr[6]]
        f.close()
    
    with open(path_out + name_los_short, "w", encoding='utf-8') as outfile:
        json.dump(los_short, outfile, ensure_ascii=False)


def los_agg(ver, los_agg_type):
    '''
    Функция агрегации ЛОСа до кластерных и транспортных районов
    los_agg_type - агрегация по транспортным районам (tr_district), кластерным районам (сl_district), кластерам (cluster)
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_ref_stops = 'ref_stops' + ver + '.csv'
    name_los = 'los' + ver + '.csv'
    name_agg_out = 'los_agg_' + los_agg_type + ver + '.csv'

    ref_stops = pd.read_csv(path_src + name_ref_stops, delimiter=';', encoding='utf-8')
    los = pd.read_csv(path_out + name_los, delimiter=';', encoding='utf-8', 
                        dtype = {'link': 'object', 'source': 'int32', 
                               'target': 'int32', 'all_vars': 'object', 
                               'los': 'float32', 'best_time': 'float32',
                               'los_speed': 'float32', 'los_distance': 'float32',
                               'los_carry': 'float32', 'los_distance': 'float32',
                               'aver_changes': 'float32'})
        
    ref_stops = ref_stops[['id_stop', los_agg_type]]
    ref_stops.rename(columns={"id_stop": "source", los_agg_type: "agg_source"}, inplace=True)
    los = los.merge(ref_stops, how='left', on='source')
    ref_stops.rename(columns={"source": "target", "agg_source": "agg_target"}, inplace=True)
    los = los.merge(ref_stops, how='left', on='target')
    los = los[los['agg_source']!=los['agg_target']]
    los = los[los['all_vars']!='W']
    los['weight_los'] = los['los']*los['los_carry']
    los['weight_spd'] = los['los_speed']*los['los_carry']
    los['weight_changes'] = los['aver_changes']*los['los_carry']
    los_agg = los.groupby(['agg_source', 'agg_target'])[['weight_los', 'weight_spd', 'weight_changes', 'los_carry']].sum()
    los_agg['los'] = los_agg['weight_los']/los_agg['los_carry']
    los_agg['los_speed'] = los_agg['weight_spd']/los_agg['los_carry']
    los_agg['los_changes'] = los_agg['weight_changes']/los_agg['los_carry']
    los_agg.drop(labels=['weight_spd', 'weight_los', 'weight_changes'], axis='columns', inplace=True)
    los_agg.to_csv(path_out + name_agg_out, sep=';', encoding='utf-8') #, compression='zip'


def los_agg_allcalc(ver=''):
    los_agg_types = ['tr_district', 'cluster'] #'cl_district'
    for agt in los_agg_types:
        los_agg(ver, agt)


def flow_distrib(ver, period='day'):
    '''
    Функция расчета распределения потоков по маршрутам за различные периоды времени (по умолчанию - сутки).
    Итерируем по линкам в flow, распределяем по маршрутам на основе all_carrs.
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_flow = 'flow' + ver + '.csv'
    name_los_short = 'los_short' + ver
    name_vec_short = 'vectors_short' + ver
    flow_full_out = 'flow_vecs_and_routes' + ver + '.csv'
    flow_short_out = 'flow_routes' + ver + '.csv'
    flow_stats_out = 'flow_distribution_stats' + ver + '.csv'
    unserved_correspondences_out = 'unserved_correspondences' + ver + '.csv'

    flow = pd.read_csv(path_src + name_flow, delimiter=';', encoding='utf-8')
    if period not in flow.columns[4:]:
        # print(f'Rsim: ошибочный параметр {period}')
        return None
    else:
        flow = flow[flow['source']!=flow['target']]
        flow = flow[['link', 'source', 'target', period]]
        flow = flow.rename(columns={period: "period"})

    with open(path_out + name_los_short, 'r', encoding='utf-8') as openfile:
        los = json.load(openfile)

    with open(path_out + name_vec_short, 'r', encoding='utf-8') as openfile:
        vectors = json.load(openfile)

    all_trips = flow['period'].sum()
    unserved_correspondences = flow[~flow['link'].isin(los)]
    demand_without_los = unserved_correspondences['period'].sum()
    demand_with_los = all_trips - demand_without_los
    share_without_los = (demand_without_los/all_trips*100) if all_trips > 0 else 0
    unserved_correspondences_export = (
        unserved_correspondences.loc[
            unserved_correspondences['period'] > 0,
            ['source', 'target', 'period'],
        ]
        .rename(columns={'period': 'flow'})
        .groupby(['source', 'target'], as_index=False)['flow'].sum()
        .sort_values(['source', 'target'])
    )
    unserved_correspondences_export.to_csv(
        path_out + unserved_correspondences_out,
        sep=';',
        index=False,
        encoding='utf-8',
    )

    flow_on_vectors = {}
    for f in flow.itertuples():
        if f.link in los:
            if los[f.link][0] == 'W': continue 
            variant_sum_carry = sum(los[f.link][1])
            for i in range(len(los[f.link][0])): 
                variant_part_flow = (f.period*los[f.link][1][i])/variant_sum_carry
                for j in los[f.link][0][i]:
                    if j in flow_on_vectors:
                        flow_on_vectors[j] += variant_part_flow
                    else:
                        flow_on_vectors[j] = variant_part_flow

    flow_on_rnds = {}
    count = 0
    for v in flow_on_vectors:
        if v not in vectors:
            continue
        rnds_sum_carry = sum(vectors[v][1])
        for i in range(len(vectors[v][0])):
            rnd_part_flow = flow_on_vectors[v]*vectors[v][1][i]/rnds_sum_carry
            flow_on_rnds[count] = [v, vectors[v][0][i], rnd_part_flow]
            count += 1
    flow_on_rnds_df = pd.DataFrame.from_dict(flow_on_rnds, orient='index', columns=['vector', 'rnd', 'flow'])
    
    distributed_transport_flow = flow_on_rnds_df['flow'].sum()
    distributed_flow_to_demand_ratio = (
        distributed_transport_flow/demand_with_los if demand_with_los > 0 else 0
    )

    flow_stats = pd.DataFrame([{
        'period': period,
        'total_demand': all_trips,
        'demand_with_los': demand_with_los,
        'demand_without_los': demand_without_los,
        'share_without_los': share_without_los,
        'unserved_correspondences_count': len(unserved_correspondences_export),
        'distributed_transport_flow': distributed_transport_flow,
        'distributed_flow_to_demand_ratio': distributed_flow_to_demand_ratio,
    }])
    flow_stats.to_csv(path_out + flow_stats_out, sep=';', index=False, encoding='utf-8')
    
    # flow_on_rnds_df['route'] = flow_on_rnds_df['rnd'].apply(lambda x: x.split('-')[0]) - OPTIMIZE!
    flow_on_rnds_df['route'] = flow_on_rnds_df['rnd'].str.split('-').str[0]
    flow_on_vecs_and_routes = flow_on_rnds_df.groupby(['vector', 'route'])['flow'].sum().reset_index()
    
    # flow_on_vecs_and_routes['source'] = flow_on_vecs_and_routes['vector'].apply(lambda x: x.split('>')[0]) - OPTIMIZE
    # flow_on_vecs_and_routes['target'] = flow_on_vecs_and_routes['vector'].apply(lambda x: x.split('>')[1])
    flow_on_vecs_and_routes['source'] = flow_on_vecs_and_routes['vector'].str.split('>').str[0]
    flow_on_vecs_and_routes['target'] = flow_on_vecs_and_routes['vector'].str.split('>').str[1]

    
    flow_period_name = f'flow_{period}'
    flow_on_vecs_and_routes.rename(columns={'flow': flow_period_name, 'vector': 'link'}, inplace=True)
    flow_on_vecs_and_routes.to_csv(path_out + flow_full_out, sep=';', index=False, encoding='utf-8') #, compression='zip'
    
    flow_routes = flow_on_vecs_and_routes.groupby(['route'])[flow_period_name].sum().reset_index()
    flow_routes.to_csv(path_out + flow_short_out, sep=';', index=False, encoding='utf-8') #, compression='zip'


def add_walk_to_stop(ver, max_dist_walk):
    '''
    в качестве длины пешего подхода взять заданный радиус пешей доступности ОП и умножить на 0.6 
    (делим радиус на 2 (распределение домов вдоль радиуса) и ужножить на коэфф. непрямолинейности 1.2)
    аналогично в отношении add_distance (radius out of reach)
    max_dist_walk - в километрах
    '''
    # max_dist_walk = float(max_dist_walk)
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    max_dist_walk_meters = max_dist_walk*1000
    name_stops_in = 'geom_stops' + ver + '.csv'
    name_tr_district_in = 'geom_tr_distr' + ver + '.csv'
    name_uu_stop_out = 'geom_uu_stop' + ver + '.csv'
    name_geom_area_not_available_out = 'geom_area_not_available' + ver + '.csv'
    walk_dist_out = 'walk_dist' + ver + '.csv'
    
    geo_stop = load_geo_csv(path_src + name_stops_in)
    geo_stop.drop(columns=['wkt'], inplace=True)
    metric_crs = select_utm_crs(geo_stop)
    geo_stop = geo_stop.to_crs(metric_crs)
    geo_stop['geometry'] = geo_stop['geometry'].buffer(distance=max_dist_walk_meters)

    uu_stop = gpd.GeoDataFrame(geometry=[geo_stop.geometry.union_all()], crs=metric_crs)

    geo_district = load_geo_csv(path_src + name_tr_district_in)
    geo_district.drop(columns=['wkt'], inplace=True)
    geo_district = geo_district.to_crs(metric_crs)
    geo_district['tr_area'] = round(geo_district.area/1000**2, 3)

    # пешая доступность для плотно покрытых остановками районов
    stops_overlay = geo_stop.overlay(geo_district, how='intersection')
    stops_overlay['stop_area'] = round(stops_overlay.area/1000**2, 3)
    stops_overlay = stops_overlay.dissolve(by="id_tr_distr", aggfunc={"stop_area": "sum"})
    stops_overlay = stops_overlay.merge(geo_district, how='left', on='id_tr_distr')
    stops_overlay['area_ratio'] = (stops_overlay['tr_area']**2)/stops_overlay['stop_area']
    stops_overlay['walk_distance_som'] = ((stops_overlay['area_ratio']/3.14)**.5)*1.1
    stops_overlay = pd.DataFrame(stops_overlay[['id_tr_distr', 'walk_distance_som']]) 

    area_available = geo_district.overlay(uu_stop, how='intersection')
    area_available['area_av'] = round(area_available.area/1000**2, 3)

    area_not_available = geo_district.overlay(uu_stop, how='difference')
    area_not_available['area_not_av'] = round(area_not_available.area/1000**2, 3)

    area_available_by_district = (
        pd.DataFrame(area_available[['id_tr_distr', 'area_av']])
        .groupby('id_tr_distr', as_index=False)['area_av']
        .sum()
    )
    geo_fin = pd.DataFrame(geo_district[['id_tr_distr', 'tr_area']])
    geo_fin = geo_fin.merge(area_available_by_district, how='left', on='id_tr_distr')
    geo_fin['area_av'] = geo_fin['area_av'].fillna(0)
    # умножаем на коэф. непрямолинейности и делим на 2, т.к. принимаем равномерное распределение подходов, отностиельно радиусв доступности
    k_nonstraight = 1.15/2 
    geo_fin['delta_area'] = (geo_fin['tr_area'] - geo_fin['area_av']).clip(lower=0)
    geo_fin['addit_walk'] = (geo_fin['delta_area']/3.14)**.5
    # geo_fin['addit_walk'] = geo_fin['addit_walk'].apply(lambda x: 5 if x > 5 else x) # исключаем кривые тр. районы - OPTIMIZE
    geo_fin['addit_walk'] = np.where(geo_fin['addit_walk'] > 5, 5, geo_fin['addit_walk'])
    geo_fin['walk_distance_anam'] = (geo_fin['addit_walk'] + max_dist_walk)*k_nonstraight
    geo_fin['walk_distance_anam'] = geo_fin['walk_distance_anam'].fillna(max_dist_walk)
    geo_fin = geo_fin.merge(stops_overlay, how='left', on='id_tr_distr')
    # f = lambda x : x.walk_distance_anam if x.walk_distance_som > 3 else x.walk_distance_som - OPTIMIZE
    # geo_fin['walk_distance'] = geo_fin.apply(f, axis=1).round(4)
    use_anamorphosis = (
        geo_fin['walk_distance_som'].isna()
        | (geo_fin['walk_distance_som'] > 3)
    )
    geo_fin['walk_distance'] = np.where(
        use_anamorphosis,
        geo_fin['walk_distance_anam'],
        geo_fin['walk_distance_som'],
    ).round(4)

    uu_stop = uu_stop.to_crs('epsg:4326').rename_geometry('wkt')
    uu_stop.to_csv(path_out + name_uu_stop_out, sep=';', encoding='utf-8') #, compression='zip'
    # uu_stop.to_file(path_out + name_uu_stop_out, driver='GeoJSON')
    area_not_available.rename_geometry('wkt', inplace=True)
    area_not_available = area_not_available.to_crs('epsg:4326')
    area_not_available.to_csv(path_out + name_geom_area_not_available_out, sep=';', index=False, encoding='utf-8')
    # area_not_available.to_file(path_out + name_geom_area_not_available_out, driver='GeoJSON')

    geo_fin['max_dist_walk'] = max_dist_walk
    df = pd.DataFrame(geo_fin[['id_tr_distr', 'walk_distance']])
    df = df.astype({'id_tr_distr': 'int64', 'walk_distance': 'float32'})
    df.to_csv(path_out + walk_dist_out, sep=';', index=False, encoding='utf-8')
    return df


def indicators_calc_flow(ver, agg_type, period, ped_speed, max_dist_walk):
    '''
    Расчет интегральных показателей с взвешиванием по исходной матрице корреспонденций
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    max_dist_walk = float(max_dist_walk) # km
    ped_speed = float(ped_speed)
    name_ref_stops = 'ref_stops' + ver + '.csv'
    name_flow_in = 'flow' + ver + '.csv'
    name_los_in = 'los' + ver + '.csv'
    name_los_agg_in = 'los_agg_' + agg_type + ver + '.csv'
    name_los_agg_out = 'los_agg_by_flow_' + agg_type + ver + '.csv'
    name_los_agg_short_out = 'los_agg_by_flow_short_' + agg_type + ver + '.csv'
    flow_period_name = f'flow_{period}'
    ref_stops = pd.read_csv(path_src + name_ref_stops, delimiter=';', encoding='utf-8')
    los_agg = pd.read_csv(path_out + name_los_agg_in, delimiter=';', encoding='utf-8')
    flow = pd.read_csv(path_src + name_flow_in, delimiter=';', encoding='utf-8')
    los = pd.read_csv(
        path_out + name_los_in,
        delimiter=';',
        encoding='utf-8',
        usecols=['source', 'target', 'all_vars'],
    )

    if period not in flow.columns:
        raise ValueError(f'Период {period} отсутствует в OD-матрице {name_flow_in}')

    flow = flow.loc[
        flow['source'] != flow['target'],
        ['source', 'target', period],
    ].rename(columns={period: flow_period_name})

    transport_los = (
        los.loc[los['all_vars'] != 'W', ['source', 'target']]
        .drop_duplicates()
    )
    flow = flow.merge(transport_los, how='inner', on=['source', 'target'])

    ref_stops = ref_stops[['id_stop', agg_type]]
    ref_stops.rename(columns={'id_stop': 'source'}, inplace=True)
    flow = flow.merge(ref_stops, how='left', on='source')
    flow.rename(columns={agg_type: 'agg_source'}, inplace=True)
    ref_stops.rename(columns={'source': 'target'}, inplace=True)
    flow = flow.merge(ref_stops, how='left', on='target')
    flow.rename(columns={agg_type: 'agg_target'}, inplace=True)
    flow = flow[['agg_source', 'agg_target', flow_period_name]]
    flow = flow[flow['agg_source'] != flow['agg_target']]
    flow = flow.groupby(['agg_source', 'agg_target'], as_index=False)[flow_period_name].sum().round(2)
    los_agg = los_agg.merge(flow, how='left', on=['agg_source', 'agg_target'])
    los_agg[flow_period_name] = los_agg[flow_period_name].fillna(0)
    los_agg_short = los_agg.copy()

    los_agg['weight_los'] = los_agg['los']*los_agg[flow_period_name]
    los_agg['weight_los_speed'] = los_agg['los_speed']*los_agg[flow_period_name]
    los_agg['weight_changes'] = los_agg['los_changes']*los_agg[flow_period_name]
    los_agg = los_agg.groupby(['agg_source', 'agg_target']).agg({'weight_los': 'sum', 'weight_los_speed': 'sum',  
                                'weight_changes': 'sum', flow_period_name: 'sum', 'los_carry': 'min'})
    los_agg['los_by_flow'] = los_agg['weight_los']/los_agg[flow_period_name]
    los_agg['los_spd_by_flow'] = los_agg['weight_los_speed']/los_agg[flow_period_name]
    los_agg['changes_by_flow'] = los_agg['weight_changes']/los_agg[flow_period_name]

    total_los_changes = los_agg['changes_by_flow'].sum()
    total_flow = los_agg[flow_period_name].sum()
    los_agg = los_agg[los_agg[flow_period_name]>0]
    integral_los = los_agg['weight_los'].sum()/total_flow
    integral_los_speed = los_agg['weight_los_speed'].sum()/total_flow
    # integral_los_changes = total_flow/(total_flow - total_los_changes) + 1
    integral_los_changes = los_agg['weight_changes'].sum()/total_flow + 1
    los_agg.drop(labels=['weight_los', 'weight_los_speed', 'weight_changes'], axis='columns', inplace=True)
    los_agg = los_agg.round(3)
    los_agg.to_csv(path_out + name_los_agg_out, sep=';', encoding='utf-8')
    # агрегируем только до исходящих районов
    los_agg_short['weight_los'] = los_agg_short['los']*los_agg_short[flow_period_name]
    los_agg_short['weight_los_speed'] = los_agg_short['los_speed']*los_agg_short[flow_period_name]
    los_agg_short['weight_changes'] = los_agg_short['los_changes']*los_agg_short[flow_period_name]
    # los_agg_short.astype({'id_tr_distr': 'int64'})
    los_agg_short['agg_source'] = los_agg_short['agg_source'].astype('int64')
    # добавляем пешие подходы только для агрегированных тр. районов
    calc_walk = False
    if max_dist_walk > 0 and agg_type == 'tr_district':
        calc_walk = True
        walk_analyze = add_walk_to_stop(ver, max_dist_walk)
        walk_analyze.rename(columns={'id_tr_distr': 'agg_source'}, inplace=True)
        los_agg_short = los_agg_short.merge(walk_analyze, how='left', on='agg_source')
        los_agg_short['walk_distance'] = los_agg_short['walk_distance'].fillna(max_dist_walk/2) # заполняем пробелы, если они есть
        los_agg_short['weight_w_distance'] = los_agg_short['walk_distance']*los_agg_short[flow_period_name]
        los_agg_short = los_agg_short.groupby(['agg_source'])[['weight_los', 
                                            'weight_los_speed', 'weight_changes', 'weight_w_distance', flow_period_name]].sum()
    else:
        los_agg_short = los_agg_short.groupby(['agg_source'])[['weight_los', 
                                            'weight_los_speed', 'weight_changes', flow_period_name]].sum()

    los_agg_short['los_by_flow'] = los_agg_short['weight_los']/los_agg_short[flow_period_name]
    los_agg_short['los_spd_by_flow'] = los_agg_short['weight_los_speed']/los_agg_short[flow_period_name]
    los_agg_short['changes_by_flow'] = los_agg_short['weight_changes']/los_agg_short[flow_period_name]

    if calc_walk:
        los_agg_short['walk_distance'] = los_agg_short['weight_w_distance']/los_agg_short[flow_period_name]
        los_agg_short['los_with_walk'] = los_agg_short['los_by_flow'] + los_agg_short['walk_distance']/ped_speed
        los_agg_short['los_spd_with_walk'] = (los_agg_short['los_by_flow']*los_agg_short['los_spd_by_flow'])/los_agg_short['los_with_walk']
        los_agg_short['weight_los_with_walk'] = los_agg_short['los_with_walk']*los_agg_short[flow_period_name]
        los_agg_short['weight_los_spd_with_walk'] = los_agg_short['los_spd_with_walk']*los_agg_short[flow_period_name]
        integral_los_with_walk = los_agg_short['weight_los_with_walk'].sum()/total_flow
        integral_los_speed_with_walk = los_agg_short['weight_los_spd_with_walk'].sum()/total_flow
        los_agg_short = los_agg_short[los_agg_short[flow_period_name]>0]
        los_agg_short.drop(labels=['weight_los', 'weight_los_speed', 'weight_changes', flow_period_name, 'weight_w_distance', 
                            'weight_los_with_walk', 'weight_los_spd_with_walk'], axis='columns', inplace=True)
        los_agg_short = los_agg_short.round(3)
        los_agg_short.to_csv(path_out + name_los_agg_short_out, sep=';', encoding='utf-8') #, compression='zip'
        return integral_los, integral_los_speed, integral_los_changes, integral_los_with_walk, integral_los_speed_with_walk
    else:
        los_agg_short = los_agg_short[los_agg_short[flow_period_name]>0]
        los_agg_short.drop(labels=['weight_los', 'weight_los_speed', 'weight_changes', flow_period_name], axis='columns', inplace=True)
        los_agg_short = los_agg_short.round(3)
        los_agg_short.to_csv(path_out + name_los_agg_short_out, sep=';', encoding='utf-8') #, compression='zip'
        return integral_los, integral_los_speed, integral_los_changes


def indicators_calc_carry(ver, agg_type, ped_speed, max_dist_walk):
    '''
    Функция расчета интегральных показателей на основе потока: общесетевые среднее время и скорость поездки
    Периода здесь нет, так как в rnd изначально содержит значение только по одному исследуемому периоду!
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    max_dist_walk = float(max_dist_walk) # km
    ped_speed = float(ped_speed)
    name_los_agg_in = 'los_agg_' + agg_type + ver + '.csv'
    name_los_agg_out = 'los_agg_by_carry_' + agg_type + ver + '.csv'
    name_los_agg_short_out = 'los_agg_by_carry_short_' + agg_type + ver + '.csv'
    los_agg = pd.read_csv(path_out + name_los_agg_in, delimiter=';', encoding='utf-8')
    los_agg_short = los_agg.copy()

    los_agg['weight_los'] = los_agg['los']*los_agg['los_carry']
    los_agg['weight_los_speed'] = los_agg['los_speed']*los_agg['los_carry']
    los_agg['weight_changes'] = los_agg['los_changes']*los_agg['los_carry']
    los_agg = los_agg.groupby(['agg_source', 'agg_target'])[['weight_los', 'weight_los_speed', 'weight_changes', 'los_carry']].sum()
    los_agg['los_by_carry'] = los_agg['weight_los']/los_agg['los_carry']
    los_agg['los_spd_by_carry'] = los_agg['weight_los_speed']/los_agg['los_carry']
    los_agg['changes_by_carry'] = los_agg['weight_changes']/los_agg['los_carry']
    los_agg = los_agg[los_agg['los_carry']>0]
    total_los_changes = los_agg['changes_by_carry'].sum()
    total_carry = los_agg['los_carry'].sum()

    integral_los = los_agg['weight_los'].sum()/total_carry
    integral_los_speed = los_agg['weight_los_speed'].sum()/total_carry
    # integral_los_changes = total_carry/(total_carry - total_los_changes) + 1
    integral_los_changes = los_agg['weight_changes'].sum()/total_carry + 1
    los_agg.drop(labels=['weight_los', 'weight_los_speed', 'weight_changes'], axis='columns', inplace=True)
    los_agg = los_agg.round(3)
    los_agg.to_csv(path_out + name_los_agg_out, sep=';', encoding='utf-8') #, compression='zip'
    # агрегируем только до исходящих районов
    los_agg_short['weight_los'] = los_agg_short['los']*los_agg_short['los_carry']
    los_agg_short['weight_los_speed'] = los_agg_short['los_speed']*los_agg_short['los_carry']
    los_agg_short['weight_changes'] = los_agg_short['los_changes']*los_agg_short['los_carry']
    los_agg_short['agg_source'] = los_agg_short['agg_source'].astype('int64')
    # добавляем пешие подходы только для агрегированных тр. районов
    calc_walk = False
    if max_dist_walk > 0 and agg_type == 'tr_district':
        calc_walk = True
        walk_analyze = add_walk_to_stop(ver, max_dist_walk)
        walk_analyze.rename(columns={'id_tr_distr': 'agg_source'}, inplace=True)
        los_agg_short =  los_agg_short.merge(walk_analyze, how='left', on='agg_source')
        los_agg_short['walk_distance'] = los_agg_short['walk_distance'].fillna(max_dist_walk/2) # заполняем пробелы, если они есть
        los_agg_short['weight_w_distance'] = los_agg_short['walk_distance']*los_agg_short['los_carry']
        los_agg_short = los_agg_short.groupby(['agg_source'])[['weight_los', 
                                            'weight_los_speed', 'weight_changes', 'weight_w_distance', 'los_carry']].sum()
    else:
        los_agg_short = los_agg_short.groupby(['agg_source'])[['weight_los', 
                                            'weight_los_speed', 'weight_changes', 'los_carry']].sum()

    los_agg_short['los_by_carry'] = los_agg_short['weight_los']/los_agg_short['los_carry']
    los_agg_short['los_spd_by_carry'] = los_agg_short['weight_los_speed']/los_agg_short['los_carry']
    los_agg_short['changes_by_carry'] = los_agg_short['weight_changes']/los_agg_short['los_carry']

    if calc_walk:
        los_agg_short['walk_distance'] = los_agg_short['weight_w_distance']/los_agg_short['los_carry']
        los_agg_short['los_with_walk'] = los_agg_short['los_by_carry'] + los_agg_short['walk_distance']/ped_speed
        los_agg_short['los_spd_with_walk'] = (los_agg_short['los_by_carry']*los_agg_short['los_spd_by_carry'])/los_agg_short['los_with_walk']
        los_agg_short['weight_los_with_walk'] = los_agg_short['los_with_walk']*los_agg_short['los_carry']
        los_agg_short['weight_los_spd_with_walk'] = los_agg_short['los_spd_with_walk']*los_agg_short['los_carry']
        integral_los_with_walk = los_agg_short['weight_los_with_walk'].sum()/total_carry
        integral_los_speed_with_walk = los_agg_short['weight_los_spd_with_walk'].sum()/total_carry
        los_agg_short = los_agg_short[los_agg_short['los_by_carry']>0]
        los_agg_short.drop(labels=['weight_los', 'weight_los_speed', 'weight_changes', 'los_carry', 'weight_w_distance',
                             'weight_los_with_walk', 'weight_los_spd_with_walk'], axis='columns', inplace=True)
        los_agg_short = los_agg_short.round(3)
        los_agg_short.to_csv(path_out + name_los_agg_short_out, sep=';', encoding='utf-8') #, compression='zip'
        return integral_los, integral_los_speed, integral_los_changes, integral_los_with_walk, integral_los_speed_with_walk
    else:
        los_agg_short = los_agg_short[los_agg_short['los_by_carry']>0]
        los_agg_short.drop(labels=['weight_los', 'weight_los_speed', 'weight_changes', 'los_carry'], axis='columns', inplace=True)
        los_agg_short = los_agg_short.round(3)
        los_agg_short.to_csv(path_out + name_los_agg_short_out, sep=';', encoding='utf-8') #, compression='zip'
        return integral_los, integral_los_speed, integral_los_changes


def indicators_agg_allcalc_flow(ver='', period='day', ped_speed=0, max_dist_walk=0):
    '''
    Расчет индикаторов для кластеров и тр. районов на основе потоков
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_out = 'indicators_agg_flow' + ver + '.csv'
    los_agg_types = ['tr_district', 'cluster']
    with open(path_out + name_out, 'w', encoding='utf-8', newline='') as f:
        f.write('los'+';'+'los_speed'+';'+'los_changes'+';'+'los_with_walk'+';'+'los_speed_with_walk'+'\n')
        for agt in los_agg_types:
            if max_dist_walk > 0 and agt == 'tr_district':
                i_los, i_los_speed, i_los_changes, i_los_with_walk, i_los_speed_with_walk = indicators_calc_flow(ver, agt, period, ped_speed, max_dist_walk)
                f.write(str(i_los)+';'+str(i_los_speed)+';'+str(round(i_los_changes, 2))+';'+str(i_los_with_walk)+';'+str(i_los_speed_with_walk)+'\n')
            else:
                i_los, i_los_speed, i_los_changes = indicators_calc_flow(ver, agt, period, ped_speed, max_dist_walk)
                f.write(str(i_los)+';'+str(i_los_speed)+';'+str(round(i_los_changes, 2))+';'+'0'+';'+'0'+'\n')
        f.close()


def indicators_agg_allcalc_carry(ver='', ped_speed=0, max_dist_walk=0):
    '''
    Расчет индикаторов для кластеров и тр. районов на основе провозной возможности
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_out = 'indicators_agg_carry' + ver + '.csv'
    los_agg_types = ['tr_district', 'cluster']
    with open(path_out + name_out, 'w', encoding='utf-8', newline='') as f:
        f.write('los'+';'+'los_speed'+';'+'los_changes'+';'+'los_with_walk'+';'+'los_speed_with_walk'+'\n')
        for agt in los_agg_types:
            if max_dist_walk > 0 and agt == 'tr_district':
                i_los, i_los_speed, i_los_changes, i_los_with_walk, i_los_speed_with_walk = indicators_calc_carry(ver, agt, ped_speed, max_dist_walk)
                f.write(str(i_los)+';'+str(i_los_speed)+';'+str(round(i_los_changes, 2))+';'+str(i_los_with_walk)+';'+str(i_los_speed_with_walk)+'\n')
            else:
                i_los, i_los_speed, i_los_changes = indicators_calc_carry(ver, agt, ped_speed, max_dist_walk)
                f.write(str(i_los)+';'+str(i_los_speed)+';'+str(round(i_los_changes, 2))+';'+'0'+';'+'0'+'\n')
        f.close()


def geom_gen(ver, flow_base):
    try:
        path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
        agg_types = ['cluster', 'tr_district']
        geom_types = ['geom_clusters', 'geom_tr_distr']
        geom_id = ['id_cluster', 'id_tr_distr']

        for i in range(len(agg_types)):
            if flow_base:
                name_los_agg_in_out = 'los_agg_by_flow_' + agg_types[i] + ver + '.csv'
                name_los_agg_short_in_out = 'los_agg_by_flow_short_' + agg_types[i] + ver + '.csv'
            else:
                name_los_agg_in_out = 'los_agg_by_carry_' + agg_types[i] + ver + '.csv'
                name_los_agg_short_in_out = 'los_agg_by_carry_short_' + agg_types[i] + ver + '.csv'
            name_geom = geom_types[i] + ver + '.csv'
            los_agg_short = pd.read_csv(path_out + name_los_agg_short_in_out, delimiter=';', encoding='utf-8')
            geom = pd.read_csv(path_src + name_geom, delimiter=';', encoding='utf-8')
            geom.rename(columns={geom_id[i]: 'agg_source'}, inplace=True)
            geom['agg_source'] = geom['agg_source'].astype('int64')
            # добавляем геом к сгруппированным лосам
            los_agg_short['agg_source'] = los_agg_short['agg_source'].astype('int64')
            los_agg_short = los_agg_short.merge(geom, how='left', on=['agg_source'])
            los_agg_short.to_csv(path_out + name_los_agg_short_in_out, sep=';', encoding='utf-8', index=False) #, compression='zip'
            # добавляем геом к полным лосам, вешаем геом на таргет
            los_agg = pd.read_csv(path_out + name_los_agg_in_out, delimiter=';', encoding='utf-8')
            los_agg['agg_target'] = los_agg['agg_target'].astype('int64')
            geom.rename(columns={'agg_source': 'agg_target'}, inplace=True)
            los_agg = los_agg.merge(geom, how='left', on=['agg_target'])
            los_agg.to_csv(path_out + name_los_agg_in_out, sep=';', encoding='utf-8', index=False) #, compression='zip'
    except Exception as app_error:
        app.logger.info(f"{session['username']} geom_gen error:\n{geom.head(3)}")


def remove_temp(ver):
    '''
    Удаляем временные файлы
    '''
    # Удаляем файл rnd_combi
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    rnd_combi = 'rnd_combi' + ver + '.csv'
    os.remove(path_out + rnd_combi)
    # Удаляем файл los_short
    los_short = 'los_short' + ver
    os.remove(path_out + los_short)
    # Удаляем файл vectors (без пеших подходов)
    vectors = 'vectors' + ver + '.csv'
    os.remove(path_out + vectors)
    # Удаляем файл vectors (без vectors подходов)
    vectors_short = 'vectors_short' + ver
    os.remove(path_out + vectors_short)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_histo(df, userpath_out, x, name_out, title, x_name, y_name):
    series = pd.to_numeric(df[x], errors='coerce')
    plot_df = df.loc[series > 0].copy()
    plot_df[x] = series[series > 0]

    fig, ax = plt.subplots()
    sns.set_theme(palette='pastel')

    if plot_df.empty:
        ax.set(xlabel=x_name, ylabel=y_name, title=title)
        ax.text(0.5, 0.5, 'Нет данных для построения', ha='center', va='center', transform=ax.transAxes)
    else:
        kde_enabled = plot_df[x].shape[0] > 1
        histo = sns.histplot(data=plot_df, x=x, kde=kde_enabled, color="#4CB391", ax=ax)
        histo.set(xlabel=x_name, ylabel=y_name, title=title)
        histo.margins(x=0, y=0)

    fig.set_size_inches(7.7, 4.2)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(f'{userpath_out}{name_out}.png', dpi=80)
    plt.close(fig)


def get_ref_links(ver):
    '''
    Выгружаем справочник линки-rnd
    '''
    _, path_src = path_gen(session['username'], 'los', ver[1:], False)
    name_in = 'vectors_w' + ver + '.csv'
    name_out = 'ref_links' + ver + '.csv'
    ref_links = pd.read_csv(path_src + name_in, delimiter=';', encoding='utf-8')
    ref_links = ref_links[['link', 'RNDs']]
    ref_links.to_csv(path_src + name_out, sep=';', encoding='utf-8', index=False) #, compression='zip'


def do_los_rich(ver, period):
    '''
    Обогащаем выдачу los расстояними по воздушной линии (или графу) и потоком зв выбранный период
    '''
    path_src, path_out = path_gen(session['username'], 'los', ver[1:], False)
    name_los_in_out = 'los' + ver + '.csv'
    name_flow_in = 'flow' + ver + '.csv'
    name_dist_matrix_in = 'matrix_dist_stops' + ver + '.csv'
    los = pd.read_csv(path_out + name_los_in_out, delimiter=';', encoding='utf-8') 
    flow = pd.read_csv(path_src + name_flow_in, delimiter=';', encoding='utf-8')
    flow = flow[['link', period]]
    flow_period_name = f'flow_{period}'
    flow.rename(columns={period: flow_period_name}, inplace=True)
    dist_matrix = pd.read_csv(path_src + name_dist_matrix_in, delimiter=';', encoding='utf-8') 
    dist_matrix = dist_matrix[['link', 'distance']]
    dist_matrix.rename(columns={'distance': 'shortest_distance'}, inplace=True)
    los = pd.merge(los, flow, how='left', on='link')
    los = pd.merge(los, dist_matrix, how='left', on='link')
    los.to_csv(path_out + name_los_in_out, sep=';', encoding='utf-8', index=False)


########## Картограммы и прочий визуальный стафф #############
def make_isochrone(path_src, path_out, proj_name, target):
    '''
    Рисуем изохрону транспортной доступности ОП
    '''
    default_location = [55.75, 37.62]
    name_geom_stops = 'geom_stops_' + proj_name + '.csv'
    geom_stops = load_geo_csv(path_src + name_geom_stops)
    geom_stops['id_stop'] = pd.to_numeric(
        geom_stops['id_stop'], errors='raise'
    ).astype('int64')

    name_los_out = 'los_iso_' + proj_name + '.csv'
    los_iso_path = path_out + name_los_out
    required_iso_columns = {'source', 'target', 'los', 'all_vars', 'wkt'}
    iso_prepared = os.path.isfile(los_iso_path)
    if iso_prepared:
        gdf = load_geo_csv(los_iso_path)
        # Кэш старого формата не содержит признака пешей связи и должен быть
        # один раз пересоздан в транспортном формате.
        iso_prepared = required_iso_columns.issubset(gdf.columns)

    if not iso_prepared:
        name_los_in = 'los_' + proj_name + '.csv'
        los = pd.read_csv(path_out + name_los_in, delimiter=';', encoding='utf-8')
        los = los.loc[
            (los['all_vars'] != 'W') & (los['los'] > 0),
            ['source', 'target', 'los', 'all_vars'],
        ].copy()
        los['source'] = pd.to_numeric(los['source'], errors='raise').astype('int64')
        los['target'] = pd.to_numeric(los['target'], errors='raise').astype('int64')

        geom_targets = geom_stops[['id_stop', 'wkt']].rename(
            columns={'id_stop': 'target'}
        )
        los = los.merge(geom_targets, how='left', on='target')
        los = los[los['wkt'].notna()]
        los.to_csv(los_iso_path, sep=';', encoding='utf-8', index=False)
        gdf = load_geo_csv(los_iso_path)

    # if gdf.is_valid == False: 
    #     app.logger.info(f'{session['username']} make_isochrone error: geo data is not valid')
    #     return None
    if gdf.empty:
        app.logger.warning('make_isochrone: empty source data for project %s', proj_name)
        return [], _make_los_map(default_location)

    gdf = gdf.set_crs('epsg:4326')
    gdf['source'] = pd.to_numeric(gdf['source'], errors='raise').astype('int64')
    gdf['target'] = pd.to_numeric(gdf['target'], errors='raise').astype('int64')
    gdf['los'] = pd.to_numeric(gdf['los'], errors='raise').astype('float32')
    gdf = gdf[(gdf['all_vars'] != 'W') & (gdf['los'] > 0)]

    targets = np.sort(gdf.source.unique()).tolist()
    if len(targets) == 0:
        app.logger.warning('make_isochrone: no source targets for project %s', proj_name)
        return [], _make_los_map(default_location)

    if target is None:
        push_target = targets[0]
    else:
        try:
            push_target = int(target)
        except (TypeError, ValueError):
            app.logger.warning(
                'make_isochrone: invalid source stop %s in project %s; using %s',
                target, proj_name, targets[0],
            )
            push_target = targets[0]

    if push_target not in targets:
        app.logger.warning(
            'make_isochrone: source stop %s has no transport LOS in project %s; using %s',
            push_target, proj_name, targets[0],
        )
        push_target = targets[0]

    source_marker = geom_stops.loc[geom_stops['id_stop'] == push_target].iloc[:1]
    if source_marker.empty:
        app.logger.warning('make_isochrone: source marker not found for stop %s in project %s', push_target, proj_name)

    gdf = gdf[gdf.source == push_target]
    if gdf.empty:
        app.logger.warning('make_isochrone: no LOS rows for source %s in project %s', push_target, proj_name)
        map_location = default_location
        if not source_marker.empty:
            sm = source_marker[:1].get_coordinates()
            map_location = [sm.y.iloc[0], sm.x.iloc[0]]
        return targets, _make_los_map(map_location)

    if not source_marker.empty:
        marker_row = gpd.GeoDataFrame(
            {
                'source': [push_target],
                'target': [push_target],
                'los': [0.0],
                'all_vars': ['SOURCE'],
            },
            geometry=source_marker.geometry.values,
            crs='epsg:4326',
        )
        gdf = gpd.GeoDataFrame(
            pd.concat([marker_row, gdf], ignore_index=True),
            geometry='geometry',
            crs='epsg:4326',
        )

    max_los = gdf.los.max()*60
    stage_step = 10
    max_stage = max(stage_step, int(np.ceil(max_los / stage_step) * stage_step))
    los_stages = list(range(stage_step, max_stage + stage_step, stage_step))

    geoms = []
    feats = []
    for s in los_stages:
        temp_df = gdf[gdf.los <= s/60]
        if temp_df.empty:
            continue
        points = temp_df.geometry.unary_union
        if len(temp_df.geometry.drop_duplicates()) >= 3:
            geom = concave_hull(
                points,
                ratio=0.4,
                allow_holes=False,
            )
        else:
            geom = points.convex_hull
        if geom.is_empty:
            continue
        geoms.append(geom)
        feats.append(s)

    if not geoms:
        app.logger.warning('make_isochrone: no geometries built for source %s in project %s', push_target, proj_name)
        map_location = default_location
        if not source_marker.empty:
            sm = source_marker[:1].get_coordinates()
            map_location = [sm.y.iloc[0], sm.x.iloc[0]]
        m = _make_los_map(map_location)
        if not source_marker.empty:
            folium.Marker(
            location=map_location,
            tooltip=f"Id {push_target}",
            icon=folium.Icon(icon="bus", icon_color="white", color="darkblue", prefix='fa'),
            ).add_to(m)
        return targets, m

    diff_geom = []
    previous_geom = None
    for i, geom in enumerate(geoms):
        if previous_geom is None:
            stage_geom = geom
            previous_geom = geom
        else:
            cumulative_geom = previous_geom.union(geom)
            stage_geom = cumulative_geom.difference(previous_geom)
            previous_geom = cumulative_geom

        if stage_geom.is_empty:
            continue

        cur_df = gpd.GeoDataFrame(crs='epsg:4326', geometry=[stage_geom])
        cur_df['los'] = str(feats[i])
        cur_df['order'] = i
        diff_geom.append(cur_df.to_json())

    to_merc = gdf.to_crs('epsg:3857')
    zone_center = to_merc.dissolve().centroid
    zone_center = zone_center.to_crs('epsg:4326')
    x = zone_center.x[0]
    y = zone_center.y[0]
    m = _make_los_map([y, x])

    fill_colors = sns.blend_palette(
        ['#009933', '#00cc66', '#80ff80', '#ffff99', '#ffbf80', '#ff3300'],
        n_colors=max(len(feats), 2),
    ).as_hex()
    for i, geom_json in enumerate(diff_geom):
        folium.GeoJson(geom_json,
                    name = str(i),
                    style_function = lambda x: {'fillColor': fill_colors[x['properties']['order']],
                            'color':'#404040', 
                            'fillOpacity': 0.4, 
                            'weight': 0.2},
                    tooltip = folium.features.GeoJsonTooltip(
                                    fields=['los'],
                                    aliases=['Доступность (минут):'],
                                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 4px;")) 
                    ).add_to(m)
    folium.LayerControl().add_to(m)

    if not source_marker.empty:
        sm = source_marker[:1].get_coordinates()
        folium.Marker(
        location=[sm.y.iloc[0], sm.x.iloc[0]],
        tooltip=f"Id {push_target}",
        icon=folium.Icon(icon="bus", icon_color="white", color="darkblue", prefix='fa'),
        ).add_to(m)
    return targets, m


def prepare_geo_los(path, proj_name, geom_type, id_name, only_df, flow_base):
    if flow_base:
        name_geom = 'los_agg_by_flow_short_' + geom_type + '_' + proj_name + '.csv'
    else:
        name_geom = 'los_agg_by_carry_short_' + geom_type + '_' + proj_name + '.csv'
    df = pd.read_csv(path + name_geom, delimiter=';', encoding='utf-8')
    df = df.rename(columns={"agg_source": id_name})
    df = df[df['wkt'].notna()]
    df.drop(columns=['wkt'], inplace=True)
    geo_df = load_geo_csv(path + name_geom)
    geo_df = geo_df[geo_df['geometry']!=None]
    geo_df.drop(columns=['wkt'], inplace=True)
    geo_df = geo_df.rename(columns={"agg_source": id_name})
    geo_df = geo_df.set_crs('epsg:4326')
    if flow_base:
        max_los_speed = df['los_spd_by_flow'].max()
        min_los_speed = df['los_spd_by_flow'].min()
    else:
        max_los_speed = df['los_spd_by_carry'].max()
        min_los_speed = df['los_spd_by_carry'].min()
    if only_df == True:
        return df, geo_df
    else:
        return float(max_los_speed), float(min_los_speed), df, geo_df


def make_los_map_tr(path_src, proj_name, prj_info):
    '''
    Картограммы уровня обслуживания в разрезе транспортных районов
    '''
    agg_types = ['cluster', 'tr_district']

    if prj_info['status_src'] == 3:
        df, geo_df = prepare_geo_los(path_src, proj_name, agg_types[1], "id_tr_distr", True, False)
        weight_type_spd = "los_spd_by_carry"
        weight_type_los = "los_by_carry"
    elif prj_info['status_src'] == 4:
        df, geo_df = prepare_geo_los(path_src, proj_name, agg_types[1], "id_tr_distr", True, True)
        weight_type_spd = "los_spd_by_flow"
        weight_type_los = "los_by_flow"
    # geo_df[weight_type_spd] = geo_df[weight_type_spd].astype('float32').round(3)
    # geo_df[weight_type_los] = geo_df[weight_type_los].astype('float32').round(3)
    df = df.round(3)
    to_merc = geo_df.to_crs('epsg:3857')
    zone_center = to_merc.dissolve().centroid
    zone_center = zone_center.to_crs('epsg:4326')
    x = zone_center.x[0]
    y = zone_center.y[0]

    m = _make_los_map([y, x], min_zoom=11)
    folium.Choropleth(
    geo_data=geo_df,
    name="Скорость",
    data=df,
    columns=["id_tr_distr", weight_type_spd],
    key_on="feature.properties.id_tr_distr",
    fill_color="BuGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Уровень обслуживания - cкорость (км/ч)",
    ).add_to(m)
    folium.Choropleth(
    geo_data=geo_df,
    name="Время",
    data=df,
    columns=["id_tr_distr", weight_type_los],
    key_on="feature.properties.id_tr_distr",
    fill_color="OrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    show=False,
    legend_name="Уровень обслуживания - время (ч)",
    ).add_to(m)

    tr_geom_json = geo_df.to_json()
    style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.30, 
                                'weight': 0.1}
    tips = folium.features.GeoJson(
    data=tr_geom_json,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=["id_tr_distr", weight_type_spd, weight_type_los],
        aliases=["Район:", "Скорость (км/ч):", "Время (ч):"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
    )
    m.add_child(tips)
    m.keep_in_front(tips)
    # folium.TileLayer('OpenStreetMap',name="OpenStreetMap",control=True).add_to(m)
    # folium.TileLayer('cartodbdark_matter',name="Темный фон",control=True).add_to(m)
    folium.LayerControl().add_to(m)
    iframe = m.get_root()._repr_html_()
    return m


def make_los_map_cl(path_geom_src, path_cluster_src, proj_name, prj_info):
    '''
    Картограммы уровня обслуживания в разрезе кластеров
    '''
    agg_types = ['cluster', 'tr_district']

    if prj_info['status_src'] == 3:
        max_los_speed, min_los_speed, df, geo_df = prepare_geo_los(path_cluster_src, proj_name, agg_types[0], 'id_cluster', False, False)
        weight_type_spd = "los_spd_by_carry"
        weight_type_los = "los_by_carry"
    elif prj_info['status_src'] == 4:
        max_los_speed, min_los_speed, df, geo_df = prepare_geo_los(path_cluster_src, proj_name, agg_types[0], 'id_cluster', False, True)
        weight_type_spd = "los_spd_by_flow"
        weight_type_los = "los_by_flow"
    to_merc = geo_df.to_crs('epsg:3857')
    zone_center = to_merc.dissolve().centroid
    zone_center = zone_center.to_crs('epsg:4326')
    x = zone_center.x[0]
    y = zone_center.y[0]
    clusters_geom_json = geo_df.to_json()
    m = _make_los_map([y, x], min_zoom=11)

    cl_colors = ['#ff8566', '#ffff99', '#b3ffb3', '#00e600']
    # Делим фактический диапазон скоростей на четыре равных интервала.
    cl_color_range = [
        round(value, 1)
        for value in np.linspace(min_los_speed, max_los_speed, 5)[1:-1]
    ]
    
    def get_color_range(los, colors, color_range):
        if float(los) < color_range[0]:
            c = colors[0]
        elif float(los) >= color_range[0] and float(los) < color_range[1]:
            c = colors[1]
        elif float(los) >= color_range[1] and float(los) < color_range[2]:
            c = colors[2]
        elif float(los) >= color_range[2]:
            c = colors[3]
        return c

    style_function_cl = lambda x: {'fillColor': get_color_range(
                            float(x['properties'][weight_type_spd]), cl_colors, cl_color_range), 
                            'color': '#006622',
                            'fillOpacity': 0.4, 
                            'weight': 0.2}
    highlight_function_cl = lambda x: {'fillColor': '#e6ffee',
                                'color':'#404040', 
                                'fillOpacity': 0.20, 
                                'weight': 0.2}
    clusters = folium.features.GeoJson(
    data=clusters_geom_json,
    marker=folium.CircleMarker(
                radius=17,
                fill_color='green',
                color='green',
            ),
    style_function=style_function_cl, 
    control=False,
    highlight_function=highlight_function_cl,
    tooltip=folium.features.GeoJsonTooltip(
        fields=["id_cluster", weight_type_spd, weight_type_los],
        aliases=["Кластер:", "Скорость (км/ч):", "Время (ч):"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
    )
    m.add_child(clusters)
    m.keep_in_front(clusters)
    stops_geom = load_geo_df(path_geom_src, proj_name, 'geom_stops')
    stops_geom_json = stops_geom.to_json()

    style_function_stops = lambda x: {'fillColor': '#b3b3b3', 
                            'color':'#404040', 
                            'fillOpacity': 0.7, 
                            'weight': 0.2}
    highlight_function_stops = lambda x: {'fillColor': '#e6e6e6', 
                                'color':'#404040', 
                                'fillOpacity': 0.50, 
                                'weight': 0.2}
    stops = folium.features.GeoJson(
    data=stops_geom_json,
    marker=folium.CircleMarker(
                radius=3,
                fill_color='green',
                color='green',
            ),
    style_function=style_function_stops,
    control=False,
    highlight_function=highlight_function_stops,
    tooltip=folium.features.GeoJsonTooltip(
        fields=["id_stop", "name"],
        aliases=["Код остановки:", "Наименование:"],
        labels=True,
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px") 
    )
    )
    m.add_child(stops)
    m.keep_in_front(stops)
    # folium.TileLayer('OpenStreetMap', name="CartoDB",control=True).add_to(m)
    # folium.TileLayer('cartodbdark_matter',name="Темный фон",control=True).add_to(m) #or tiles="Stamen Watercolor"
    folium.LayerControl().add_to(m)
    return cl_color_range, m


def make_walk_avail_map(path_src_data, path_processed_data, proj_name, prj_info):
    '''
    Карта пешей доступности
    '''
    stops_geom = load_geo_df(path_src_data, proj_name, 'geom_stops')
    stops_geom_json = stops_geom.to_json()
    stops_geom = stops_geom.set_crs('epsg:4326')
    to_merc = stops_geom.to_crs('epsg:3857')
    zone_center = to_merc.dissolve().centroid
    zone_center = zone_center.to_crs('epsg:4326')
    x = zone_center.x[0]
    y = zone_center.y[0]

    m = _make_los_map([y, x], min_zoom=11)

    style_function_stops = lambda x: {'fillColor': '#0073e6', 
                            'color':'#404040', 
                            'fillOpacity': 0.7, 
                            'weight': 0.2}
    highlight_function_stops = lambda x: {'fillColor': '#0073e6', 
                                'color':'#404040', 
                                'fillOpacity': 0.50, 
                                'weight': 0.2}
    stops = folium.features.GeoJson(
    data=stops_geom_json,
    marker=folium.CircleMarker(
                radius=3,
                fill_color='blue',
                color='blue',
            ),
    style_function=style_function_stops,
    control=True,
    name="Остановки",
    highlight_function=highlight_function_stops,
    tooltip=folium.features.GeoJsonTooltip(
        fields=["id_stop", "name"],
        aliases=["Код остановки:", "Наименование:"],
        labels=True,
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px") 
    )
    )
    m.add_child(stops)
    m.keep_in_front(stops)

    # Зоны пешей доступности - объединенные буферы
    uu_stop_geom = load_geo_df(path_processed_data, proj_name, 'geom_uu_stop')
    uu_stop_geom_json = uu_stop_geom.to_json()

    style_function_uu = lambda x: {'fillColor': '#4dff4d', 
                            'color':'#404040', 
                            'fillOpacity': 0.4, 
                            'weight': 0.2}
    highlight_function_uu = lambda x: {'fillColor': '#4dff4d', 
                                'color':'#404040', 
                                'fillOpacity': 0.20, 
                                'weight': 0.2}
    uu_stop = folium.features.GeoJson(
    data=uu_stop_geom_json,
    control=True,
    name="Зона пешей доступности",
    style_function=style_function_uu,
    highlight_function=highlight_function_uu,
    )
    m.add_child(uu_stop)
    # m.keep_in_front(uu_stop)

    # Зоны неудовлетворительной доступности - разбитые транспортные районы
    area_not_aw_geom = load_geo_df(path_processed_data, proj_name, 'geom_area_not_available')
    area_not_aw_geom_json = area_not_aw_geom.to_json()

    style_function_aw = lambda x: {'fillColor': '#737373', 
                            'color':'#404040', 
                            'fillOpacity': 0.4, 
                            'weight': 0.2}
    highlight_function_aw = lambda x: {'fillColor': '#737373', 
                                'color':'#404040', 
                                'fillOpacity': 0.30, 
                                'weight': 0.2}
    area_not_av = folium.features.GeoJson(
    data=area_not_aw_geom_json,
    control=True,
    name="Территории неуд. доступности",
    style_function=style_function_aw,
    highlight_function=highlight_function_aw,
    tooltip=folium.features.GeoJsonTooltip(
    fields=["id_tr_distr", "area_not_av"],
    aliases=["Район:", "Площадь неудовл. доступности (км2):"],
    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
    )
    m.add_child(area_not_av)
    # folium.TileLayer('OpenStreetMap',name="OpenStreetMap",control=True).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def prepare_geo_los_detailed(path, proj_name, geom_type, id_name, flow_base, target):
    if flow_base:
        name_geom = 'los_agg_by_flow_' + geom_type + '_' + proj_name + '.csv'
        name_geom_short = 'los_agg_by_flow_short_' + geom_type + '_' + proj_name + '.csv'
    else:
        name_geom = 'los_agg_by_carry_' + geom_type + '_' + proj_name + '.csv'
        name_geom_short = 'los_agg_by_carry_short_' + geom_type + '_' + proj_name + '.csv'

    df = pd.read_csv(path + name_geom, delimiter=';', encoding='utf-8')
    df['agg_source'] = pd.to_numeric(df['agg_source'], errors='raise').astype('int64')
    df['agg_target'] = pd.to_numeric(df['agg_target'], errors='raise').astype('int64')
    targets = np.sort(df['agg_source'].unique()).tolist()
    if not targets:
        raise ValueError(f'Нет исходящих районов для детальной карты проекта {proj_name}')

    if target is None:
        push_target = targets[0]
    else:
        try:
            push_target = int(target)
        except (TypeError, ValueError):
            push_target = targets[0]
    if push_target not in targets:
        push_target = targets[0]

    df = df.loc[df['agg_source'] == push_target].copy()
    df = df.rename(columns={"agg_target": id_name})
    df.drop(columns=['wkt'], inplace=True)

    geo_df = load_geo_csv(path + name_geom)
    geo_df['agg_source'] = pd.to_numeric(
        geo_df['agg_source'], errors='raise'
    ).astype('int64')
    geo_df['agg_target'] = pd.to_numeric(
        geo_df['agg_target'], errors='raise'
    ).astype('int64')
    geo_df = geo_df.loc[
        (geo_df['agg_source'] == push_target) & geo_df.geometry.notna()
    ].copy()
    geo_df.drop(columns=['wkt'], inplace=True)
    geo_df = geo_df.rename(columns={"agg_target": id_name})

    # В полном файле геометрия относится к району назначения. Геометрию
    # выбранного исходного района берём из агрегированного файла по источникам.
    source_geo = load_geo_csv(path + name_geom_short)
    source_geo['agg_source'] = pd.to_numeric(
        source_geo['agg_source'], errors='raise'
    ).astype('int64')
    source_geo = source_geo.loc[
        (source_geo['agg_source'] == push_target) & source_geo.geometry.notna()
    ].iloc[:1]
    if source_geo.empty:
        source_marker = gpd.GeoSeries([], crs='epsg:4326')
    else:
        source_marker = source_geo.to_crs('epsg:3857').geometry.centroid
        source_marker = source_marker.to_crs('epsg:4326')

    if flow_base:
        max_los_speed = df['los_spd_by_flow'].max()
        min_los_speed = df['los_spd_by_flow'].min()
    else:
        max_los_speed = df['los_spd_by_carry'].max()
        min_los_speed = df['los_spd_by_carry'].min()
    return float(max_los_speed), float(min_los_speed), df, geo_df, targets, source_marker, push_target


def make_los_map_tr_separate(path_src, prj_info, req_data, proj_name):
    '''
    Картограммы доступности отдельных тр. районов
    '''
    agg_types = ['cluster', 'tr_district']

    if 'tr_source' not in req_data:
        tr_source = None  
    else:
        tr_source = int(req_data['tr_source'])

    if prj_info['status_src'] == 3:
        max_los_speed, min_los_speed, df, geo_df, targets, source_marker, source_id = prepare_geo_los_detailed(path_src, proj_name, agg_types[1], "id_tr_distr", False, tr_source)
        weight_type_spd = "los_spd_by_carry"
        weight_type_los = "los_by_carry"
    elif prj_info['status_src'] == 4:
        max_los_speed, min_los_speed, df, geo_df, targets, source_marker, source_id = prepare_geo_los_detailed(path_src, proj_name, agg_types[1], "id_tr_distr", True, tr_source)
        weight_type_spd = "los_spd_by_flow"
        weight_type_los = "los_by_flow"

    df = df.round(3)
    if geo_df.empty:
        app.logger.warning(
            'make_los_map_tr_separate: no geometry for source %s in project %s',
            source_id, proj_name,
        )
        m = _make_los_map([55.75, 37.62])
        return tr_source, targets, m

    to_merc = geo_df.to_crs('epsg:3857')
    zone_center = to_merc.dissolve().centroid
    zone_center = zone_center.to_crs('epsg:4326')
    x = zone_center.x.iloc[0]
    y = zone_center.y.iloc[0]

    m = _make_los_map([y, x], min_zoom=11)
    folium.Choropleth(
    geo_data=geo_df,
    name="Скорость",
    data=df,
    columns=["id_tr_distr", weight_type_spd],
    key_on="feature.properties.id_tr_distr",
    fill_color="BuGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Уровень обслуживания - cкорость (км/ч)",
    ).add_to(m)
    folium.Choropleth(
    geo_data=geo_df,
    name="Время",
    data=df,
    columns=["id_tr_distr", weight_type_los],
    key_on="feature.properties.id_tr_distr",
    fill_color="OrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    show=False,
    legend_name="Уровень обслуживания - время (ч)",
    ).add_to(m)

    tr_geom_json = geo_df.to_json()
    style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.30, 
                                'weight': 0.1}
    tips = folium.features.GeoJson(
    data=tr_geom_json,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=["id_tr_distr", weight_type_spd, weight_type_los],
        aliases=["Район:", "Скорость (км/ч):", "Время (ч):"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
    )
    m.add_child(tips)
    m.keep_in_front(tips)

    if not source_marker.empty:
        folium.Marker(
        location=[source_marker.y.iloc[0], source_marker.x.iloc[0]],
        tooltip=f"Id {source_id}",
        icon=folium.Icon(icon="bus", icon_color="white", color="darkblue", prefix='fa'),
        ).add_to(m)
    else:
        app.logger.warning(
            'make_los_map_tr_separate: source marker not found for district %s in project %s',
            source_id, proj_name,
        )
    # folium.TileLayer('OpenStreetMap',name="OpenStreetMap",control=True).add_to(m)
    # folium.TileLayer('cartodbdark_matter',name="Темный фон",control=True).add_to(m)
    folium.LayerControl().add_to(m)
    return tr_source, targets, m
