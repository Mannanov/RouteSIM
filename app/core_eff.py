import pandas as pd
import csv
# from collections import deque
import os.path
import json
from serv_funcs import path_gen
import numpy as np
import cairo
from routes import session
from rsim import app
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


### Дублирование ###
def rnd_combi_prepare(ver):
    '''
    готовим комбинации всех межостановочных связей на маршрутах по направлениям движения
    ver - код проекта
    Кодировка: UTF-8
    сортировка по rnd (route/direction/order)
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_in = 'rnd_short' + ver + '.csv'
        name_out = 'rnd_short_combi' + ver + '.csv'
        name_out_simple = 'rnd_short_simple' + ver + '.csv'
            
        reader = csv.reader(
            open(path_src + name_in, 'r', encoding='utf-8', newline=''),
            delimiter=";",
        )

        prebase = {}
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            prebase[count] = {}
            prebase[count]['rnd'] = row[0]
            prebase[count]['source'] = row[2]
            prebase[count]['target'] = row[3]
            prebase[count]['dist'] = row[4]
            count += 1

        with open(path_out + name_out, 'w', encoding='utf-8', newline='') as f:
            f.write('rnd'+';'+'route'+';'+'direction'+';'+'link'+';'+'dist'+'\n')
            for i in prebase:
                rnd = prebase[i]['rnd']
                route = prebase[i]['rnd'].split('-')[0]
                direction = prebase[i]['rnd'].split('-')[1]
                start_point = prebase[i]['source']
                position = int(i)
                sum_length = 0
                for j in prebase:
                    if rnd == prebase[j]['rnd']:
                        if int(j) >= position:
                            sum_length += float(prebase[j]['dist'])
                            f.write(rnd+";"+route+";"+direction+";"+start_point+">"+prebase[j]['target']+";"+str(round(sum_length, 3))+'\n')
            f.close()

        with open(path_out + name_out_simple, 'w', encoding='utf-8', newline='') as f:
            f.write('rnd'+';'+'route'+';'+'direction'+';'+'link'+';'+'dist'+'\n')
            for i in prebase:
                rnd = prebase[i]['rnd']
                route = prebase[i]['rnd'].split('-')[0]
                direction = prebase[i]['rnd'].split('-')[1]
                start_point = prebase[i]['source']
                position = int(i)
                sum_length = '0'
                f.write(rnd+";"+route+";"+direction+";"+start_point+">"+prebase[i]['target']+";"+sum_length+'\n')
            f.close()

        rnd_combi = pd.read_csv(path_out + name_out, delimiter=';', encoding='utf-8')
        rnd_combi.to_csv(path_out + name_out, sep=';', index=False, encoding='utf-8')

        rnd_simple = pd.read_csv(path_out + name_out_simple, encoding='utf-8', delimiter=';')
        rnd_simple.to_csv(path_out + name_out_simple, sep=';', index=False, encoding='utf-8')

    except Exception:
        app.logger.exception('rnd_combi_prepare error')
        raise


def routes_doubl(ver, rnd=False, hard_calc=False):
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        if hard_calc:
            name_in = 'rnd_short_combi' + ver + '.csv'
        else:
            name_in = 'rnd_short_simple' + ver + '.csv'

        name_column_out = 'column_doubl' + ver + '.csv'
        name_matrix_out = 'matrix_doubl' + ver + '.csv'

        reader = csv.reader(
            open(path_out + name_in, 'r', encoding='utf-8', newline=''),
            delimiter=";",
        )  # сортируем массив по МВН
        prevroute = ''
        rbase = {}
        #собираем в кучу МВН - все обеспечиваемые связи
        label = 1
        for row in reader:
            if label == 1:
                label = 0
                continue
                
            if rnd:
                route = row[0] #rnd
            else:
                route = row[1] #route
            segment = row[3] #link
            # if rnd:
            #     route = row[0].split('-')[0]

            if route != prevroute:
                rbase[route] = [segment]
                prevroute = route
            else:
                rbase[route].append(segment)

        #Вывод в столбик
        rmatch = {}
        for i in rbase:
            rmatch[i] = {}
            col = len(rbase[i])
            rmatch[i]['sum_seg'] = col

        count = 0
        for ri in rmatch:
            for rj in rbase: 
                if rj != ri:
                    count = 0
                    for si in rbase[ri]:
                        for sj in rbase[rj]: #segments
                            if si == sj:
                                count += 1
                                rmatch[ri][rj] = count

        f = open(path_out + name_column_out, 'w', encoding='utf-8', newline='')
        f.write('RND' + ';' + 'RND_compare' + ';' + 'double %' + '\n')

        for i in rmatch:
            for j in rmatch[i]:
                longstr = ''
                h = rmatch[i][j]
                d_proc = round(rmatch[i][j]/rmatch[i]['sum_seg']*100, 1)
        #         if h > 5:   # filter na mnogo matches
                if d_proc > 5:
                    #print('x')
                    if j != 'sum_seg':
                        if d_proc > 100:
                            d_proc = 100
                        longstr = longstr + ";" + j + ";" + str(d_proc) # выдача в процентах
        #             else:
        #                 longstr = longstr + ";" + j + ";" + str(rmatch[i][j])
                        f.write(i + longstr + '\n')
                else: continue
        f.close()

        double_df = pd.read_csv(path_out + name_column_out, encoding='utf-8', delimiter=';')
        double_df = double_df.sort_values(by=['RND', 'double %'], ascending=False)
        double_df.to_csv(path_out + name_column_out, sep=';', encoding='utf-8', index=False)

        #Вывод матрицы
        rmatch = {}
        for i in rbase:
            rmatch[i] = {}
            col = len(rbase[i])
            rmatch[i]['sum_seg'] = col

        for i in rmatch:  # delaem vihod bez dirok
            for j in rbase:
                rmatch[i][j] = 0

        count = 0                    
        for ri in rmatch:
            for rj in rbase: 
                    if rj != ri and rj != 'sum_seg':
                        count = 0
                        for si in rbase[ri]:
                            for sj in rbase[rj]:
                                if si == sj:
                                    count += 1
                                    rmatch[ri][rj] = count

        f = open(path_out + name_matrix_out, 'w', encoding='utf-8', newline='')
        longstr = ''
        for i in rmatch:
            longstr += str(i) + ";"
        f.write('route'+';'+'sum_seg'+';'+ longstr[:-1] + '\n')

        for i in rmatch:
            longstr = ''
            for j in rmatch[i]:
                if j != 'sum_seg':
                    d_proc = round(rmatch[i][j]/rmatch[i]['sum_seg']*100, 1)
                    if d_proc > 100:
                        d_proc = 100
        #             elif d_proc < 5:
        #                 d_proc = ''
                    longstr += ";" + str(d_proc)
                else:
                    longstr += ";" + str(rmatch[i][j])
            f.write(i + longstr + '\n')
        f.close()
        double_df = pd.read_csv(path_out + name_matrix_out, encoding='utf-8', delimiter=';')
        double_df = double_df.drop(['sum_seg'], axis=1)
        double_df = double_df.rename(columns={'route': 'маршрут'})
        double_df.to_csv(path_out + name_matrix_out, sep=';', encoding='utf-8', index=False)
    except Exception as app_error:
        app.logger.info(f'routes_doubl error:\n{app_error}')


def df_styler_double(styler):
    # styler.set_caption("Double")
    try:
        styler.format(precision=1)
        styler.hide(axis="index")
        styler.background_gradient(axis=None, vmin=0, vmax=100, cmap="Oranges") #GnRd YlGnBu YlOrBr Oranges Reds YlGn
        return styler
    except BaseException as app_error:
        app.logger.info(f'df_styler_double error:\n{app_error}')    


def df_styler_renew(styler):
    try:
        styler.format(precision=2)
        styler.hide(axis="index")
        styler.background_gradient(axis=None, vmin=0, vmax=1, cmap="YlGnBu")
        return styler
    except Exception as app_error:
        app.logger.info(f'df_styler_renew error:\n{app_error}')   


def df_styler_irregularity(styler):
    try:
        irregularity_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "BlueNeutralGreen", ["#2166ac", "#ffffbf", "#1a9850"]
        )
        styler.format(precision=2)
        styler.hide(axis="index")
        styler.background_gradient(axis=None, vmin=-1, vmax=1, cmap=irregularity_cmap)
        return styler
    except Exception as app_error:
        app.logger.info(f'df_styler_irregularity error:\n{app_error}')


### Загрузка на перегонах ###
def calc_line_load(ver):
    '''
    Поток на перегонах по всем периодам
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_corrs_in = 'routes_flow' + ver + '.csv'
        name_corrs_out = 'routes_flow_ext' + ver + '.csv'
        name_rnd_combi_in = 'rnd_short_combi' + ver + '.csv'
        name_rnd_in = 'rnd_short' + ver + '.csv'
        name_carry_in = 'carry' + ver + '.csv'
        name_out_aver_distance = 'mean_distance' + ver + '.csv'
        name_out_aver_distance_rnd = 'mean_distance_rnd' + ver + '.csv'
        name_out = 'route_load' + ver + '.csv'
        name_out_seg_renew = 'route_renew_segments' + ver + '.csv'

        f_hours = ['h' + str(h) for h in range(5, 24)]
        h_dict = {h: 'float64' for h in f_hours}
        routes_corrs_types = {'route': 'object', 'link': 'object', 'source': 'int64', 
                                   'target': 'int64', 'day': 'float64'}
        routes_corrs_types = routes_corrs_types | h_dict
        df_corrs = pd.read_csv(
            path_src + name_corrs_in,
            delimiter=";",
            encoding='utf-8',
            dtype=routes_corrs_types,
        )
        df_rnd_combi = pd.read_csv(
            path_out + name_rnd_combi_in, delimiter=";", encoding='utf-8'
        )
        df_rnd = pd.read_csv(
            path_src + name_rnd_in, delimiter=";", encoding='utf-8'
        )
        df_corrs = df_corrs.merge(df_rnd_combi, on=['route', 'link'])
        df_corrs['rnd'] = df_corrs['route'] + '-' + df_corrs['direction']
        all_periods = ['h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14','h15','h16', 'h17',
                          'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'day']
        for p in all_periods:
            df_rnd[p] = 0
            for c in df_corrs.itertuples():
                rnd_idx = df_rnd.index[(df_rnd['rnd']==c.rnd) & (df_rnd['source']==c.source)]
                if len(rnd_idx) == 0:
                    continue
                for l in df_rnd.loc[rnd_idx[0]:].itertuples():
                    if l.rnd != c.rnd: break
                    df_rnd.loc[l.Index, p] += getattr(c, p)
                    if c.target == l.target: break

        df_corrs.to_csv(path_out + name_corrs_out, sep=';', index=False, encoding='utf-8')
        df_rnd.to_csv(path_out + name_out, sep=';', index=False, encoding='utf-8')

        # Средняя протяженность поездки, взвешенная по суточному потоку.
        def weighted_mean_distance(group_column):
            grouped = (df_corrs.assign(weighted_dist=df_corrs['dist'] * df_corrs['day'])
                .groupby(group_column)[['weighted_dist', 'day']]
                .sum())
            grouped['dist'] = np.where(grouped['day'] != 0, grouped['weighted_dist'] / grouped['day'], 0)
            return grouped[['dist']].round(2)

        mean_distance = weighted_mean_distance('route')
        mean_distance = mean_distance.rename(columns={'dist': 'протяженность'})
        mean_distance.to_csv(path_out + name_out_aver_distance, sep=';', encoding='utf-8')

        # добавляем Ксменности по каждому сегменту (период - сутки)
        df_carry = pd.read_csv(
            path_src + name_carry_in, delimiter=";", encoding='utf-8'
        )
        df_carry = df_carry[['rnd', 'day']]
        df_carry = df_carry.rename(columns={'day': 'day_carry'})
        # df_rnd['route'] = df_rnd['rnd'].apply(lambda x: x.split('-')[0])
        df_rnd = df_rnd.merge(df_carry, on=['rnd'])
        df_rnd['day_renew'] = round(df_rnd['day']/(df_rnd['day_carry']), 3)
        df_rnd = df_rnd[['rnd', 'order', 'source', 'target', 'distance', 'day_renew']]
        df_rnd.to_csv(path_out + name_out_seg_renew, sep=';', index=False, encoding='utf-8')


    except Exception as app_error:
        app.logger.info(f'calc_line_load error:\n{app_error}')


def calc_static_load(ver):
    '''
    Расчет коэффициента статической загрузки на межостановочных перегонах
    по каждому часовому периоду и за день
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_load_in = 'route_load' + ver + '.csv'
        name_carry_in = 'carry' + ver + '.csv'
        name_out = 'routes_static' + ver + '.csv'

        all_periods = ['h' + str(h) for h in range(5, 24)]
        all_periods.append('day')

        df_load = pd.read_csv(
            path_out + name_load_in, delimiter=";", encoding='utf-8'
        )
        df_carry = pd.read_csv(
            path_src + name_carry_in, delimiter=";", encoding='utf-8'
        )
        df_carry = df_carry[['rnd'] + all_periods]

        df_static = df_load.merge(df_carry, on=['rnd'], how='left', suffixes=('_flow', '_carry'))

        drop_cols = []
        for p in all_periods:
            flow_col = p + '_flow'
            carry_col = p + '_carry'
            df_static[p] = np.where(
                df_static[carry_col] != 0,
                (df_static[flow_col] / df_static[carry_col]).round(3),
                np.where(df_static[flow_col] == 0, 0, np.nan)
            )
            drop_cols.append(flow_col)
            drop_cols.append(carry_col)

        df_static = df_static.drop(drop_cols, axis=1)
        df_static = df_static.fillna(0)
        result_cols = ['rnd', 'order', 'source', 'target', 'distance'] + all_periods
        df_static = df_static[result_cols]
        df_static.to_csv(path_out + name_out, sep=';', encoding='utf-8', index=False)
    except Exception as app_error:
        app.logger.info(f'calc_static_load error:\n{app_error}')


def calc_irregularity(ver):
    '''
    Расчет знакового индекса направленной неравномерности по часам.
    Индекс определяется как (прямой поток - обратный поток) /
    (прямой поток + обратный поток).
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_flow_in = 'routes_flow_ext' + ver + '.csv'
        name_out = 'routes_irregularity' + ver + '.csv'

        f_hours = ['h' + str(h) for h in range(5, 24)]
        df_flow = pd.read_csv(
            path_out + name_flow_in, delimiter=";", encoding='utf-8'
        )
        df_flow = df_flow.groupby(['rnd'], as_index=False)[f_hours].sum()

        # По формату rnd последняя часть после '-' задает направление, а все
        # предшествующие части — код маршрута. Первое направление является
        # прямым, второе — обратным.
        df_flow[['route', 'direction']] = df_flow['rnd'].astype(str).str.rsplit(
            '-', n=1, expand=True
        )
        df_flow = df_flow.sort_values(by=['route', 'direction'])

        irregularity_rows = []
        for route, df_route in df_flow.groupby('route', sort=False):
            if df_route.shape[0] < 2:
                continue

            direct_flow = df_route.iloc[1]
            reverse_flow = df_route.iloc[0]
            irregularity_row = {'route': route}
            for p in f_hours:
                total_flow = direct_flow[p] + reverse_flow[p]
                if total_flow == 0:
                    irregularity_row[p] = 0
                else:
                    irregularity_row[p] = round(
                        (direct_flow[p] - reverse_flow[p]) / total_flow, 2
                    )
            irregularity_rows.append(irregularity_row)

        df_irregularity = pd.DataFrame(irregularity_rows, columns=['route'] + f_hours)
        df_irregularity.to_csv(path_out + name_out, sep=';', encoding='utf-8', index=False)
    except Exception as app_error:
        app.logger.info(f'calc_irregularity error:\n{app_error}')


    ### К сменности ###
def calc_renew(ver): # rnd=False)
    '''
    Расчет К сменности (таблица по всем периодам времени из flow)
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_flow_in = 'routes_flow_ext' + ver + '.csv'
        name_carry_in = 'carry' + ver + '.csv'
        name_out = 'routes_renew' + ver + '.csv'

        df_flow = pd.read_csv(
            path_out + name_flow_in, delimiter=";", encoding='utf-8'
        )
        df_carry = pd.read_csv(
            path_src + name_carry_in, delimiter=";", encoding='utf-8'
        )
        f_hours = ['h' + str(h) for h in range(5, 24)]
        f_hours.append('day')

        df_flow = df_flow.groupby(['rnd'], as_index=False)[f_hours].sum()
        # if rnd:
        #     df_flow = df_flow.groupby(['rnd'], as_index=False)[f_hours].sum()
        #     df_renew = pd.DataFrame(data=df_flow['rnd'], columns=['rnd'])
        # else:
        #     df_flow = df_flow.groupby(['route'], as_index=False)[f_hours].sum()
        #     df_renew = pd.DataFrame(data=df_flow['route'], columns=['route'])

        hour_cols = df_flow.columns[1:]

        df_flow = df_flow.merge(df_carry, on=['rnd'], suffixes=('_f', '_c'))

        drop_cols = []
        for p in f_hours:
            flow_col = p + '_f'
            carry_col = p + '_c'
            df_flow[p] = round(df_flow[flow_col]/df_flow[carry_col], 3)
            drop_cols.append(flow_col)
            drop_cols.append(carry_col)

        df_flow = df_flow.replace(np.inf, 0)
        df_flow = df_flow.replace(np.nan, 0)
        df_flow = df_flow.drop(drop_cols, axis=1)
        df_flow = df_flow.rename(columns={'rnd': 'маршрут', 'day': 'день'})
        df_flow.to_csv(path_out + name_out, sep=';', encoding='utf-8', index=False)
    except Exception as app_error:
        app.logger.info(f'calc_renew error:\n{app_error}')


def calc_dynamic(ver):
    '''
    Расчет показателя динамической загрузки маршрутов по всем периодам времени из flow
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_flow_in = 'routes_flow_ext' + ver + '.csv'
        name_carry_in = 'carry' + ver + '.csv'
        name_rnd_in = 'rnd_short' + ver + '.csv'
        name_mean_distance_in = 'mean_distance' + ver + '.csv'
        name_out = 'routes_dynamic' + ver + '.csv'

        f_hours = ['h' + str(h) for h in range(5, 24)]
        f_hours.append('day')

        df_flow = pd.read_csv(
            path_out + name_flow_in, delimiter=";", encoding='utf-8'
        )
        df_carry = pd.read_csv(
            path_src + name_carry_in, delimiter=";", encoding='utf-8'
        )
        df_rnd = pd.read_csv(
            path_src + name_rnd_in, delimiter=";", encoding='utf-8'
        )
        df_mean_distance = pd.read_csv(
            path_out + name_mean_distance_in, delimiter=";", encoding='utf-8'
        )

        df_flow['route'] = df_flow['route'].astype(str)
        df_flow = df_flow.groupby(['route'], as_index=False)[f_hours].sum()

        df_carry['route'] = df_carry['rnd'].apply(lambda x: str(x).split('-')[0])
        df_carry = df_carry.groupby(['route'], as_index=False)[f_hours].sum()

        df_rnd['route'] = df_rnd['rnd'].apply(lambda x: str(x).split('-')[0])
        df_rnd = df_rnd.groupby(['route'], as_index=False).agg({'distance': 'sum'})
        df_rnd = df_rnd.rename(columns={'distance': 'route_distance'})

        df_mean_distance['route'] = df_mean_distance['route'].astype(str)
        df_mean_distance = df_mean_distance.rename(columns={'протяженность': 'mean_distance'})

        df_dynamic = df_flow.merge(df_carry, on=['route'], suffixes=('_f', '_c'))
        df_dynamic = df_dynamic.merge(df_rnd, on=['route'])
        df_dynamic = df_dynamic.merge(df_mean_distance, on=['route'])

        drop_cols = []
        for p in f_hours:
            flow_col = p + '_f'
            carry_col = p + '_c'
            df_dynamic[p] = round(
                (df_dynamic[flow_col] * df_dynamic['mean_distance']) /
                (df_dynamic['route_distance'] * df_dynamic[carry_col]),
                3
            )
            drop_cols.append(flow_col)
            drop_cols.append(carry_col)

        df_dynamic = df_dynamic.replace(np.inf, 0)
        df_dynamic = df_dynamic.replace(np.nan, 0)
        df_dynamic = df_dynamic.drop(drop_cols + ['route_distance', 'mean_distance'], axis=1)
        df_dynamic = df_dynamic.rename(columns={'route': 'маршрут', 'day': 'день'})
        df_dynamic.to_csv(path_out + name_out, sep=';', encoding='utf-8', index=False)
    except Exception as app_error:
        app.logger.exception('calc_dynamic error')
        raise


def corrs_show(ver, period, rnd):
    '''
    Отрисовка корреспонденций на линейном графике
    '''
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_rnd_in = 'route_load' + ver + '.csv'
        name_corrs_in = 'routes_flow_ext' + ver + '.csv'
        df_rnd = pd.read_csv(
            path_out + name_rnd_in, delimiter=";", encoding='utf-8'
        )
        df_corrs = pd.read_csv(
            path_out + name_corrs_in, delimiter=";", encoding='utf-8'
        )
        max_line_width = 8
        canvas_height = 220
        canvas_width = 1200
        vertical_amp = 180
        minimal_amp = 30
        minimal_line_width = 0.5
        offset = 5
        if rnd == '-': 
            rnd = df_corrs.loc[0, 'rnd']
        if period == '-':
            period = 'day'
        df_corrs = df_corrs.loc[df_corrs['rnd']==rnd]
        max_flow = df_corrs[period].max()

        df_rnd = df_rnd.loc[df_rnd['rnd']==rnd]
        rnd_x_steps = df_rnd.shape[0]
        step_distance = (canvas_width-2*offset)/rnd_x_steps
        svg = cairo.SVGSurface(f'{path_out}route_corrs.svg', canvas_width, canvas_height)
        ctx=cairo.Context(svg)
        k_counter = 1
        df_rnd['x_coord'] = 0
        # расстановка х координат - id ОП и отрисовка перегонов
        for r in df_rnd.itertuples():
            x_coord = int((r.order - 1)*step_distance)
            df_rnd.loc[r.Index, 'x_coord'] = x_coord
            ctx.set_line_width(6)
            ctx.set_source_rgba(0.2, 0.2, 0.2, 0.7)
            ctx.move_to(offset+x_coord-2, canvas_height/2)
            ctx.line_to(offset+x_coord+2, canvas_height/2)
            ctx.set_font_size(8)
            ctx.move_to(offset+x_coord-5, canvas_height/2+15)
            ctx.set_source_rgba(0.1, 0.1, 0.1, 0.9)
            ctx.show_text(str(getattr(r, 'source')))
            last_stop = str(getattr(r, 'target'))
        ctx.move_to(canvas_width-offset-2, canvas_height/2)
        ctx.line_to(canvas_width-offset+2, canvas_height/2)
        ctx.set_font_size(8)
        ctx.move_to(x_coord+step_distance-15, canvas_height/2+15)
        ctx.show_text(last_stop)
        ctx.stroke()
        # отрисовка корреспонденций
        for c in df_corrs.itertuples():
            x_s = df_rnd.loc[df_rnd['source']==c.source, 'x_coord'].values[0] + offset
            if len(df_rnd.loc[df_rnd['source']==c.target, 'x_coord']) == 0:
                x_e = canvas_width - offset
            else:
                x_e = df_rnd.loc[df_rnd['source']==c.target, 'x_coord'].values[0] + offset
            y_se = canvas_height/2
            x_mid = x_s + (x_e-x_s)/2 + offset
            k_y_mid = (x_e-x_s)/canvas_width
            y_mid = canvas_height/2 + k_counter*(vertical_amp*k_y_mid)
            ctx.set_line_width(minimal_line_width + max_line_width*getattr(c, period)/max_flow)
            ctx.move_to(x_s, y_se)
            ctx.set_source_rgba(0, 0.7, 0, 0.3)
            ctx.curve_to(x_s, y_se, x_mid, y_mid, x_e, y_se)
            ctx.stroke()
            k_counter *= -1
        # легенда
        ctx.set_line_width(3)
        ctx.set_source_rgb(1, 0.6, 0)
        ctx.move_to(offset, canvas_height/2)
        ctx.line_to(canvas_width-offset, canvas_height/2)
        ctx.stroke()
         # линия
        ctx.move_to(5, 25-max_line_width/2)
        ctx.set_line_width(max_line_width)
        ctx.set_source_rgb(0, 0.65, 0)
        ctx.line_to(25, 25-max_line_width/2)
        ctx.stroke()
        # подпись легенды
        ctx.move_to(30, 25)
        ctx.set_font_size(12)
        ctx.set_source_rgba(0.2, 0.2, 0.2, 0.9)
        ctx.show_text(f'{rnd} макс. поток на корреспонденциях: {max_flow} пасс/{period}')
        svg.finish()
        svg.flush()
        img_path = os.path.relpath(path_out + 'route_corrs.svg', app.static_folder).replace(os.sep, '/')
        return img_path
    except Exception as app_error:
        app.logger.exception('corrs_show error')
        return None



def get_rnd_line_load(ver, rnd):
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_in = 'route_load' + ver + '.csv'
        df_load = pd.read_csv(
            path_out + name_in, delimiter=";", encoding='utf-8'
        )
        df_load = df_load[df_load['rnd']==rnd]
        df_load = df_load.drop(['rnd'], axis=1) #, 'route', 'day_carry', 'day_renew'
        df_load = df_load.rename(columns={'day': 'день', 'source': 'из ОП', 'target': 'в ОП', 'order': 'порядок', 'distance': 'расстояние'})
        load_string = df_load.style.set_properties(**{'background-color': '#f2f2f2', 'color': 'black', 
                                'border-color': 'white'}).bar(subset=['h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14','h15','h16', 'h17',
                          'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'день'], color='#85e085').hide(axis="index").format(precision=2).to_html()
        return load_string
    except Exception as app_error:
        app.logger.info(f'get_rnd_line_load error:\n{app_error}')


def load_mean_distance(ver):
    try:
        path_src, path_out = path_gen(session['username'], 'eff', ver[1:], False)
        name_in = 'mean_distance' + ver + '.csv'
        df_mean = pd.read_csv(
            path_out + name_in,
            delimiter=";",
            encoding='utf-8',
            index_col='route',
        )
        df_mean.sort_values(by='route', inplace=True)
        # df_mean.reset_index(inplace=True)
        # df_mean = df_mean.rename(columns={'route': 'маршрут'})
        # df_mean.astype({'маршрут': 'int32'}).dtypes
        df_mean = df_mean.T
        mean_string = df_mean.style.set_properties(**{'background-color': '#f2f2f2', 'color': 'black', 
                                'border-color': 'white'}).format(precision=2).to_html()
        return mean_string
    except Exception as app_error:
        app.logger.info(f'load_mean_distance error:\n{app_error}')


def get_rnd_line_renew(path_src, ver, rnd):
    '''
    Отображение загрузки межостановочных перегонов по выбраннаму маршруту
    '''
    try:
        name_in = 'route_renew_segments' + ver + '.csv'
        df_renew = pd.read_csv(
            path_src + name_in, delimiter=";", encoding='utf-8'
        )
        df_renew['rnd'] = df_renew['rnd'].astype(str).str.strip()
        df_renew = df_renew[df_renew['rnd'] == str(rnd).strip()]
        df_renew = df_renew[['rnd', 'order', 'day_renew']]
        return df_renew.sort_values('order')
    except Exception:
        app.logger.exception('get_rnd_line_renew error')
        return pd.DataFrame(columns=['rnd', 'order', 'day_renew'])


def create_histo_renew(df, userpath_out, name_out, title, x_name, y_name):
    if df.empty:
        return False

    sns.set_theme(palette='pastel')
    fig, axis = plt.subplots()
    axis.plot(df['order'], df['day_renew'], color='g', marker='o', markersize=3)
    axis.set(xlabel=x_name, ylabel=y_name, title=title)
    fig.set_size_inches(15, 4.5)
    fig.tight_layout()
    fig.savefig(f'{userpath_out}{name_out}.png', dpi=80)
    plt.close(fig)
    return True
