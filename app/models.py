import os
import sqlite3
from datetime import date, datetime, timedelta, timezone

from dotenv import dotenv_values
from werkzeug.security import generate_password_hash
from forecast_engine.config import METHOD_VERSION, LEGACY_METHOD_VERSION


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
config = {
    **dotenv_values(ENV_PATH),
    **os.environ,
}
db_path = config.get('SQLITE_PATH', 'routesim.db')
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, db_path))

sqlite3.register_adapter(date, lambda value: value.isoformat())
sqlite3.register_converter('DATE', lambda value: date.fromisoformat(value.decode('utf-8')))

PROJECT_TABLES = {
    'los': 'los_projects',
    'eff': 'eff_projects',
    'geo': 'geo_projects',
    'demand': 'demand_projects',
    'forecast': 'forecast_projects',
}

LOS_STATUS_CALCULATING = 3
LOS_STATUS_COMPLETED = 4
LOS_STATUS_FAILED = 5
EFF_STATUS_CALCULATING = 2
EFF_STATUS_COMPLETED = 3
EFF_STATUS_FAILED = 4
DEMAND_STATUS_CALCULATING = 2
DEMAND_STATUS_COMPLETED = 3
DEMAND_STATUS_FAILED = 4
FORECAST_STATUS_CALCULATING = 2
FORECAST_STATUS_COMPLETED = 3
FORECAST_STATUS_FAILED = 4

CALCULATION_TIMEOUT_MINUTES = int(config.get('CALCULATION_TIMEOUT_MINUTES', 24 * 60))


def get_db_cur():
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return conn, cur


def init_db():
    conn, cur = get_db_cur()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            expire_date DATE NOT NULL,
            tarif_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_tarif (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarif_name TEXT NOT NULL,
            net_size INTEGER NOT NULL,
            disk_limit INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS los_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            proj_name TEXT NOT NULL,
            proj_descr TEXT NOT NULL,
            date_crt DATE NOT NULL,
            status_src INTEGER NOT NULL,
            status_geo INTEGER NOT NULL,
            status_prj INTEGER NOT NULL,
            calculation_started_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_status_los_prj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status_prj INTEGER NOT NULL,
            status_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS eff_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            proj_name TEXT NOT NULL,
            proj_descr TEXT NOT NULL,
            date_crt DATE NOT NULL,
            status_src INTEGER NOT NULL,
            status_prj INTEGER NOT NULL,
            calculation_started_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_status_eff_prj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status_prj INTEGER NOT NULL,
            status_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS geo_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            proj_name TEXT NOT NULL,
            proj_descr TEXT NOT NULL,
            date_crt DATE NOT NULL,
            status_src INTEGER NOT NULL,
            status_prj INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS demand_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            proj_name TEXT NOT NULL,
            proj_descr TEXT NOT NULL,
            date_crt DATE NOT NULL,
            status_src INTEGER NOT NULL,
            status_prj INTEGER NOT NULL,
            calculation_period TEXT,
            demand_profile_mode TEXT,
            pt_share REAL,
            pt_share_intrazonal REAL,
            access_radius_km REAL,
            min_overlap_share REAL,
            calibration_mode TEXT,
            normalize_target_volume INTEGER,
            total_boardings_period INTEGER,
            transfer_factor REAL,
            base_period_trip_volume REAL,
            target_period_trip_volume REAL,
            effective_pt_share REAL,
            effective_pt_share_status TEXT,
            calculation_started_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_status_demand_prj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status_prj INTEGER NOT NULL,
            status_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            proj_name TEXT NOT NULL,
            proj_descr TEXT NOT NULL,
            date_crt DATE NOT NULL,
            status_src INTEGER NOT NULL,
            status_prj INTEGER NOT NULL,
            recalculation_mode TEXT,
            work_service_chain_share REAL,
            hour_work_to_service INTEGER,
            hour_service_to_home INTEGER,
            analog_count INTEGER,
            los_neighbor_radius_km REAL,
            los_neighbor_count INTEGER,
            ratio_min REAL, -- исторические значения, новый расчет их не использует
            ratio_max REAL,
            distance_floor_km REAL,
            time_floor_hours REAL,
            ipf_tolerance REAL,
            ipf_max_iterations INTEGER,
            t50_work REAL,
            t50_education REAL,
            t50_services REAL,
            t50_other REAL,
            total_demand REAL,
            unallocated_demand REAL,
            unallocated_direct_visits REAL,
            unallocated_return_legs REAL,
            unallocated_trip_legs_total REAL,
            control_errors INTEGER,
            calculation_started_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_status_forecast_prj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status_prj INTEGER NOT NULL,
            status_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ref_status_geo_prj (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status_prj INTEGER NOT NULL,
            status_name TEXT NOT NULL
        )
    """)

    default_users = (
        ('user', '1111', '2099-12-15', 'L3'),
        ('cappower', '1111', '2099-12-15', 'L3'),
    )
    for username, password, expire_date, tarif_name in default_users:
        cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cur.fetchone() is None:
            cur.execute(
                """INSERT INTO users
                   (username, password, expire_date, tarif_name)
                   VALUES (?, ?, ?, ?)""",
                (
                    username,
                    generate_password_hash(password),
                    expire_date,
                    tarif_name,
                ),
            )

    cur.execute("SELECT COUNT(*) AS cnt FROM ref_tarif")
    if cur.fetchone()['cnt'] == 0:
        values = [('L1', 1500, 1000), ('L2', 3000, 2500), ('L3', 7000, 5000)]
        cur.executemany(
            "INSERT INTO ref_tarif (tarif_name, net_size, disk_limit) VALUES (?, ?, ?)",
            values,
        )

    cur.execute("SELECT COUNT(*) AS cnt FROM ref_status_los_prj")
    if cur.fetchone()['cnt'] == 0:
        values = [
            (0, 'загрузка файлов'),
            (1, 'описания сети загружены'),
            (2, 'описание сети и геометрия загружены'),
            (LOS_STATUS_CALCULATING, 'производится расчет показателей'),
            (LOS_STATUS_COMPLETED, 'расчет показателей завершен'),
            (LOS_STATUS_FAILED, 'ошибка расчета'),
        ]
        cur.executemany(
            "INSERT INTO ref_status_los_prj (status_prj, status_name) VALUES (?, ?)",
            values,
        )

    cur.execute("SELECT COUNT(*) AS cnt FROM ref_status_eff_prj")
    if cur.fetchone()['cnt'] == 0:
        values = [
            (0, 'загрузка файлов'),
            (1, 'данные загружены'),
            (EFF_STATUS_CALCULATING, 'производится расчет показателей'),
            (EFF_STATUS_COMPLETED, 'расчет показателей завершен'),
            (EFF_STATUS_FAILED, 'ошибка расчета'),
        ]
        cur.executemany(
            "INSERT INTO ref_status_eff_prj (status_prj, status_name) VALUES (?, ?)",
            values,
        )

    cur.execute("SELECT COUNT(*) AS cnt FROM ref_status_geo_prj")
    if cur.fetchone()['cnt'] == 0:
        values = [(0, 'загрузка файлов'), (1, 'данные загружены'), (2, 'граф подготовлен')]
        cur.executemany(
            "INSERT INTO ref_status_geo_prj (status_prj, status_name) VALUES (?, ?)",
            values,
        )

    cur.execute("SELECT COUNT(*) AS cnt FROM ref_status_demand_prj")
    if cur.fetchone()['cnt'] == 0:
        values = [
            (0, 'загрузка файлов'),
            (1, 'данные загружены'),
            (DEMAND_STATUS_CALCULATING, 'производится расчет спроса'),
            (DEMAND_STATUS_COMPLETED, 'расчет спроса завершен'),
            (DEMAND_STATUS_FAILED, 'ошибка расчета'),
        ]
        cur.executemany(
            "INSERT INTO ref_status_demand_prj (status_prj, status_name) VALUES (?, ?)",
            values,
        )

    cur.execute("SELECT COUNT(*) AS cnt FROM ref_status_forecast_prj")
    if cur.fetchone()['cnt'] == 0:
        values = [
            (0, 'загрузка файлов'),
            (1, 'данные загружены'),
            (FORECAST_STATUS_CALCULATING, 'производится расчет спроса'),
            (FORECAST_STATUS_COMPLETED, 'прогноз спроса завершен'),
            (FORECAST_STATUS_FAILED, 'ошибка расчета'),
        ]
        cur.executemany(
            "INSERT INTO ref_status_forecast_prj (status_prj, status_name) VALUES (?, ?)",
            values,
        )

    # Миграция существующих проектов без изменения сохранённых результатов.
    demand_columns = {row[1] for row in cur.execute('PRAGMA table_info(demand_projects)')}
    if 'pt_share_intrazonal' not in demand_columns:
        cur.execute('ALTER TABLE demand_projects ADD COLUMN pt_share_intrazonal REAL')
    forecast_columns = {row[1] for row in cur.execute('PRAGMA table_info(forecast_projects)')}
    for column in ('methodology_version', 'config_json', 'transfer_status', 'config_history_json'):
        if column not in forecast_columns:
            cur.execute(f'ALTER TABLE forecast_projects ADD COLUMN {column} TEXT')
    cur.execute('UPDATE forecast_projects SET methodology_version = ? WHERE methodology_version IS NULL',
                (LEGACY_METHOD_VERSION,))
    if 'work_service_chain_share' not in forecast_columns:
        cur.execute('ALTER TABLE forecast_projects ADD COLUMN work_service_chain_share REAL')
    for column in ('hour_work_to_service', 'hour_service_to_home'):
        if column not in forecast_columns:
            cur.execute(f'ALTER TABLE forecast_projects ADD COLUMN {column} INTEGER')
    for column in (
        'unallocated_direct_visits', 'unallocated_return_legs',
        'unallocated_trip_legs_total',
    ):
        if column not in forecast_columns:
            cur.execute(f'ALTER TABLE forecast_projects ADD COLUMN {column} REAL')
    conn.commit()
    cur.close()
    conn.close()


def _resolve_project_table(proj_type):
    try:
        return PROJECT_TABLES[proj_type]
    except KeyError as exc:
        raise ValueError(f'Unknown project type: {proj_type}') from exc


def _project_date(year, month, day):
    return date(year, month, day)


def get_user_limits(tarif_name):
    conn, cur = get_db_cur()
    cur.execute("SELECT * FROM ref_tarif WHERE tarif_name = ?", (tarif_name,))
    user_limits = cur.fetchone()
    cur.close()
    conn.close()
    return user_limits


def get_acc_by_username(username):
    conn, cur = get_db_cur()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    account = cur.fetchone()
    cur.close()
    conn.close()
    return account


def get_acc_by_id(id_session):
    conn, cur = get_db_cur()
    id_value = id_session[0] if isinstance(id_session, (list, tuple)) else id_session
    cur.execute("SELECT * FROM users WHERE id = ?", (id_value,))
    account = cur.fetchone()
    cur.close()
    conn.close()
    return account


def update_user_password(username, password):
    conn, cur = get_db_cur()
    cur.execute(
        "UPDATE users SET password = ? WHERE username = ?",
        (generate_password_hash(password), username),
    )
    conn.commit()
    updated = cur.rowcount
    cur.close()
    conn.close()
    return updated == 1


### LOS ###
def los_get_projects(username):
    recover_stale_calculations(username)
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT los_projects.proj_name, los_projects.proj_descr, los_projects.date_crt, ref_status_los_prj.status_name
           FROM los_projects JOIN ref_status_los_prj ON los_projects.status_prj = ref_status_los_prj.status_prj
           WHERE username = ? ORDER BY los_projects.id DESC""",
        (username,),
    )
    los_projects = cur.fetchall()
    cur.close()
    conn.close()
    return los_projects


def los_get_one_project(username, proj_name):
    conn, cur = get_db_cur()
    cur.execute("SELECT * FROM los_projects WHERE username = ? AND proj_name = ?", (username, proj_name))
    los_projects = cur.fetchone()
    cur.close()
    conn.close()
    return los_projects


def los_commit_new_project(username, proj_name, proj_descr, year, month, day):
    conn, cur = get_db_cur()
    cur.execute(
        "INSERT INTO los_projects (username, proj_name, proj_descr, date_crt, status_src, status_geo, status_prj) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, proj_name, proj_descr, _project_date(year, month, day), 0, 0, 0),
    )
    conn.commit()
    cur.close()
    conn.close()


def los_commit_update_status_src(username, prj, status_src):
    conn, cur = get_db_cur()
    cur.execute(
        "UPDATE los_projects SET status_src=? WHERE username = ? AND proj_name = ?",
        (status_src, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def los_commit_update_status_geo(username, prj, status_geo):
    conn, cur = get_db_cur()
    cur.execute(
        "UPDATE los_projects SET status_geo=? WHERE username = ? AND proj_name = ?",
        (status_geo, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def los_commit_update_status_prj(username, prj, status_prj):
    conn, cur = get_db_cur()
    cur.execute(
        """UPDATE los_projects
           SET status_prj=?, calculation_started_at=NULL
           WHERE username = ? AND proj_name = ?""",
        (status_prj, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def delete_project(username, proj_type, proj_name):
    conn, cur = get_db_cur()
    project_table = _resolve_project_table(proj_type)
    cur.execute(
        f"DELETE FROM {project_table} WHERE username = ? AND proj_name = ?",
        (username, proj_name),
    )
    conn.commit()
    cur.close()
    conn.close()


### EFF ###
def eff_commit_new_project(username, proj_name, proj_descr, year, month, day):
    conn, cur = get_db_cur()
    cur.execute(
        "INSERT INTO eff_projects (username, proj_name, proj_descr, date_crt, status_src, status_prj) VALUES (?, ?, ?, ?, ?, ?)",
        (username, proj_name, proj_descr, _project_date(year, month, day), 0, 0),
    )
    conn.commit()
    cur.close()
    conn.close()


def eff_commit_update_status_src(username, prj, status_src):
    conn, cur = get_db_cur()
    cur.execute(
        "UPDATE eff_projects SET status_src=? WHERE username = ? AND proj_name = ?",
        (status_src, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def eff_commit_update_status_prj(username, prj, status_prj):
    conn, cur = get_db_cur()
    cur.execute(
        """UPDATE eff_projects
           SET status_prj=?, calculation_started_at=NULL
           WHERE username = ? AND proj_name = ?""",
        (status_prj, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def eff_get_projects(username):
    recover_stale_calculations(username)
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT eff_projects.proj_name, eff_projects.proj_descr, eff_projects.date_crt, ref_status_eff_prj.status_name
           FROM eff_projects JOIN ref_status_eff_prj ON eff_projects.status_prj = ref_status_eff_prj.status_prj
           WHERE username = ? ORDER BY eff_projects.id DESC""",
        (username,),
    )
    eff_projects = cur.fetchall()
    cur.close()
    conn.close()
    return eff_projects


def eff_get_one_project(username, proj_name):
    conn, cur = get_db_cur()
    cur.execute("SELECT * FROM eff_projects WHERE username = ? AND proj_name = ?", (username, proj_name))
    eff_projects = cur.fetchone()
    cur.close()
    conn.close()
    return eff_projects


### DEMAND ###
def demand_get_projects(username):
    """Возвращает demand-проекты пользователя."""
    recover_stale_calculations(username)
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT demand_projects.proj_name, demand_projects.proj_descr,
                  demand_projects.date_crt, ref_status_demand_prj.status_name
           FROM demand_projects
           JOIN ref_status_demand_prj
             ON demand_projects.status_prj = ref_status_demand_prj.status_prj
           WHERE username = ? ORDER BY demand_projects.id DESC""",
        (username,),
    )
    projects = cur.fetchall()
    cur.close()
    conn.close()
    return projects


def demand_get_one_project(username, proj_name):
    """Возвращает один demand-проект пользователя."""
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT * FROM demand_projects
           WHERE username = ? AND proj_name = ?""",
        (username, proj_name),
    )
    project = cur.fetchone()
    cur.close()
    conn.close()
    return project


def demand_commit_new_project(
    username, proj_name, proj_descr, year, month, day
):
    """Создаёт demand-проект со статусом загрузки исходных данных."""
    conn, cur = get_db_cur()
    cur.execute(
        """INSERT INTO demand_projects
           (username, proj_name, proj_descr, date_crt, status_src, status_prj)
           VALUES (?, ?, ?, ?, 0, 0)""",
        (
            username,
            proj_name,
            proj_descr,
            _project_date(year, month, day),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def demand_commit_update_status_src(username, proj_name, status_src):
    """Обновляет число загруженных файлов demand-проекта."""
    conn, cur = get_db_cur()
    cur.execute(
        """UPDATE demand_projects SET status_src=?
           WHERE username = ? AND proj_name = ?""",
        (status_src, username, proj_name),
    )
    conn.commit()
    cur.close()
    conn.close()


def demand_commit_update_status_prj(
    username,
    proj_name,
    status_prj,
    pt_share=None,
    calculation_period=None,
    demand_profile_mode=None,
    access_radius_km=None,
    min_overlap_share=None,
    calibration_mode=None,
    normalize_target_volume=None,
    total_boardings_period=None,
    transfer_factor=None,
    base_period_trip_volume=None,
    target_period_trip_volume=None,
    effective_pt_share=None,
    effective_pt_share_status=None,
    pt_share_intrazonal=None,
):
    """Обновляет статус и параметры расчёта demand-проекта."""
    conn, cur = get_db_cur()
    cur.execute(
        """UPDATE demand_projects
           SET status_prj=?, pt_share=COALESCE(?, pt_share),
               pt_share_intrazonal=COALESCE(?, pt_share_intrazonal),
               calculation_period=COALESCE(?, calculation_period),
               demand_profile_mode=COALESCE(?, demand_profile_mode),
               access_radius_km=COALESCE(?, access_radius_km),
               min_overlap_share=COALESCE(?, min_overlap_share),
               calibration_mode=COALESCE(?, calibration_mode),
               normalize_target_volume=COALESCE(
                   ?, normalize_target_volume
               ),
               total_boardings_period=COALESCE(
                   ?, total_boardings_period
               ),
               transfer_factor=COALESCE(?, transfer_factor),
               base_period_trip_volume=COALESCE(
                   ?, base_period_trip_volume
               ),
               target_period_trip_volume=COALESCE(
                   ?, target_period_trip_volume
               ),
               effective_pt_share=COALESCE(?, effective_pt_share),
               effective_pt_share_status=COALESCE(
                   ?, effective_pt_share_status
               ),
               calculation_started_at=NULL
           WHERE username = ? AND proj_name = ?""",
        (
            status_prj,
            pt_share,
            pt_share_intrazonal,
            calculation_period,
            demand_profile_mode,
            access_radius_km,
            min_overlap_share,
            calibration_mode,
            normalize_target_volume,
            total_boardings_period,
            transfer_factor,
            base_period_trip_volume,
            target_period_trip_volume,
            effective_pt_share,
            effective_pt_share_status,
            username,
            proj_name,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


### FORECAST ###
def forecast_get_projects(username):
    """Возвращает проекты прогноза спроса пользователя."""
    recover_stale_calculations(username)
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT forecast_projects.proj_name,
                  forecast_projects.proj_descr,
                  forecast_projects.date_crt,
                  ref_status_forecast_prj.status_name
           FROM forecast_projects
           JOIN ref_status_forecast_prj
             ON forecast_projects.status_prj =
                ref_status_forecast_prj.status_prj
           WHERE username = ? ORDER BY forecast_projects.id DESC""",
        (username,),
    )
    projects = cur.fetchall()
    cur.close()
    conn.close()
    return projects


def forecast_get_one_project(username, proj_name):
    """Возвращает один проект прогноза спроса."""
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT * FROM forecast_projects
           WHERE username = ? AND proj_name = ?""",
        (username, proj_name),
    )
    project = cur.fetchone()
    cur.close()
    conn.close()
    return project


def forecast_commit_new_project(
    username, proj_name, proj_descr, year, month, day
):
    """Создаёт проект прогноза со статусом загрузки данных."""
    conn, cur = get_db_cur()
    cur.execute(
        """INSERT INTO forecast_projects
           (username, proj_name, proj_descr, date_crt, status_src, status_prj, methodology_version)
           VALUES (?, ?, ?, ?, 0, 0, ?)""",
        (
            username,
            proj_name,
            proj_descr,
            _project_date(year, month, day),
            METHOD_VERSION,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def forecast_commit_update_status_src(username, proj_name, status_src):
    """Обновляет состояние загрузки исходного набора."""
    conn, cur = get_db_cur()
    cur.execute(
        """UPDATE forecast_projects SET status_src = ?
           WHERE username = ? AND proj_name = ?""",
        (status_src, username, proj_name),
    )
    conn.commit()
    cur.close()
    conn.close()


def forecast_commit_update_status_prj(
    username, proj_name, status_prj, recalculation_mode=None, analog_count=None,
    los_neighbor_radius_km=None, los_neighbor_count=None,
    distance_floor_km=None, time_floor_hours=None,
    ipf_tolerance=None, ipf_max_iterations=None, t50_work=None,
    t50_education=None, t50_services=None, t50_other=None,
    total_demand=None, unallocated_direct_visits=None,
    unallocated_return_legs=None, unallocated_trip_legs_total=None,
    control_errors=None,
    work_service_chain_share=None,
    hour_work_to_service=None, hour_service_to_home=None,
    methodology_version=None, config_json=None, transfer_status=None,
):
    """Обновляет статус, параметры и итоги forecast-проекта."""
    conn, cur = get_db_cur()
    cur.execute(
        """UPDATE forecast_projects
           SET status_prj = ?,
               methodology_version = COALESCE(?, methodology_version),
               config_json = COALESCE(?, config_json),
               transfer_status = COALESCE(?, transfer_status),
               recalculation_mode = COALESCE(?, recalculation_mode),
               work_service_chain_share = COALESCE(?, work_service_chain_share),
               hour_work_to_service = COALESCE(?, hour_work_to_service),
               hour_service_to_home = COALESCE(?, hour_service_to_home),
               analog_count = COALESCE(?, analog_count),
               los_neighbor_radius_km = COALESCE(?, los_neighbor_radius_km),
               los_neighbor_count = COALESCE(?, los_neighbor_count),
               distance_floor_km = COALESCE(?, distance_floor_km),
               time_floor_hours = COALESCE(?, time_floor_hours),
               ipf_tolerance = COALESCE(?, ipf_tolerance),
               ipf_max_iterations = COALESCE(?, ipf_max_iterations),
               t50_work = COALESCE(?, t50_work),
               t50_education = COALESCE(?, t50_education),
               t50_services = COALESCE(?, t50_services),
               t50_other = COALESCE(?, t50_other),
               total_demand = COALESCE(?, total_demand),
               unallocated_direct_visits = COALESCE(?, unallocated_direct_visits),
               unallocated_return_legs = COALESCE(?, unallocated_return_legs),
               unallocated_trip_legs_total = COALESCE(?, unallocated_trip_legs_total),
               control_errors = COALESCE(?, control_errors),
               calculation_started_at = NULL
           WHERE username = ? AND proj_name = ?""",
        (
            status_prj,
            methodology_version,
            config_json,
            transfer_status,
            recalculation_mode,
            work_service_chain_share,
            hour_work_to_service,
            hour_service_to_home,
            analog_count,
            los_neighbor_radius_km,
            los_neighbor_count,
            distance_floor_km,
            time_floor_hours,
            ipf_tolerance,
            ipf_max_iterations,
            t50_work,
            t50_education,
            t50_services,
            t50_other,
            total_demand,
            unallocated_direct_visits,
            unallocated_return_legs,
            unallocated_trip_legs_total,
            control_errors,
            username,
            proj_name,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def forecast_store_config(username, proj_name, config_json):
    conn, cur = get_db_cur()
    cur.execute('UPDATE forecast_projects SET config_json = ? WHERE username = ? AND proj_name = ?',
                (config_json, username, proj_name))
    conn.commit()
    conn.close()


def _calculation_project_states(cur):
    states = [
        ('los_projects', LOS_STATUS_CALCULATING, LOS_STATUS_FAILED),
        ('eff_projects', EFF_STATUS_CALCULATING, EFF_STATUS_FAILED),
        ('demand_projects', DEMAND_STATUS_CALCULATING, DEMAND_STATUS_FAILED),
        ('forecast_projects', FORECAST_STATUS_CALCULATING, FORECAST_STATUS_FAILED),
    ]
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'forecast_bl_projects'").fetchone():
        states.append(('forecast_bl_projects', FORECAST_STATUS_CALCULATING, FORECAST_STATUS_FAILED))
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'forecast_bl_2_projects'").fetchone():
        states.append(('forecast_bl_2_projects', FORECAST_STATUS_CALCULATING, FORECAST_STATUS_FAILED))
    return states


def _calculation_running(cur, username):
    states = _calculation_project_states(cur)
    clauses = [f'EXISTS(SELECT 1 FROM {table} WHERE username = ? AND status_prj = ?)' for table, _, _ in states]
    parameters = [value for _, status, _ in states for value in (username, status)]
    return cur.execute('SELECT ' + ' OR '.join(clauses), parameters).fetchone()[0]


def mark_calculation_started(username, proj_type, proj_name):
    if proj_type == 'los':
        table_name = 'los_projects'
        calculating_status = LOS_STATUS_CALCULATING
    elif proj_type == 'eff':
        table_name = 'eff_projects'
        calculating_status = EFF_STATUS_CALCULATING
    elif proj_type == 'demand':
        table_name = 'demand_projects'
        calculating_status = DEMAND_STATUS_CALCULATING
    elif proj_type == 'forecast':
        table_name = 'forecast_projects'
        calculating_status = FORECAST_STATUS_CALCULATING
    else:
        raise ValueError(f'Unsupported calculation project type: {proj_type}')

    started_at = datetime.now(timezone.utc).isoformat()
    conn, cur = get_db_cur()
    try:
        cur.execute('BEGIN IMMEDIATE')
        if _calculation_running(cur, username):
            raise ValueError('Дождитесь завершения расчёта другого проекта')
        cur.execute(
            f"""UPDATE {table_name}
                SET status_prj = ?, calculation_started_at = ?
                WHERE username = ? AND proj_name = ?""",
            (calculating_status, started_at, username, proj_name),
        )
        conn.commit()
    finally:
        conn.close()


def recover_stale_calculations(username):
    stale_before = (datetime.now(timezone.utc) - timedelta(
        minutes=CALCULATION_TIMEOUT_MINUTES
    )).isoformat()
    conn, cur = get_db_cur()
    recovered = 0
    for table_name, calculating_status, failed_status in _calculation_project_states(cur):
        cur.execute(
            f"""UPDATE {table_name}
                SET status_prj = ?, calculation_started_at = NULL
                WHERE username = ? AND status_prj = ?
                  AND (calculation_started_at IS NULL OR calculation_started_at < ?)""",
            (failed_status, username, calculating_status, stale_before),
        )
        recovered += cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return recovered


def recover_interrupted_calculations():
    """Release calculations that cannot survive an application restart."""
    conn, cur = get_db_cur()
    recovered = 0
    for table_name, calculating_status, failed_status in _calculation_project_states(cur):
        cur.execute(
            f"""UPDATE {table_name}
                SET status_prj = ?, calculation_started_at = NULL
                WHERE status_prj = ?""",
            (failed_status, calculating_status),
        )
        recovered += cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return recovered


def check_calculate_process(username):
    recover_stale_calculations(username)
    conn, cur = get_db_cur()
    calc_run = _calculation_running(cur, username)
    cur.close()
    conn.close()
    return calc_run


### GEO ###
def geo_get_projects(username):
    conn, cur = get_db_cur()
    cur.execute(
        """SELECT geo_projects.proj_name, geo_projects.proj_descr, geo_projects.date_crt, ref_status_geo_prj.status_name
           FROM geo_projects JOIN ref_status_geo_prj ON geo_projects.status_prj = ref_status_geo_prj.status_prj
           WHERE username = ? ORDER BY geo_projects.id DESC""",
        (username,),
    )
    geo_projects = cur.fetchall()
    cur.close()
    conn.close()
    return geo_projects


def geo_commit_new_project(username, proj_name, proj_descr, year, month, day):
    conn, cur = get_db_cur()
    cur.execute(
        "INSERT INTO geo_projects (username, proj_name, proj_descr, date_crt, status_src, status_prj) VALUES (?, ?, ?, ?, ?, ?)",
        (username, proj_name, proj_descr, _project_date(year, month, day), 0, 0),
    )
    conn.commit()
    cur.close()
    conn.close()


def geo_commit_update_status_src(username, prj, status_src):
    conn, cur = get_db_cur()
    cur.execute(
        "UPDATE geo_projects SET status_src=? WHERE username = ? AND proj_name = ?",
        (status_src, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def geo_commit_update_status_prj(username, prj, status_prj):
    conn, cur = get_db_cur()
    cur.execute(
        "UPDATE geo_projects SET status_prj=? WHERE username = ? AND proj_name = ?",
        (status_prj, username, prj),
    )
    conn.commit()
    cur.close()
    conn.close()


def geo_get_one_project(username, proj_name):
    conn, cur = get_db_cur()
    cur.execute("SELECT * FROM geo_projects WHERE username = ? AND proj_name = ?", (username, proj_name))
    geo_projects = cur.fetchone()
    cur.close()
    conn.close()
    return geo_projects


### USERS ###
def get_users():
    conn, cur = get_db_cur()
    cur.execute("SELECT username, expire_date, tarif_name FROM users")
    users_data = cur.fetchall()
    cur.close()
    conn.close()
    return users_data


def get_ref_tarif():
    conn, cur = get_db_cur()
    cur.execute("SELECT tarif_name FROM ref_tarif")
    tarif_data = cur.fetchall()
    cur.close()
    conn.close()
    return tarif_data


def commit_new_user(username, password, expire_date, tarif_name):
    conn, cur = get_db_cur()
    cur.execute(
        "INSERT INTO users (username, password, expire_date, tarif_name) VALUES (?, ?, ?, ?)",
        (username, generate_password_hash(password), expire_date, tarif_name),
    )
    conn.commit()
    cur.close()
    conn.close()


def update_new_user(username, password, expire_date, tarif_name):
    conn, cur = get_db_cur()
    if password:
        cur.execute(
            """UPDATE users
               SET password = ?, expire_date = ?, tarif_name = ?
               WHERE username = ?""",
            (
                generate_password_hash(password),
                expire_date,
                tarif_name,
                username,
            ),
        )
    else:
        cur.execute(
            """UPDATE users
               SET expire_date = ?, tarif_name = ?
               WHERE username = ?""",
            (expire_date, tarif_name, username),
        )
    conn.commit()
    cur.close()
    conn.close()
