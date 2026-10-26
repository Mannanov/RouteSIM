from flask import Flask, session
from datetime import timedelta
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from dotenv import dotenv_values


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')

app.logger.setLevel(logging.INFO)
app.logger.handlers.clear()

log_formatter = logging.Formatter(
    fmt='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
app.logger.addHandler(stream_handler)

log_dir = os.path.join(BASE_DIR, 'log')
os.makedirs(log_dir, exist_ok=True)
file_handler = RotatingFileHandler(os.path.join(log_dir, 'app_info'), maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(log_formatter)
app.logger.addHandler(file_handler)

app.logger.propagate = False

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)
werkzeug_logger.handlers.clear()
werkzeug_logger.addHandler(stream_handler)
werkzeug_logger.addHandler(file_handler)
werkzeug_logger.propagate = False

app.logger.info('Server restarted')

config = {
    **dotenv_values(ENV_PATH),
    **os.environ,
}
app.config['MAX_CONTENT_LENGTH'] = 512*1024*1024
app.config['GEO_MAX_STOP_SNAP_DISTANCE_M'] = config.get('GEO_MAX_STOP_SNAP_DISTANCE_M', 100.0)
app.config['SECRET_KEY'] = config['SKEY']

import models
from forecast_engine import forecast_service

forecast_service.init_project_storage()

recovered_calculations = models.recover_interrupted_calculations()
if recovered_calculations:
    app.logger.warning(
        'Marked %s interrupted calculation(s) as failed after application restart',
        recovered_calculations,
    )

import routes
from demand_routes import demand_bp
from forecast_routes import forecast_bp

app.register_blueprint(demand_bp)
app.register_blueprint(forecast_bp)

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=60)
