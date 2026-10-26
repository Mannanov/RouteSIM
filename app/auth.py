from flask import request, session, redirect, url_for, render_template, flash
from werkzeug.security import check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, PasswordField
from wtforms.validators import DataRequired, EqualTo, Length
from rsim import app
from models import get_acc_by_username, get_acc_by_id, get_user_limits, update_user_password
from serv_funcs import check_disk_size
import time
# from run_server import logger


class LoginForm(FlaskForm):
    username = StringField("Пользователь: ", validators=[DataRequired()])
    password = PasswordField("Пароль: ", validators=[DataRequired()])
    submit = SubmitField("Войти")


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField(
        "Текущий пароль",
        validators=[DataRequired()],
    )
    new_password = PasswordField(
        "Новый пароль",
        validators=[DataRequired(), Length(min=8, max=128)],
    )
    confirm_password = PasswordField(
        "Повторите новый пароль",
        validators=[
            DataRequired(),
            EqualTo('new_password', message='Новые пароли не совпадают.'),
        ],
    )
    submit = SubmitField("Изменить пароль")
 
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            account = get_acc_by_username(form.username.data)
            if account:
                password_rs = account['password']
                if check_password_hash(password_rs, form.password.data):
                    session['loggedin'] = True
                    session['id'] = account['id']
                    session['username'] = account['username']
                    if not app.debug:
                        user_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                        app.logger.info(f'{form.username.data} - login successful. IP: {user_ip}')
                    return redirect(url_for('landing'))
                else:
                    flash('Некорректный логин или пароль')
                    time.sleep(1.5)
            else:
                    flash('Некорректный логин или пароль')
                    time.sleep(1.5)

    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
def profile(): 
    if 'loggedin' in session:
        account = get_acc_by_id([session['id']])
        form = ChangePasswordForm()
        if form.validate_on_submit():
            if not check_password_hash(account['password'], form.current_password.data):
                flash('Текущий пароль указан неверно.')
            elif check_password_hash(account['password'], form.new_password.data):
                flash('Новый пароль должен отличаться от текущего.')
            elif update_user_password(session['username'], form.new_password.data):
                app.logger.info('%s - password changed', session['username'])
                flash('Пароль успешно изменен.')
                return redirect(url_for('profile'))
            else:
                app.logger.error('%s - password update affected no user', session['username'])
                flash('Не удалось изменить пароль. Повторите попытку.')

        cur_disk = check_disk_size(session['username'])
        user_limits = get_user_limits(account['tarif_name'])
        occ_space = int(round(cur_disk/user_limits['disk_limit']*100, 2))
        if occ_space > 100: occ_space = 100
        app.logger.info(f'{session["username"]} - occ_space: {occ_space}')
        return render_template('profile.html', username=session['username'], tarif_name=account['tarif_name'], title="Профиль", occ_space=occ_space, cur_disk=cur_disk, form=form)
    else:
        return redirect(url_for('login'))


@app.route('/cookie')
def cook_info():
    return render_template('cookie.html')
