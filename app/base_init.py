from models import DB_PATH, init_db


if __name__ == '__main__':
    init_db()
    print(f'SQLite database initialized at: {DB_PATH}')
