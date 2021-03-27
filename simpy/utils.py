# import platform
import pandas as pd

DBTYPE = 'sqlite'  # if platform.system() == 'Windows' else 'mysql'

P_MARKER = "?" if DBTYPE == 'sqlite' else "%s"

SQLParamList = lambda n: '' if n <= 0 else (P_MARKER + ',') * (n - 1) + P_MARKER

TRAINER_DB_NAME = "simpy_template\\laisim_trainer"
BACKOFFICE_DB_NAME = "laisim_backoffice"


def db_connect(db_name: str, check_same_thread=True):
    if DBTYPE == 'sqlite':
        import sqlite3 as dbengine
        return dbengine.connect("{}.db".format(db_name),check_same_thread=check_same_thread)
    else:
        raise Exception("Invalid DB Type")


def select_record(dbcon, sql: str, params: tuple) -> tuple:
    cursor = dbcon.cursor()
    cursor.execute(sql, params)
    return cursor.fetchone()


def select_all(dbcon, sql: str, params: tuple = None ) -> list:
    cursor = dbcon.cursor()
    if params is None:
        cursor.execute(sql)
    else:
        cursor.execute(sql, params)
    return cursor.fetchall()


def table_fetch_all(dbname: str, table: str):
    db = db_connect(dbname)
    sql = '''SELECT * FROM {}'''.format(table)
    return pd.read_sql_query(sql, db)
