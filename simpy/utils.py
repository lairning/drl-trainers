
# import platform

DBTYPE = 'sqlite' # if platform.system() == 'Windows' else 'mysql'

P_MARKER = "?" if DBTYPE == 'sqlite' else "%s"

SQLParamList = lambda n: '' if n <= 0 else (P_MARKER+',')*(n-1) + P_MARKER


DB_NAME = "laisimpy"

def db_connect(db_name: str):
    if DBTYPE == 'sqlite':
        import sqlite3 as dbengine
        return dbengine.connect("{}.db".format(db_name))
    else:
        raise Exception("Invalid DB Type")

def select_record(dbcon, sql: str, params: tuple) -> tuple:
    cursor = dbcon.cursor()
    cursor.execute(sql, params)
    return cursor.fetchone()

def select_all(dbcon, sql: str, params: tuple) -> tuple:
    cursor = dbcon.cursor()
    cursor.execute(sql, params)
    return cursor.fetchall()
