from utils import db_connect, BACKOFFICE_DB_NAME


def recreate_db():
    db = db_connect(BACKOFFICE_DB_NAME)

    try:
        db.execute("drop table if exists trainer")


    except Exception as e:
        raise e

    # A SIM Model corresponds to a Python implementation of a simulation
    db.execute('''create table trainer
                   (id INTEGER PRIMARY KEY,
                    name unicode,
                    data json
                    )''')


if __name__ == "__main__":
    recreate_db()
