from utils import db_connect, BACKOFFICE_DB_NAME


def recreate_db():
    db = db_connect(BACKOFFICE_DB_NAME)

    try:
        # db.execute("drop table if exists trainer_cluster")
        db.execute("drop table if exists sim_config")
        db.execute("drop table if exists policy")
        db.execute("drop table if exists policy_run")
        db.execute("drop table if exists baseline_run")
    except Exception as e:
        raise e

    # An entry for each trainer cluster. Stop will be filled with the remove operation.
    # db.execute('''create table trainer_cluster
    #               (id INTEGER PRIMARY KEY,
    #                name unicode,
    #                cloud_provider unicode,
    #                start TIMESTAMP,
    #                stop TIMESTAMP,
    #                config json
    #                )''')

    # An entry for each policy deployed. It may associated (or not) with an endpoint.
    db.execute('''create table policy
                   (trainer_id INTEGER,
                    policy_id INTEGER,
                    backend_name unicode,                 
                    data json,
                    PRIMARY KEY(trainer_id, policy_id),
                    FOREIGN KEY(trainer_id) REFERENCES trainer_cluster(id) ON DELETE CASCADE
                    )''')

if __name__ == "__main__":
    recreate_db()
