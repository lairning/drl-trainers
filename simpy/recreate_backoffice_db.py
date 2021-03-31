from utils import db_connect, BACKOFFICE_DB_NAME


def recreate_db():
    db = db_connect(BACKOFFICE_DB_NAME)

    try:
        #db.execute("drop table if exists trainer_cluster")
        db.execute("drop table if exists sim_config")
        db.execute("drop table if exists policy")
        db.execute("drop table if exists policy_run")
        db.execute("drop table if exists baseline_run")
    except Exception as e:
        raise e

    # An entry for each trainer cluster. Stop will be filled with the remove operation.
    db.execute('''create table trainer_cluster
                   (id INTEGER PRIMARY KEY,
                    name unicode,
                    cloud_provider unicode,
                    start TIMESTAMP,
                    stop TIMESTAMP,
                    config json
                    )''')

    db.execute('''create table sim_config
                   (cluster_id INTEGER,
                    config_id INTEGER,
                    model_name unicode,
                    config_name unicode,
                    baseline_avg float,
                    config json,
                    PRIMARY KEY(cluster_id, config_id),
                    FOREIGN KEY(cluster_id) REFERENCES trainer_cluster(id) ON DELETE CASCADE
                    )''')

    # An entry for each policy created by a trainer cluster
    # one cluster may have more than one simulation model
    db.execute('''create table policy
                   (cluster_id INTEGER,
                    policy_id INTEGER,
                    sim_config_id INTEGER,
                    checkpoint unicode,        
                    agent_config json,
                    sim_config json,
                    backend_name unicode, -- name of the backend if created                 
                    data json,
                    PRIMARY KEY(cluster_id, policy_id),
                    FOREIGN KEY(cluster_id, sim_config_id) REFERENCES sim_config(cluster_id, sim_config_id) ON DELETE CASCADE
                    )''')

    db.execute('''create table policy_run
                   (cluster_id INTEGER,
                    run_id INTEGER,
                    policy_id integer,
                    time_start TIMESTAMP,
                    simulations integer,
                    duration float,
                    results json,
                    other_data json,
                    PRIMARY KEY(cluster_id, run_id),
                    FOREIGN KEY(cluster_id, policy_id) REFERENCES policy(cluster_id, policy_id) ON DELETE CASCADE
                    )''')

    # Baseline Runs for Sim Configs
    db.execute('''create table baseline_run
                   (cluster_id INTEGER,
                    run_id INTEGER,
                    sim_config_id INTEGER,
                    time_start TIMESTAMP,
                    simulations integer,
                    duration float,
                    results json,
                    other_data json,
                    PRIMARY KEY(cluster_id, run_id),
                    FOREIGN KEY(cluster_id, sim_config_id) REFERENCES sim_config(cluster_id, sim_config_id) ON DELETE CASCADE
                    )''')


if __name__ == "__main__":
    recreate_db()
