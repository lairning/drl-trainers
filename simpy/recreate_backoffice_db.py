from utils import db_connect, BACKOFFICE_DB_NAME


def recreate_db():
    db = db_connect(BACKOFFICE_DB_NAME)

    try:
        db.execute("drop table if exists trainer_cluster")
        db.execute("drop table if exists policy")


    except Exception as e:
        raise e

    # An entry for each trainer cluster that is launched
    db.execute('''create table trainer_cluster
                   (id INTEGER PRIMARY KEY,
                    name unicode,
                    data json
                    )''')

    # An entry for each policy created by a trainer cluster
    # one cluster may have more than one simulation model
    db.execute('''create table policy
                   (cluster_id INTEGER,
                    policy_id INTEGER,
                    model_name unicode,
                    checkpoint unicode,        
                    agent_config json,
                    sim_config json,
                    backend integer default 0, -- Does the policy has a backend                 
                    data json,
                    PRIMARY KEY(cluster_id, policy_id),
                    FOREIGN KEY(cluster_id) REFERENCES trainer_cluster(id)
                    )''')



if __name__ == "__main__":
    recreate_db()
