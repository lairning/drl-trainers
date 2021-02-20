from utils import db_connect, DB_NAME


def recreate_db():
    db = db_connect(DB_NAME)

    try:
        db.execute("drop table if exists sim_model")
        db.execute("drop table if exists training_session")
        db.execute("drop table if exists training_iteration")
        db.execute("drop table if exists policy")
        db.execute("drop table if exists policy_run")
        db.execute("drop table if exists policy_run_sim")

    except Exception as e:
        raise e

    db.execute('''create table sim_model
                   (id INTEGER PRIMARY KEY,
                    name unicode,
                    other_data json
                    )''')

    db.execute('''create table training_session
                   (id INTEGER PRIMARY KEY,
                    sim_model_id integer,
                    time_start TIMESTAMP,
                    duration float,
                    best_policy integer,
                    agent_config json,
                    sim_config json,
                    other_data json,
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id)
                    )''')

    db.execute('''create table training_iteration
                   (training_session_id integer,
                    id integer,
                    reward_mean float,
                    reward_min float,
                    reward_max float,
                    checkpoint unicode,
                    duration float,
                    time_start TIMESTAMP,
                    other_data json,
                    PRIMARY KEY(training_session_id, id)
                    FOREIGN KEY(training_session_id) REFERENCES training_session(id)
                    )''')

    db.execute('''create table policy
                   (id INTEGER PRIMARY KEY,
                    sim_model_id integer,
                    session_id integer,   -- Only used for AI policies
                    iteration_id integer,        -- Only used for AI policies
                    checkpoint unicode,         -- Only used for AI policies
                    agent_config json,
                    sim_config json,
                    other_data json,
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id)
                    )''')

    db.execute('''create table policy_run
                   (id INTEGER PRIMARY KEY,
                    policy_id integer,
                    time_start TIMESTAMP,
                    simulations integer,
                    duration float,
                    results json,
                    other_data json,
                    FOREIGN KEY(policy_id) REFERENCES policy(id)
                    )''')


if __name__ == "__main__":
    recreate_db()
