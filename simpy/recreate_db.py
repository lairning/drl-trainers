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
                    config json,
                    other_data json
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id)
                    )''')

    db.execute('''create table training_iteration
                   (id INTEGER PRIMARY KEY,
                    training_session_id integer,
                    reward_mean float,
                    reward_min float,
                    reward_max float,
                    checkpoint unicode,
                    duration float,
                    time_start TIMESTAMP,
                    other_data json
                    FOREIGN KEY(training_session_id) REFERENCES training_session(id)
                    )''')

    db.execute('''create table policy
                   (id INTEGER PRIMARY KEY,
                    policy_type integer,        -- Defines if the policy is AI (1) or Baseline (0)
                    sim_model_id integer,
                    training_iteration integer, -- Only used for AI policies
                    checkpoint unicode,         -- Only used for AI policies
                    other_data json
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id)
                    )''')

    db.execute('''create table policy_run
                   (id INTEGER PRIMARY KEY,
                    policy_id integer,
                    time_start TIMESTAMP,
                    duration float,
                    other_data json
                    FOREIGN KEY(policy_id) REFERENCES policy(id)
                    )''')

    db.execute('''create table policy_run_sim
                   (id INTEGER PRIMARY KEY,
                    policy_run_id integer,
                    reward float,
                    duration float,
                    time_start TIMESTAMP,
                    other_data json
                    FOREIGN KEY(policy_run_id) REFERENCES policy_run(id)
                    )''')


if __name__ == "__main__":
    recreate_db()
