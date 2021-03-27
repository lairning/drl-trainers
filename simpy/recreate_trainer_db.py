from utils import db_connect, TRAINER_DB_NAME


def recreate_db():
    db = db_connect(TRAINER_DB_NAME)

    try:
        db.execute("drop table if exists sim_model")
        db.execute("drop table if exists sim_config")
        db.execute("drop table if exists training_session")
        db.execute("drop table if exists training_iteration")
        db.execute("drop table if exists policy")
        db.execute("drop table if exists policy_run")
        db.execute("drop table if exists baseline_run")


    except Exception as e:
        raise e

    # A SIM Model corresponds to a Python implementation of a simulation
    db.execute('''create table sim_model
                   (id INTEGER PRIMARY KEY,
                    name unicode,
                    other_data json
                    )''')

    # A Simulation implementation can be executed with different configurations: simulation duration,
    # interval between between each agent action, and other specific parameters (eX. mean time between arrivals)
    # This stable stores configurations that may be used in Training and for running a Simulation using a certain policy
    db.execute('''create table sim_config
                   (id INTEGER PRIMARY KEY,
                    sim_model_id integer,
                    name unicode,
                    baseline_avg float,
                    config json,
                    other_data json,
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id)
                    )''')

    db.execute('''create table training_session
                   (id INTEGER PRIMARY KEY,
                    sim_model_id integer,
                    sim_config_id integer,
                    time_start TIMESTAMP,
                    duration float,
                    best_policy integer,
                    agent_config json,
                    other_data json,
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id),
                    FOREIGN KEY(sim_config_id) REFERENCES sim_config(id)
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
                    PRIMARY KEY(training_session_id, id),
                    FOREIGN KEY(training_session_id) REFERENCES training_session(id)
                    )''')
    # AI Policies
    db.execute('''create table policy
                   (id INTEGER PRIMARY KEY,
                    sim_model_id integer,
                    sim_config_id integer,
                    session_id integer,        
                    iteration_id integer,       
                    checkpoint unicode,        
                    agent_config json,          
                    other_data json,
                    FOREIGN KEY(sim_model_id) REFERENCES sim_model(id),
                    FOREIGN KEY(sim_config_id) REFERENCES sim_config(id)
                    )''')

    db.execute('''create table policy_run
                   (id INTEGER PRIMARY KEY,
                    policy_id integer,
                    time_start TIMESTAMP,
                    simulations integer,
                    duration float,
                    results json,
                    other_data json,
                    FOREIGN KEY(policy_id) REFERENCES policy(id) ON DELETE CASCADE
                    )''')

    # Baseline Runs for Sim Configs
    db.execute('''create table baseline_run
                   (id INTEGER PRIMARY KEY,
                    sim_config_id integer,
                    time_start TIMESTAMP,
                    simulations integer,
                    duration float,
                    results json,
                    other_data json,
                    FOREIGN KEY(sim_config_id) REFERENCES sim_config(id)
                    )''')


if __name__ == "__main__":
    recreate_db()
