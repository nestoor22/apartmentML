
def start_scrapping():
    from scrap_data import run_spiders as collect_data
    collect_data()


def insert_data_to_json():
    from refactor_json_files import create_json_for_db as data_to_json
    data_to_json()


def create_sql_database():
    from download_apartments_db import load_apartments_info_to_db as data_to_db
    data_to_db()


def prepare_data():
    from work_with_data import fill_all_absent_data as create_better_db
    create_better_db()


def save_trained_model():
    from ml_system import run_training_and_save_apartment_price_model as create_model
    create_model()


start_scrapping()

insert_data_to_json()

create_sql_database()

prepare_data()

save_trained_model()


