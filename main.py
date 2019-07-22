

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


create_sql_database()

prepare_data()