
def start_scrapping():
    from scrap_data import run_spiders as collect_data
    collect_data()


def insert_data_to_json():
    from clean_json_files.refactor_json_files import create_json_for_db as data_to_json
    data_to_json()


def create_sql_database():
    from db_work.download_apartments_db import load_apartments_info_to_db as data_to_db
    data_to_db()


def work_with_data():
    from db_work.work_with_data import change_building_types, fill_absent_data_and_remove_incorrect
    fill_absent_data_and_remove_incorrect()
    change_building_types()


def save_trained_models():
    from ml_system import train_all_models as train
    train()


# start_scrapping()

# insert_data_to_json()

# create_sql_database()
#
# work_with_data()

save_trained_models()


