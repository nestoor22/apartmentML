
def start_scrapping():
    from scrap_data import run_spiders as collect_data
    collect_data()


def save_data_to_db():
    from clean_json_files.prepare_data_about_lviv_apartments import create_json_for_db as save_data
    save_data()


def work_with_data():
    from db_work.work_with_data import change_building_types, fill_absent_data_and_remove_incorrect
    from data_analysis.plots import plot_and_save_all
    fill_absent_data_and_remove_incorrect()
    change_building_types()
    plot_and_save_all()


def save_trained_models():
    from ml_system import train_all_models as train
    train()


# start_scrapping()

# insert_data_to_json()

# create_sql_database()
#
# work_with_data()

save_trained_models()


