import json
from apartment import Apartment
from dbclass import ApartmentsDB


def load_apartments_info_to_db():
    apartment_data_base = ApartmentsDB()
    list_of_dict = json.load(open('info_to_db.json'))
    for apart_dict in list_of_dict:
        apart_obj = Apartment()
        for key in apart_dict:
            apart_obj.__dict__[key] = apart_dict[key]
        apartment_data_base.add_to_db(apart_obj)
    return 1
