import json
from dbclass import ApartmentsDB
from apartment import Apartment


def load_apartments_info_to_db():
    ApartmentDataBase = ApartmentsDB()
    list_of_dict = json.load(open('info_to_db.json'))
    for apart_dict in list_of_dict:
        apart_obj = Apartment()
        for key in apart_dict:
            apart_obj.__dict__[key] = apart_dict[key]
        ApartmentDataBase.add_to_db(apart_obj)
    return 1


if __name__ == '__main__':
    load_apartments_info_to_db()