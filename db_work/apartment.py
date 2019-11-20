
class Apartment:
    def __init__(self):
        self.id = None
        self.address = None
        self.square_meter_cost = None
        self.cost = None
        self.area = None
        self.rooms = None
        self.living_area = None
        self.kitchen_area = None
        self.floor = None
        self.floors = None
        self.ceiling_height = None
        self.building_type = None
        self.conditions = None
        self.walls_material = None
        self.balconies = None

    def __str__(self):
        return self.address
