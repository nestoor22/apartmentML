
class Apartment:
    def __init__(self):
        self.id = None
        self.address = None
        self.cost = None
        self.area = None
        self.rooms = None
        self.living_area = None
        self.kitchen_area = None
        self.floor = None
        self.floors = None
        self.building_type = None
        self.conditions = None
        self.walls_material = None
        self.balconies = None
        self.city = None

    def __str__(self):
        return self.address
