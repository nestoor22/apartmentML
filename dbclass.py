import sqlite3


_cached = {}


class ApartmentsDB:

    def __init__(self):
        self.Address         = None
        self.SquareMeterCost = None
        self.Cost            = None
        self.Area            = None
        self.Rooms           = None
        self.Area            = None
        self.LivingArea      = None
        self.KitchenArea     = None
        self.Floor           = None
        self.Floors          = None
        self.CeilingHeight   = None
        self.BuildingType    = None
        self.Condition       = None
        self.WallsMaterial   = None
        self.Balconies       = None
        self.material        = None

    def connect(self):
        if 'ApartmentsInfo.db' in _cached:
            return _cached['ApartmentsInfo.db']
        else:
            return sqlite3.connect('ApartmentsInfo.db')

    def get_field_type(self, v):
        if v is None:
            return None
        if isinstance(v, str):
            return "TEXT"
        if isinstance(v, list):
            return "TEXT"
        if isinstance(v, float):
            return "FLOAT"
        if isinstance(v, int):
            return "BIGINT"

    def create_table(self):
        db = self.connect()
        fields = []

        for f in self.__dict__.keys():
            if f == "id":
                continue

            v = getattr(self, f)
            if v is None:
                pass
            else:
                field_type = self.get_field_type(v)
                if field_type is None:
                    pass
                else:
                    fields.append(f + " " + field_type)

        #
        # fields.insert(0, "id BIGINT")

        #
        sql = "CREATE TABLE IF NOT EXISTS {} (".format('apartment_info')
        sql = sql + ",".join(fields)
        sql = sql + ')'
        db.cursor().execute(sql)
        db.commit()

    def check_table(self):
        """ Check table existens and structure. Create if need, update if need """
        db = self.connect()
        db.row_factory = None
        c = db.cursor()
        sql = """
            SELECT count(*) as cnt FROM sqlite_master WHERE type='table' AND name='{}';
        """.format('apartment_info')
        c.execute(sql)
        count = c.fetchone()[0]
        # db.close()

        #
        if count == 0:
            self.create_table()
        else:
            self.update_table()

    def update_table(self):
        """ Update table structure. Add absent columns. Python object attributes -> sqlite columns.  """
        # db_fields
        db = self.connect()
        c = db.cursor()
        c.execute("SELECT * FROM {} LIMIT 1".format('apartment_info'))
        db_fields = [description[0] for description in c.description]

        # object fields
        obj_fields = self.__dict__.keys()

        # get new fields
        new_fields = []
        for f in obj_fields:
            if f in db_fields:
                pass
            else:
                new_fields.append(f)

        #
        fields = []
        for f in new_fields:
            v = getattr(self, f)
            field_type = self.get_field_type(v)

            if v is None:
                pass
            elif field_type is None:
                pass
            else:
                fields.append(f + " " + field_type)

        #
        for field in fields:
            sql = "ALTER TABLE {} ADD {}".format('apartment_info', field)
            c = db.cursor()
            c.execute(sql)

    def add_info_to_class(self, apart_object):
        for key in apart_object.__dict__.keys():
            self.__dict__[key] = getattr(apart_object, key, None)

    def add_to_db(self, apart_object):
        db = self.connect()
        self.add_info_to_class(apart_object)
        self.check_table()

        fields = []
        placeholders = []
        values = []

        for field in self.__dict__.keys():
            value = getattr(self, field, None)
            if value:
                fields.append(field)
                values.append(value)
                placeholders.append("?")

                # insert
        c = db.cursor()
        sql = """
            INSERT INTO {}
                ({})
                VALUES
                ({})
            """.format(
            "apartment_info",
            ",".join(fields),
            ",".join(placeholders)
        )
        try:
            c.execute(sql, values)
        except sqlite3.OperationalError:
            print('-----------------------------------------')
        db.commit()

    def get_from_db(self, **condition):
        pass