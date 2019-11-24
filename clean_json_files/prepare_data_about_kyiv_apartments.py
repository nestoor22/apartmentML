import os
import re
import json
import mtranslate
from helpers import get_distance

information_about_apartments = json.load(open('../json_files/kyiv_info.json'))

