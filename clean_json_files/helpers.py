import gmaps
import googlemaps
from settings import GOOGLE_MAPS_API_KEY

gmaps.configure(api_key=GOOGLE_MAPS_API_KEY)   # Your api here
api = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


def get_distance(x, y):
    # try:
    return float(api.directions(x, y)[0]['legs'][0]['distance']['text'].replace(' km', ''))
    # except Exception:
    #     return 0.0
