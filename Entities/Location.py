from math import radians, cos, sin, sqrt, atan2

class Location:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        # Rayon de la Terre en kilomètres
        R = 6371.0

        # Conversion des coordonnées de degrés en radians
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)

        # Différences de coordonnées
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Formule de Haversine
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        return distance

    def distance_to(self, other_location):
        return self.haversine(self.latitude, self.longitude, other_location.latitude, other_location.longitude)