"""
world_cities.py — liste UNIQUE des villes du monde partagée par
ingestion.py, historique.py et dashboard.py.

Modèle de données (à respecter partout) :
  - Un CLIENT a son propre cc_num et SA ville de naissance/domicile, fixe
    pour toute sa vie (champs city / lat / long envoyés à l'API de scoring).
  - Le LIEU DE TRANSACTION (merch_lat / merch_long) est indépendant du
    client : il est tiré au hasard, à CHAQUE transaction, parmi TOUTES les
    villes du monde de cette liste — y compris la ville de naissance du
    client (une carte peut très bien acheter chez elle).
  - Le dashboard retrouve le NOM de la ville de transaction en associant
    (merch_lat, merch_long) à la ville la plus proche de cette liste
    (nearest_world_city) : la base ne stocke que des coordonnées.
"""

import math
import random

# (nom, latitude, longitude)
WORLD_CITIES = [
    ("Paris", 48.8566, 2.3522), ("Lyon", 45.7640, 4.8357),
    ("Marseille", 43.2965, 5.3698), ("Nantes", 47.2184, -1.5536),
    ("Toulouse", 43.6047, 1.4442), ("Kairouan", 35.6786, 10.0963),
    ("Tunis", 36.8065, 10.1815), ("Casablanca", 33.5731, -7.5898),
    ("Alger", 36.7538, 3.0588), ("Dakar", 14.7167, -17.4677),
    ("Abidjan", 5.3600, -4.0083), ("Le Caire", 30.0444, 31.2357),
    ("Lagos", 6.5244, 3.3792), ("Nairobi", -1.2921, 36.8219),
    ("Johannesburg", -26.2041, 28.0473), ("Le Cap", -33.9249, 18.4241),
    ("Londres", 51.5074, -0.1278), ("Manchester", 53.4808, -2.2426),
    ("Dublin", 53.3498, -6.2603), ("Berlin", 52.5200, 13.4050),
    ("Munich", 48.1351, 11.5820), ("Madrid", 40.4168, -3.7038),
    ("Barcelone", 41.3851, 2.1734), ("Lisbonne", 38.7223, -9.1393),
    ("Rome", 41.9028, 12.4964), ("Milan", 45.4642, 9.1900),
    ("Amsterdam", 52.3676, 4.9041), ("Bruxelles", 50.8503, 4.3517),
    ("Genève", 46.2044, 6.1432), ("Zurich", 47.3769, 8.5417),
    ("Vienne", 48.2082, 16.3738), ("Varsovie", 52.2297, 21.0122),
    ("Prague", 50.0755, 14.4378), ("Budapest", 47.4979, 19.0402),
    ("Athènes", 37.9838, 23.7275), ("Stockholm", 59.3293, 18.0686),
    ("Oslo", 59.9139, 10.7522), ("Copenhague", 55.6761, 12.5683),
    ("Helsinki", 60.1699, 24.9384), ("Moscou", 55.7558, 37.6173),
    ("Istanbul", 41.0082, 28.9784), ("Dubaï", 25.2048, 55.2708),
    ("Abu Dhabi", 24.4539, 54.3773), ("Doha", 25.2854, 51.5310),
    ("Riyad", 24.7136, 46.6753), ("Tel Aviv", 32.0853, 34.7818),
    ("New York", 40.7128, -74.0060), ("Los Angeles", 34.0522, -118.2437),
    ("Chicago", 41.8781, -87.6298), ("Miami", 25.7617, -80.1918),
    ("Houston", 29.7604, -95.3698), ("Toronto", 43.6532, -79.3832),
    ("Vancouver", 49.2827, -123.1207), ("Montréal", 45.5019, -73.5674),
    ("Mexico", 19.4326, -99.1332), ("São Paulo", -23.5505, -46.6333),
    ("Rio de Janeiro", -22.9068, -43.1729), ("Buenos Aires", -34.6037, -58.3816),
    ("Santiago", -33.4489, -70.6693), ("Bogota", 4.7110, -74.0721),
    ("Lima", -12.0464, -77.0428), ("Tokyo", 35.6762, 139.6503),
    ("Osaka", 34.6937, 135.5023), ("Séoul", 37.5665, 126.9780),
    ("Pékin", 39.9042, 116.4074), ("Shanghai", 31.2304, 121.4737),
    ("Hong Kong", 22.3193, 114.1694), ("Singapour", 1.3521, 103.8198),
    ("Bangkok", 13.7563, 100.5018), ("Jakarta", -6.2088, 106.8456),
    ("Kuala Lumpur", 3.1390, 101.6869), ("Manille", 14.5995, 120.9842),
    ("Mumbai", 19.0760, 72.8777), ("Delhi", 28.7041, 77.1025),
    ("Bangalore", 12.9716, 77.5946), ("Karachi", 24.8607, 67.0011),
    ("Sydney", -33.8688, 151.2093), ("Melbourne", -37.8136, 144.9631),
    ("Auckland", -36.8485, 174.7633),
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def with_extra_cities(extra):
    """WORLD_CITIES + villes supplémentaires (ex. domiciles de clients absents
    de la liste), sans doublon (même nom ou à moins de 5 km d'une ville connue)."""
    cities = list(WORLD_CITIES)
    for name, lat, lon in extra:
        if any(n == name or haversine_km(lat, lon, la, lo) < 5 for n, la, lo in cities):
            continue
        cities.append((name, lat, lon))
    return cities


def random_transaction_location(cities=None, jitter_deg=0.05):
    """Lieu de transaction aléatoire : (nom_ville, lat, long) tiré uniformément
    parmi toutes les villes, avec un léger bruit (~5 km) autour du centre-ville.
    Aucun lien avec le client qui paie."""
    name, lat, lon = random.choice(cities or WORLD_CITIES)
    lat = max(-90.0, min(90.0, lat + random.uniform(-jitter_deg, jitter_deg)))
    lon = max(-180.0, min(180.0, lon + random.uniform(-jitter_deg, jitter_deg)))
    return name, round(lat, 6), round(lon, 6)


def nearest_world_city(lat, lon, max_km=60.0, cities=None):
    """(nom, lat_centre, lon_centre, distance_km) de la ville la plus proche
    dans max_km, sinon None."""
    best = None
    for name, la, lo in (cities or WORLD_CITIES):
        d = haversine_km(lat, lon, la, lo)
        if best is None or d < best[3]:
            best = (name, la, lo, d)
    return best if best and best[3] <= max_km else None