# services/osm_service.py

import math
import time

import requests


class OSMService:
    def __init__(self):
        self.overpass_urls = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter"
        ]
        self.overpass_url = self.overpass_urls[0]
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
        self.good_stop_score = 65
        self.cache_ttl_seconds = 600
        self.cache = {}
        self.headers = {
            "User-Agent": "DriverDrowsinessRestStopFinder/1.0"
        }

        self.type_labels = {
            "services": "Trạm dịch vụ",
            "rest_area": "Trạm nghỉ",
            "fuel": "Cây xăng",
            "restaurant": "Nhà hàng",
            "cafe": "Quán cà phê",
            "fast_food": "Quán ăn nhanh",
            "food_court": "Khu ăn uống",
            "motel": "Nhà nghỉ",
            "hotel": "Khách sạn",
            "guest_house": "Nhà khách",
            "hostel": "Nhà nghỉ giá rẻ",
            "convenience": "Cửa hàng tiện lợi",
            "supermarket": "Siêu thị",
            "bakery": "Tiệm bánh",
            "mall": "Trung tâm thương mại",
            "parking": "Bãi đỗ xe",
            "toilets": "Nhà vệ sinh",
            "bus_station": "Bến xe",
            "charging_station": "Trạm sạc",
            "marketplace": "Chợ",
            "drinking_water": "Nước uống",
            "shelter": "Chỗ trú",
            "bench": "Ghế nghỉ",
            "park": "Công viên",
            "picnic_site": "Điểm picnic",
            "information": "Điểm thông tin"
        }

        self.type_scores = {
            "services": 100,
            "rest_area": 96,
            "fuel": 90,
            "motel": 82,
            "hotel": 80,
            "guest_house": 78,
            "hostel": 76,
            "restaurant": 74,
            "cafe": 72,
            "food_court": 70,
            "fast_food": 66,
            "convenience": 62,
            "supermarket": 58,
            "bakery": 54,
            "mall": 54,
            "bus_station": 52,
            "charging_station": 50,
            "marketplace": 48,
            "park": 46,
            "picnic_site": 46,
            "shelter": 42,
            "parking": 40,
            "toilets": 36,
            "drinking_water": 30,
            "bench": 28,
            "information": 25
        }

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def bearing_degrees(self, lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)

        y = math.sin(dlon) * math.cos(lat2_rad)
        x = (
            math.cos(lat1_rad) * math.sin(lat2_rad)
            - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        )

        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def angle_difference(self, angle_a, angle_b):
        return abs((angle_a - angle_b + 180) % 360 - 180)

    def cache_key(self, lat, lon, radius, heading):
        heading_bucket = None

        if heading is not None:
            heading_bucket = int(round(float(heading) / 30.0) * 30) % 360

        return (
            round(float(lat), 3),
            round(float(lon), 3),
            int(radius),
            heading_bucket
        )

    def get_cached(self, key):
        item = self.cache.get(key)

        if not item:
            return None

        created_at, value = item

        if time.time() - created_at > self.cache_ttl_seconds:
            self.cache.pop(key, None)
            return None

        return value

    def set_cached(self, key, value):
        self.cache[key] = (time.time(), value)

        if len(self.cache) > 100:
            oldest_key = min(self.cache, key=lambda item: self.cache[item][0])
            self.cache.pop(oldest_key, None)

    def get_poi_raw_type(self, tags):
        return (
            tags.get("highway")
            or tags.get("amenity")
            or tags.get("tourism")
            or tags.get("shop")
            or tags.get("leisure")
            or "unknown"
        )

    def classify_poi(self, tags):
        raw_type = self.get_poi_raw_type(tags)
        return self.type_labels.get(raw_type, raw_type)

    def is_public_poi(self, tags):
        access = (tags.get("access") or "").lower()
        service = (tags.get("service") or "").lower()
        parking = (tags.get("parking") or "").lower()

        blocked_values = {"private", "no", "permit", "delivery"}

        if access in blocked_values:
            return False

        if service in {"private", "driveway"}:
            return False

        if parking == "private":
            return False

        return True

    def calculate_rest_score(self, raw_type, tags, distance_km):
        score = self.type_scores.get(raw_type, 20)

        if tags.get("name"):
            score += 8

        if tags.get("opening_hours"):
            score += 4

        if tags.get("toilets") == "yes":
            score += 5

        if tags.get("parking") == "yes":
            score += 5

        if tags.get("drinking_water") == "yes":
            score += 3

        distance_penalty = min(distance_km * 1.2, 35)
        return max(score - distance_penalty, 0)

    def build_overpass_query(self, lat, lon, radius):
        return f"""
        [out:json][timeout:25];
        (
          nwr["highway"~"services|rest_area|bus_stop"](around:{radius},{lat},{lon});
          nwr["amenity"~"fuel|restaurant|cafe|fast_food|food_court|parking|toilets|bus_station|charging_station|marketplace|drinking_water|bench|shelter|pub|bar"](around:{radius},{lat},{lon});
          nwr["tourism"~"hotel|motel|guest_house|hostel|picnic_site|information"](around:{radius},{lat},{lon});
          nwr["shop"~"convenience|supermarket|bakery|mall|department_store"](around:{radius},{lat},{lon});
          nwr["leisure"~"park|garden|picnic_table"](around:{radius},{lat},{lon});
        );
        out center tags;
        """

    def request_overpass(self, query):
        last_error = None

        for url in self.overpass_urls:
            try:
                response = requests.get(
                    url,
                    params={"data": query},
                    headers=self.headers,
                    timeout=20
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                continue

        raise last_error

    def element_coordinates(self, element):
        poi_lat = element.get("lat")
        poi_lon = element.get("lon")

        if poi_lat is None or poi_lon is None:
            center = element.get("center", {})
            poi_lat = center.get("lat")
            poi_lon = center.get("lon")

        return poi_lat, poi_lon

    def apply_route_context(self, stop, lat, lon, heading=None):
        if heading is None:
            stop["route_score"] = 0.0
            stop["route_text"] = None
            return stop

        try:
            heading = float(heading)
        except Exception:
            stop["route_score"] = 0.0
            stop["route_text"] = None
            return stop

        bearing = self.bearing_degrees(lat, lon, stop["lat"], stop["lon"])
        diff = self.angle_difference(heading, bearing)

        if diff <= 45:
            route_score = 14.0
            route_text = "Nằm phía trước hướng di chuyển"
        elif diff <= 90:
            route_score = 6.0
            route_text = "Gần hướng di chuyển"
        elif diff >= 140:
            route_score = -10.0
            route_text = "Lệch/ngược hướng di chuyển"
        else:
            route_score = -2.0
            route_text = "Lệch nhẹ khỏi hướng di chuyển"

        stop["bearing"] = round(bearing, 1)
        stop["heading_diff"] = round(diff, 1)
        stop["route_score"] = route_score
        stop["route_text"] = route_text
        stop["rest_score"] = round(max(stop["rest_score"] + route_score, 0), 1)
        return stop

    def normalize_stop(self, element, lat, lon, heading=None):
        tags = element.get("tags", {})

        if not self.is_public_poi(tags):
            return None

        poi_lat, poi_lon = self.element_coordinates(element)

        if poi_lat is None or poi_lon is None:
            return None

        distance_km = self.haversine(lat, lon, poi_lat, poi_lon)
        raw_type = self.get_poi_raw_type(tags)
        poi_type = self.classify_poi(tags)
        name = tags.get("name") or poi_type
        rest_score = self.calculate_rest_score(raw_type, tags, distance_km)

        stop = {
            "name": name,
            "type": poi_type,
            "lat": poi_lat,
            "lon": poi_lon,
            "rest_score": round(rest_score, 1),
            "distance_km": round(distance_km, 2),
            "distance_text": f"{round(distance_km, 2)} km",
            "map_url": (
                "https://www.openstreetmap.org/"
                f"?mlat={poi_lat}&mlon={poi_lon}#map=18/{poi_lat}/{poi_lon}"
            ),
            "direction_url": (
                "https://www.openstreetmap.org/directions"
                f"?from={lat}%2C{lon}&to={poi_lat}%2C{poi_lon}"
            )
        }

        return self.apply_route_context(stop, lat, lon, heading)

    def normalize_nominatim_stop(self, item, lat, lon, query_type, heading=None):
        try:
            poi_lat = float(item["lat"])
            poi_lon = float(item["lon"])
        except Exception:
            return None

        distance_km = self.haversine(lat, lon, poi_lat, poi_lon)
        raw_type = query_type
        poi_type = self.type_labels.get(raw_type, raw_type)
        name = item.get("name") or item.get("display_name", "").split(",")[0] or poi_type
        rest_score = self.calculate_rest_score(raw_type, {"name": name}, distance_km)

        stop = {
            "name": name,
            "type": poi_type,
            "lat": poi_lat,
            "lon": poi_lon,
            "rest_score": round(rest_score, 1),
            "distance_km": round(distance_km, 2),
            "distance_text": f"{round(distance_km, 2)} km",
            "map_url": (
                "https://www.openstreetmap.org/"
                f"?mlat={poi_lat}&mlon={poi_lon}#map=18/{poi_lat}/{poi_lon}"
            ),
            "direction_url": (
                "https://www.openstreetmap.org/directions"
                f"?from={lat}%2C{lon}&to={poi_lat}%2C{poi_lon}"
            )
        }

        return self.apply_route_context(stop, lat, lon, heading)

    def find_nominatim_fallback(self, lat, lon, radius, heading=None):
        delta = max(radius / 111000.0, 0.01)
        viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
        queries = [
            ("cafe", "cafe"),
            ("restaurant", "restaurant"),
            ("fuel", "fuel"),
            ("rest area", "rest_area"),
            ("convenience store", "convenience"),
            ("hotel", "hotel"),
            ("park", "park")
        ]

        stops = []

        for text, raw_type in queries:
            try:
                response = requests.get(
                    self.nominatim_url,
                    params={
                        "q": text,
                        "format": "jsonv2",
                        "limit": 8,
                        "bounded": 1,
                        "viewbox": viewbox
                    },
                    headers=self.headers,
                    timeout=12
                )
                response.raise_for_status()

                for item in response.json():
                    stop = self.normalize_nominatim_stop(
                        item,
                        lat,
                        lon,
                        raw_type,
                        heading
                    )
                    if stop is not None and stop["distance_km"] <= radius / 1000:
                        stops.append(stop)
            except Exception:
                continue

        return self.unique_and_sort(stops)

    def unique_and_sort(self, stops, limit=8):
        stops.sort(
            key=lambda item: (
                -item["rest_score"],
                item["distance_km"]
            )
        )

        unique = []
        seen = set()

        for stop in stops:
            key = (
                stop["name"].lower(),
                round(stop["lat"], 5),
                round(stop["lon"], 5)
            )

            if key not in seen:
                unique.append(stop)
                seen.add(key)

            if len(unique) >= limit:
                break

        return unique

    def find_nearest_rest_stop(self, lat, lon, radius=5000, heading=None):
        key = self.cache_key(lat, lon, radius, heading)
        cached = self.get_cached(key)

        if cached is not None:
            return cached

        query = self.build_overpass_query(lat, lon, radius)

        try:
            data = self.request_overpass(query)
            stops = []

            for element in data.get("elements", []):
                stop = self.normalize_stop(element, lat, lon, heading)

                if stop is not None:
                    stops.append(stop)

            stops = self.unique_and_sort(stops)

            if stops:
                self.set_cached(key, stops)
                return stops

            fallback_stops = self.find_nominatim_fallback(lat, lon, radius, heading)
            self.set_cached(key, fallback_stops)
            return fallback_stops

        except Exception as e:
            print("OSMService error:", e)
            fallback_stops = self.find_nominatim_fallback(lat, lon, radius, heading)
            self.set_cached(key, fallback_stops)
            return fallback_stops

    def find_rest_stop_auto_radius(self, lat, lon, max_radius=40000, heading=None):
        radius_steps = [5000, 10000, 20000, 40000, 60000]
        best_result = {
            "radius_used": max_radius,
            "stops": []
        }

        for radius in radius_steps:
            if radius > max_radius:
                continue

            stops = self.find_nearest_rest_stop(lat, lon, radius, heading)

            if stops:
                best_result = {
                    "radius_used": radius,
                    "stops": stops
                }

                if stops[0].get("rest_score", 0) >= self.good_stop_score:
                    return best_result

        return best_result
