# services/osm_service.py

import math
import requests


class OSMService:
    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"

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

    def classify_poi(self, tags):
        mapping = {
            "cafe": "Quán cà phê",
            "restaurant": "Nhà hàng",
            "fast_food": "Quán ăn nhanh",
            "fuel": "Cây xăng",
            "parking": "Bãi đỗ xe",
            "toilets": "Nhà vệ sinh",
            "hotel": "Khách sạn",
            "motel": "Nhà nghỉ",
            "guest_house": "Nhà khách",
            "services": "Trạm dịch vụ",
            "rest_area": "Trạm nghỉ"
        }

        raw_type = (
            tags.get("amenity")
            or tags.get("tourism")
            or tags.get("highway")
            or "unknown"
        )

        return mapping.get(raw_type, raw_type)

    def find_nearest_rest_stop(self, lat, lon, radius=5000):
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"~"cafe|restaurant|fast_food|fuel|parking|toilets"](around:{radius},{lat},{lon});
          way["amenity"~"cafe|restaurant|fast_food|fuel|parking|toilets"](around:{radius},{lat},{lon});
          relation["amenity"~"cafe|restaurant|fast_food|fuel|parking|toilets"](around:{radius},{lat},{lon});

          node["tourism"~"hotel|motel|guest_house"](around:{radius},{lat},{lon});
          way["tourism"~"hotel|motel|guest_house"](around:{radius},{lat},{lon});
          relation["tourism"~"hotel|motel|guest_house"](around:{radius},{lat},{lon});

          node["highway"~"services|rest_area"](around:{radius},{lat},{lon});
          way["highway"~"services|rest_area"](around:{radius},{lat},{lon});
          relation["highway"~"services|rest_area"](around:{radius},{lat},{lon});
        );
        out center tags;
        """

        try:
            response = requests.get(
                self.overpass_url,
                params={"data": query},
                timeout=20
            )

            response.raise_for_status()
            data = response.json()

            stops = []

            for element in data.get("elements", []):
                tags = element.get("tags", {})

                poi_lat = element.get("lat")
                poi_lon = element.get("lon")

                if poi_lat is None or poi_lon is None:
                    center = element.get("center", {})
                    poi_lat = center.get("lat")
                    poi_lon = center.get("lon")

                if poi_lat is None or poi_lon is None:
                    continue

                distance_km = self.haversine(lat, lon, poi_lat, poi_lon)

                poi_type = self.classify_poi(tags)
                name = tags.get("name") or poi_type

                stops.append({
                    "name": name,
                    "type": poi_type,
                    "lat": poi_lat,
                    "lon": poi_lon,
                    "distance_km": round(distance_km, 2),
                    "distance_text": f"{round(distance_km, 2)} km",
                    "map_url": f"https://www.openstreetmap.org/?mlat={poi_lat}&mlon={poi_lon}#map=18/{poi_lat}/{poi_lon}",
                    "direction_url": f"https://www.openstreetmap.org/directions?from={lat}%2C{lon}&to={poi_lat}%2C{poi_lon}"
                })

            stops.sort(key=lambda item: item["distance_km"])

            unique = []
            seen = set()

            for stop in stops:
                key = (
                    stop["name"],
                    round(stop["lat"], 5),
                    round(stop["lon"], 5)
                )

                if key not in seen:
                    unique.append(stop)
                    seen.add(key)

                if len(unique) >= 8:
                    break

            return unique

        except Exception as e:
            print("OSMService error:", e)
            return []