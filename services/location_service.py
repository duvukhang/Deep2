# services/location_service.py

import requests


class LocationService:
    def get_location_by_ip(self):
        try:
            response = requests.get("https://ipapi.co/json/", timeout=8)
            response.raise_for_status()

            data = response.json()

            lat = data.get("latitude")
            lon = data.get("longitude")

            if lat is None or lon is None:
                return None

            return {
                "lat": float(lat),
                "lon": float(lon),
                "city": data.get("city", "Không rõ"),
                "region": data.get("region", ""),
                "country": data.get("country_name", ""),
                "ip": data.get("ip", "")
            }

        except Exception as e:
            print("LocationService error:", e)
            return None