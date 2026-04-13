import requests

class OSMService:
    def find_nearest_rest_stop(self, lat, lon):
        url = f"https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node["amenity"="rest_area"](around:5000,{lat},{lon});
        out;
        """
        try:
            res = requests.get(url, params={'data': query}).json()
            stops = []
            for el in res['elements'][:5]:
                stops.append({
                    "name": el.get("tags", {}).get("name", "Rest Stop"),
                    "distance": "gần"
                })
            return stops
        except:
            return []