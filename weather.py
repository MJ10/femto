"""
The weather tool's backends, both returning Open-Meteo's forecast JSON (https://open-meteo.com/en/docs):
- simulate: deterministic and offline, for training and evals. Any location string is a place; its climate (a
  latitude, hence a seasonal temperature curve) is hashed from the name, and each day's weather from (name, date).
- live: Open-Meteo's geocoding and forecast APIs (no key), for the CLI.
The tool shows summary(raw) as JSON: current conditions and a 3-day daily forecast, starting at the asked date (or today).
    python weather.py Montreal [2025-06-16] [--live]
"""

import datetime
import hashlib
import json
import math
import random
import re
import string
import urllib.parse
import urllib.request

TODAY = "2025-06-14"  # "now" of the simulator in training and evals (the CLI uses the real date)
DAYS = 3
CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
         51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
         61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
         71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light rain showers",
         81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
         95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}
CURRENT = ["temperature_2m", "relative_humidity_2m", "weather_code", "wind_speed_10m"]
DAILY = ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_probability_max"]
UNITS = {"temperature_2m": "°C", "relative_humidity_2m": "%", "weather_code": "wmo code", "wind_speed_10m": "km/h",
         "temperature_2m_max": "°C", "temperature_2m_min": "°C", "precipitation_sum": "mm",
         "precipitation_probability_max": "%"}
DATE = re.compile(r",\s*(\d{4}-\d{2}-\d{2})\s*$")
# cities the simulator knows the latitude / longitude of (others get hashed ones), e.g. for training and eval prompts
CITIES = {name: (country, float(lat), float(lon)) for name, country, lat, lon in (row.split(":") for row in re.split(r"(?<=\d)\s+", """\
Montreal:Canada:45.5:-73.57 Toronto:Canada:43.65:-79.38 Vancouver:Canada:49.28:-123.12 Calgary:Canada:51.05:-114.07
Ottawa:Canada:45.42:-75.7 Quebec City:Canada:46.81:-71.21 Halifax:Canada:44.65:-63.57 Winnipeg:Canada:49.9:-97.14
New York:United States:40.71:-74.01 Los Angeles:United States:34.05:-118.24 Chicago:United States:41.88:-87.63
Houston:United States:29.76:-95.37 Phoenix:United States:33.45:-112.07 Seattle:United States:47.61:-122.33
San Francisco:United States:37.77:-122.42 Boston:United States:42.36:-71.06 Miami:United States:25.76:-80.19
Denver:United States:39.74:-104.99 Atlanta:United States:33.75:-84.39 Washington:United States:38.91:-77.04
Anchorage:United States:61.22:-149.9 Honolulu:United States:21.31:-157.86 Mexico City:Mexico:19.43:-99.13
Guadalajara:Mexico:20.66:-103.35 Havana:Cuba:23.11:-82.37 Bogota:Colombia:4.71:-74.07 Lima:Peru:-12.05:-77.04
Quito:Ecuador:-0.18:-78.47 Caracas:Venezuela:10.49:-66.88 Santiago:Chile:-33.45:-70.67
Buenos Aires:Argentina:-34.6:-58.38 Montevideo:Uruguay:-34.9:-56.16 Sao Paulo:Brazil:-23.55:-46.63
Rio de Janeiro:Brazil:-22.91:-43.17 Brasilia:Brazil:-15.79:-47.88 London:United Kingdom:51.51:-0.13
Manchester:United Kingdom:53.48:-2.24 Edinburgh:United Kingdom:55.95:-3.19 Dublin:Ireland:53.35:-6.26
Paris:France:48.86:2.35 Lyon:France:45.76:4.84 Marseille:France:43.3:5.37 Madrid:Spain:40.42:-3.7
Barcelona:Spain:41.39:2.17 Seville:Spain:37.39:-5.98 Lisbon:Portugal:38.72:-9.14 Porto:Portugal:41.15:-8.61
Berlin:Germany:52.52:13.4 Munich:Germany:48.14:11.58 Hamburg:Germany:53.55:9.99 Frankfurt:Germany:50.11:8.68
Amsterdam:Netherlands:52.37:4.9 Brussels:Belgium:50.85:4.35 Zurich:Switzerland:47.38:8.54 Geneva:Switzerland:46.2:6.14
Vienna:Austria:48.21:16.37 Prague:Czech Republic:50.08:14.44 Warsaw:Poland:52.23:21.01 Krakow:Poland:50.06:19.94
Budapest:Hungary:47.5:19.04 Rome:Italy:41.9:12.5 Milan:Italy:45.46:9.19 Naples:Italy:40.85:14.27
Athens:Greece:37.98:23.73 Istanbul:Turkey:41.01:28.98 Ankara:Turkey:39.93:32.86 Copenhagen:Denmark:55.68:12.57
Stockholm:Sweden:59.33:18.07 Oslo:Norway:59.91:10.75 Helsinki:Finland:60.17:24.94 Reykjavik:Iceland:64.15:-21.94
Moscow:Russia:55.76:37.62 Saint Petersburg:Russia:59.93:30.34 Kyiv:Ukraine:50.45:30.52 Bucharest:Romania:44.43:26.1
Cairo:Egypt:30.04:31.24 Casablanca:Morocco:33.57:-7.59 Lagos:Nigeria:6.52:3.38 Accra:Ghana:5.6:-0.19
Nairobi:Kenya:-1.29:36.82 Addis Ababa:Ethiopia:9.03:38.74 Johannesburg:South Africa:-26.2:28.05
Cape Town:South Africa:-33.92:18.42 Dakar:Senegal:14.72:-17.47 Tunis:Tunisia:36.81:10.18 Dubai:United Arab Emirates:25.2:55.27
Riyadh:Saudi Arabia:24.71:46.68 Tehran:Iran:35.69:51.39 Tel Aviv:Israel:32.09:34.78 Karachi:Pakistan:24.86:67.0
Delhi:India:28.61:77.21 Mumbai:India:19.08:72.88 Bangalore:India:12.97:77.59 Kolkata:India:22.57:88.36
Chennai:India:13.08:80.27 Dhaka:Bangladesh:23.81:90.41 Kathmandu:Nepal:27.72:85.32 Colombo:Sri Lanka:6.93:79.86
Bangkok:Thailand:13.76:100.5 Hanoi:Vietnam:21.03:105.85 Ho Chi Minh City:Vietnam:10.82:106.63
Singapore:Singapore:1.35:103.82 Kuala Lumpur:Malaysia:3.14:101.69 Jakarta:Indonesia:-6.21:106.85
Manila:Philippines:14.6:120.98 Beijing:China:39.9:116.41 Shanghai:China:31.23:121.47 Hong Kong:China:22.32:114.17
Guangzhou:China:23.13:113.26 Taipei:Taiwan:25.03:121.57 Seoul:South Korea:37.57:126.98 Tokyo:Japan:35.68:139.69
Osaka:Japan:34.69:135.5 Sapporo:Japan:43.06:141.35 Ulaanbaatar:Mongolia:47.89:106.91 Sydney:Australia:-33.87:151.21
Melbourne:Australia:-37.81:144.96 Brisbane:Australia:-27.47:153.03 Perth:Australia:-31.95:115.86
Auckland:New Zealand:-36.85:174.76 Wellington:New Zealand:-41.29:174.78"""))}

KNOWN = {name.casefold(): (name, *v) for name, v in CITIES.items()}


def parse(arg):
    """The tool's argument "Location[, YYYY-MM-DD]" -> (location, date or None)."""
    m = DATE.search(arg)
    return (arg[:m.start()].strip(), m.group(1)) if m else (arg.strip(), None)


def rng(*key):
    return random.Random(hashlib.sha256("|".join(map(str, key)).encode()).digest())


def simulate(location, date=None, today=TODAY):
    """Open-Meteo-shaped forecast for DAYS days from date (default today), with the current conditions at today's
    hour of the location's name. Deterministic in (location.casefold(), date, today)."""
    name = location.strip().casefold()
    if not name:
        raise ValueError("no location given")
    place = rng("place", name)
    city = KNOWN.get(name.split(",")[0].strip())
    lat, lon = city[2:] if city else (round(place.uniform(-50, 65), 2), round(place.uniform(-180, 180), 2))
    mean, amp = 30 - 0.4 * abs(lat) + place.uniform(-3, 3), 0.25 * abs(lat) + 1  # annual mean / half range, °C
    wet = place.uniform(0.1, 0.6)  # chance of a rainy day

    def day(d):
        r = rng("day", name, d)
        season = math.cos(2 * math.pi * (d.timetuple().tm_yday - 200) / 365) * (1 if lat >= 0 else -1)
        t = mean + amp * season + r.gauss(0, 3)
        rainy = r.random() < wet
        precip = round(r.expovariate(1 / 6), 1) if rainy else 0.0
        if precip:
            code = (95 if r.random() < 0.15 else r.choice([61, 63, 80])) if t > 1 else r.choice([71, 73])
            code = 65 if precip > 15 and code in (61, 63) else code
        else:
            code = r.choice([0, 1, 2, 3, 3, 45] if t < 20 else [0, 0, 1, 1, 2, 3])
        prob = min(100, int(40 + 8 * precip)) if precip else r.choice([0, 5, 10, 15, 20])
        spread = r.uniform(4, 12) / (2 if rainy else 1)
        return dict(code=code, max=round(t + spread / 2, 1), min=round(t - spread / 2, 1), precip=precip, prob=prob,
                    wind=round(r.uniform(3, 30 if not rainy else 45), 1), humidity=r.randint(55, 95) if rainy else r.randint(25, 75))

    now = datetime.date.fromisoformat(str(today))
    start = datetime.date.fromisoformat(date) if date else now
    days = [day(start + datetime.timedelta(days=i)) for i in range(DAYS)]
    cur, hour = day(now), rng("hour", name, now).randint(8, 20)
    cur_t = round(cur["min"] + (cur["max"] - cur["min"]) * math.sin(math.pi * (hour - 5) / 15), 1)
    return {"latitude": lat, "longitude": lon, "timezone": "GMT", "location": f"{city[0]}, {city[1]}" if city else string.capwords(location.strip()),
            "current_units": {"time": "iso8601", **{k: UNITS[k] for k in CURRENT}},
            "current": {"time": f"{now}T{hour:02d}:00", "temperature_2m": cur_t, "relative_humidity_2m": cur["humidity"],
                        "weather_code": cur["code"], "wind_speed_10m": cur["wind"]},
            "daily_units": {"time": "iso8601", **{k: UNITS[k] for k in DAILY}},
            "daily": {"time": [str(start + datetime.timedelta(days=i)) for i in range(DAYS)],
                      "weather_code": [d["code"] for d in days], "temperature_2m_max": [d["max"] for d in days],
                      "temperature_2m_min": [d["min"] for d in days], "precipitation_sum": [d["precip"] for d in days],
                      "precipitation_probability_max": [d["prob"] for d in days]}}


def get(url, **params):
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "femto"})
    with urllib.request.urlopen(req, timeout=10) as f:
        return json.load(f)


def live(location, date=None, today=None):
    """Open-Meteo's forecast for the best geocoding match of location ("City" or "City, Region/Country")."""
    name, *rest = [s.strip() for s in location.split(",")]
    places = get("https://geocoding-api.open-meteo.com/v1/search", name=name, count=10).get("results", [])
    places = [p for p in places if all(r.casefold() in (p.get("country", "") + p.get("admin1", "") + p.get("country_code", "")).casefold()
                                       for r in rest)] or places
    if not places:
        raise ValueError(f"unknown location: {location}")
    p = places[0]
    start = datetime.date.fromisoformat(date) if date else datetime.date.today()
    raw = get("https://api.open-meteo.com/v1/forecast", latitude=p["latitude"], longitude=p["longitude"],
              timezone="auto", current=",".join(CURRENT), daily=",".join(DAILY), start_date=str(start),
              end_date=str(start + datetime.timedelta(days=DAYS - 1)))
    raw["location"] = ", ".join(dict.fromkeys(x for x in (p["name"], p.get("admin1"), p.get("country")) if x))
    return raw


def summary(raw):
    """The compact view of a forecast that the tool shows."""
    c, d = raw["current"], raw["daily"]
    return {"location": raw["location"], "units": {"temperature": "°C", "wind": "km/h", "precipitation": "mm"},
            "current": {"time": c["time"], "temperature": c["temperature_2m"], "conditions": CODES.get(c["weather_code"], "unknown"),
                        "humidity": c["relative_humidity_2m"], "wind": c["wind_speed_10m"]},
            "daily": [{"date": t, "conditions": CODES.get(code, "unknown"), "high": hi, "low": lo, "precipitation": p,
                       "precipitation_chance": prob}
                      for t, code, hi, lo, p, prob in zip(d["time"], d["weather_code"], d["temperature_2m_max"],
                                                          d["temperature_2m_min"], d["precipitation_sum"],
                                                          d["precipitation_probability_max"])]}


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--live"]
    loc, date = parse(", ".join(args))
    raw = (live if "--live" in sys.argv else simulate)(loc, date)
    print(json.dumps(summary(raw), ensure_ascii=False))
