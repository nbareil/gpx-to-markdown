#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click
import gpxpy
import requests

M_PER_DEG = 111_320.0
UPHILL_SPEED_KMH = 8.0


@dataclass(frozen=True)
class TrackPoint:
    lat: float
    lon: float
    ele: Optional[float]
    time: Optional[dt.datetime]


@dataclass(frozen=True)
class StepName:
    start_m: float
    end_m: float
    name: str


@dataclass(frozen=True)
class Event:
    distance_m: float
    description: str
    range_start_m: Optional[float] = None
    range_end_m: Optional[float] = None
    time_start: Optional[dt.datetime] = None
    time_end: Optional[dt.datetime] = None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def cumulative_distances(points: Iterable[Tuple[float, float]]) -> List[float]:
    distances = [0.0]
    prev = None
    for lat, lon in points:
        if prev is not None:
            distances.append(distances[-1] + haversine_m(prev[0], prev[1], lat, lon))
        prev = (lat, lon)
    return distances


def load_gpx_points(path: Path) -> List[TrackPoint]:
    with path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)
    if not gpx.tracks:
        raise click.ClickException("No tracks found in GPX.")
    points: List[TrackPoint] = []
    for segment in gpx.tracks[0].segments:
        for point in segment.points:
            points.append(
                TrackPoint(
                    lat=point.latitude,
                    lon=point.longitude,
                    ele=point.elevation,
                    time=point.time,
                )
            )
    if not points:
        raise click.ClickException("First track contains no points.")
    return points


def sample_points(points: List[TrackPoint], max_points: int) -> List[TrackPoint]:
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def request_json(
    url: str,
    params: Dict[str, str],
    timeout: int = 30,
    debug: bool = False,
    label: str = "",
    headers: Optional[Dict[str, str]] = None,
    method: str = "get",
) -> Optional[Dict]:
    try:
        if method == "post":
            response = requests.post(url, data=params, timeout=timeout, headers=headers)
        else:
            response = requests.get(url, params=params, timeout=timeout, headers=headers)
    except requests.RequestException as exc:
        if debug:
            prefix = f"{label} " if label else ""
            click.echo(f"{prefix}request failed: {exc}", err=True)
        return None
    if response.status_code != 200:
        if debug:
            prefix = f"{label} " if label else ""
            detail = response.text.strip().replace("\n", " ")
            limit = 1200 if label.lower() == "overpass" else 300
            if len(detail) > limit:
                detail = f"{detail[:limit]}..."
            if detail:
                click.echo(
                    f"{prefix}HTTP {response.status_code} for {response.url}: {detail}", err=True
                )
            else:
                click.echo(f"{prefix}HTTP {response.status_code} for {response.url}", err=True)
        return None
    try:
        return response.json()
    except ValueError:
        if debug:
            prefix = f"{label} " if label else ""
            detail = response.text.strip().replace("\n", " ")
            limit = 1200 if label.lower() == "overpass" else 300
            if len(detail) > limit:
                detail = f"{detail[:limit]}..."
            click.echo(f"{prefix}invalid JSON from {response.url}: {detail}", err=True)
        return None


def osrm_match(
    points: List[TrackPoint],
    osrm_url: str,
    profile: str,
    max_points: int,
    debug: bool,
) -> Optional[Dict]:
    candidates = []
    if len(points) <= max_points:
        candidates.append(points)
    else:
        candidates.append(sample_points(points, max_points))
    for fallback in (500, 200, 100):
        if len(points) > fallback:
            candidates.append(sample_points(points, fallback))
    seen = set()
    for candidate in candidates:
        if len(candidate) < 2:
            continue
        if len(candidate) in seen:
            continue
        seen.add(len(candidate))
        coord_str = ";".join(f"{p.lon},{p.lat}" for p in candidate)
        url = f"{osrm_url}/match/v1/{profile}/{coord_str}"
        params = {
            "geometries": "geojson",
            "overview": "full",
            "steps": "true",
            "annotations": "false",
        }
        data = request_json(url, params=params, debug=debug, label="OSRM")
        if data and data.get("matchings"):
            return data
    return None


def parse_match_geometry(match_data: Dict) -> Optional[List[Tuple[float, float]]]:
    matchings = match_data.get("matchings", [])
    if not matchings:
        return None
    geometry = matchings[0].get("geometry")
    if not geometry or geometry.get("type") != "LineString":
        return None
    coords = []
    for lon, lat in geometry.get("coordinates", []):
        coords.append((lat, lon))
    return coords if len(coords) > 1 else None


def parse_step_names(match_data: Dict) -> List[StepName]:
    matchings = match_data.get("matchings", [])
    if not matchings:
        return []
    steps: List[StepName] = []
    cursor = 0.0
    for leg in matchings[0].get("legs", []):
        for step in leg.get("steps", []):
            name = (step.get("name") or "").strip()
            geometry = step.get("geometry") or {}
            coords = geometry.get("coordinates") or []
            if len(coords) < 2:
                continue
            step_points = [(lat, lon) for lon, lat in coords]
            length = cumulative_distances(step_points)[-1]
            if length > 0:
                steps.append(StepName(cursor, cursor + length, name))
                cursor += length
    return steps


def name_for_distance(steps: List[StepName], distance_m: float) -> str:
    for step in steps:
        if step.start_m <= distance_m <= step.end_m and step.name:
            return step.name
    return ""


def detect_turns(
    polyline: List[Tuple[float, float]],
    distances: List[float],
    turn_angle: float,
    steps: List[StepName],
) -> List[Event]:
    events: List[Event] = []
    for i in range(1, len(polyline) - 1):
        lat_prev, lon_prev = polyline[i - 1]
        lat_curr, lon_curr = polyline[i]
        lat_next, lon_next = polyline[i + 1]
        b1 = bearing_deg(lat_prev, lon_prev, lat_curr, lon_curr)
        b2 = bearing_deg(lat_curr, lon_curr, lat_next, lon_next)
        delta = (b2 - b1 + 540.0) % 360.0 - 180.0
        if abs(delta) < turn_angle:
            continue
        direction = "droite" if delta > 0 else "gauche"
        angle = abs(delta)
        road_name = name_for_distance(steps, distances[i])
        if road_name:
            detail = f'Tourner à {direction} sur "{road_name}" (≈{angle:.0f}°)'
        else:
            detail = f"Tourner à {direction} (≈{angle:.0f}°)"
        events.append(Event(distance_m=distances[i], description=detail))
    return events


def detect_climbs(
    points: List[TrackPoint],
    distances: List[float],
    min_grade: float,
    min_minutes: float,
    elevation_smooth: int,
    steps: List[StepName],
    distance_scale: float,
) -> List[Event]:
    events: List[Event] = []
    smooth_window = max(1, elevation_smooth)
    half_window = smooth_window // 2
    elevations: List[Optional[float]] = [p.ele for p in points]
    if smooth_window > 1:
        smoothed: List[Optional[float]] = []
        for i in range(len(elevations)):
            start = max(0, i - half_window)
            end = min(len(elevations), i + half_window + 1)
            window_vals = [v for v in elevations[start:end] if v is not None]
            if window_vals:
                smoothed.append(sum(window_vals) / len(window_vals))
            else:
                smoothed.append(None)
        elevations = smoothed
    in_climb = False
    start_idx = 0
    total_dist = 0.0
    total_gain = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        e1 = elevations[i]
        e2 = elevations[i + 1]
        seg_dist = haversine_m(p1.lat, p1.lon, p2.lat, p2.lon)
        if seg_dist <= 0 or e1 is None or e2 is None:
            grade = 0.0
        else:
            elev_diff = e2 - e1
            grade = (elev_diff / seg_dist) * 100.0
        uphill = seg_dist > 0 and e1 is not None and e2 is not None and (e2 - e1) > 0
        if uphill and grade >= min_grade:
            if not in_climb:
                in_climb = True
                start_idx = i
                total_dist = 0.0
                total_gain = 0.0
            total_dist += seg_dist
            total_gain += max(0.0, (e2 or 0.0) - (e1 or 0.0))
        else:
            if in_climb:
                events.extend(
                    build_climb_event(
                        points,
                        distances,
                        start_idx,
                        i,
                        total_dist,
                        total_gain,
                        min_minutes,
                        steps,
                        distance_scale,
                    )
                )
                in_climb = False
    if in_climb:
        events.extend(
            build_climb_event(
                points,
                distances,
                start_idx,
                len(points) - 1,
                total_dist,
                total_gain,
                min_minutes,
                steps,
                distance_scale,
            )
        )
    return events


def build_climb_event(
    points: List[TrackPoint],
    distances: List[float],
    start_idx: int,
    end_idx: int,
    total_dist: float,
    total_gain: float,
    min_minutes: float,
    steps: List[StepName],
    distance_scale: float,
) -> List[Event]:
    if total_dist <= 0:
        return []
    est_minutes = (total_dist / 1000.0) / UPHILL_SPEED_KMH * 60.0
    if est_minutes < min_minutes:
        return []
    start_dist = distances[start_idx]
    end_dist = distances[end_idx]
    mid_dist = (start_dist + end_dist) / 2.0
    road_name = name_for_distance(steps, mid_dist * distance_scale)
    name_text = f' sur "{road_name}"' if road_name else ""
    detail = f"Montée {total_gain:.0f} m sur {total_dist/1000.0:.1f} km (≈{est_minutes:.0f} min){name_text}"
    return [
        Event(
            distance_m=start_dist,
            description=detail,
            range_start_m=start_dist,
            range_end_m=end_dist,
            time_start=points[start_idx].time,
            time_end=points[end_idx].time,
        )
    ]


def detect_pauses(points: List[TrackPoint], distances: List[float], min_minutes: float) -> List[Event]:
    events: List[Event] = []
    if not points or any(p.time is None for p in points):
        return events
    pause_start = None
    pause_start_dist = 0.0
    pause_start_time: Optional[dt.datetime] = None
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        if p1.time is None or p2.time is None:
            continue
        delta_s = (p2.time - p1.time).total_seconds()
        if delta_s <= 0:
            continue
        seg_dist = haversine_m(p1.lat, p1.lon, p2.lat, p2.lon)
        speed = seg_dist / delta_s
        if speed <= 0.1:
            if pause_start is None:
                pause_start = i
                pause_start_dist = distances[i]
                pause_start_time = p1.time
        else:
            if pause_start is not None:
                duration = (p1.time - (pause_start_time or p1.time)).total_seconds()
                if duration >= min_minutes * 60:
                    detail = f"Pause détectée (≈{duration/60:.0f} min)"
                    events.append(
                        Event(
                            distance_m=pause_start_dist,
                            description=detail,
                            time_start=pause_start_time,
                            time_end=p1.time,
                        )
                    )
                pause_start = None
    if pause_start is not None and pause_start_time and points[-1].time:
        duration = (points[-1].time - pause_start_time).total_seconds()
        if duration >= min_minutes * 60:
            detail = f"Pause détectée (≈{duration/60:.0f} min)"
            events.append(
                Event(
                    distance_m=pause_start_dist,
                    description=detail,
                    time_start=pause_start_time,
                    time_end=points[-1].time,
                )
            )
    return events


def to_xy(lat: float, lon: float, origin_lat: float) -> Tuple[float, float]:
    x = math.radians(lon) * math.cos(math.radians(origin_lat)) * 6_371_000.0
    y = math.radians(lat) * 6_371_000.0
    return x, y


def point_to_polyline_distance_m(point: Tuple[float, float], polyline: List[Tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return float("inf")
    origin_lat = sum(lat for lat, _ in polyline) / len(polyline)
    px, py = to_xy(point[0], point[1], origin_lat)
    min_dist = float("inf")
    for (lat1, lon1), (lat2, lon2) in zip(polyline, polyline[1:]):
        x1, y1 = to_xy(lat1, lon1, origin_lat)
        x2, y2 = to_xy(lat2, lon2, origin_lat)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            dist = math.hypot(px - x1, py - y1)
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        min_dist = min(min_dist, dist)
    return min_dist


def project_distance_on_polyline(
    point: Tuple[float, float], polyline: List[Tuple[float, float]], distances: List[float]
) -> float:
    if len(polyline) < 2:
        return 0.0
    origin_lat = sum(lat for lat, _ in polyline) / len(polyline)
    px, py = to_xy(point[0], point[1], origin_lat)
    best_dist = float("inf")
    best_pos = 0.0
    for i, ((lat1, lon1), (lat2, lon2)) in enumerate(zip(polyline, polyline[1:])):
        x1, y1 = to_xy(lat1, lon1, origin_lat)
        x2, y2 = to_xy(lat2, lon2, origin_lat)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            proj_dist = math.hypot(px - x1, py - y1)
            frac = 0.0
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            proj_dist = math.hypot(px - proj_x, py - proj_y)
            frac = t
        if proj_dist < best_dist:
            best_dist = proj_dist
            best_pos = distances[i] + frac * (distances[i + 1] - distances[i])
    return best_pos


def time_at_distance(distances: List[float], times: List[Optional[dt.datetime]], target: float) -> Optional[dt.datetime]:
    if not times or any(t is None for t in times):
        return None
    if target <= distances[0]:
        return times[0]
    if target >= distances[-1]:
        return times[-1]
    for i in range(1, len(distances)):
        if distances[i] >= target:
            d0 = distances[i - 1]
            d1 = distances[i]
            t0 = times[i - 1]
            t1 = times[i]
            if t0 is None or t1 is None or d1 == d0:
                return t0
            ratio = (target - d0) / (d1 - d0)
            delta = t1 - t0
            return t0 + dt.timedelta(seconds=delta.total_seconds() * ratio)
    return times[-1]


def format_km(distance_m: float) -> str:
    return f"{distance_m/1000.0:.1f}"


def format_time(dt_value: Optional[dt.datetime]) -> str:
    if dt_value is None:
        return ""
    return dt_value.strftime("%Y-%m-%d %H:%M")


def format_time_range(start: Optional[dt.datetime], end: Optional[dt.datetime]) -> str:
    if start is None:
        return ""
    if end is None:
        return format_time(start)
    if start.date() == end.date():
        return f"{start.strftime('%Y-%m-%d %H:%M')}–{end.strftime('%H:%M')}"
    return f"{format_time(start)}–{format_time(end)}"


def format_event(event: Event, total_distances: List[float], times: List[Optional[dt.datetime]]) -> str:
    if event.range_start_m is not None and event.range_end_m is not None:
        km_label = f"km {format_km(event.range_start_m)}–{format_km(event.range_end_m)}"
        time_label = format_time_range(event.time_start, event.time_end)
    else:
        km_label = f"km {format_km(event.distance_m)}"
        if event.time_start and event.time_end:
            time_label = format_time_range(event.time_start, event.time_end)
        else:
            time_label = format_time(time_at_distance(total_distances, times, event.distance_m))
    if time_label:
        return f"- [{km_label} | {time_label}] {event.description}"
    return f"- [{km_label}] {event.description}"


def overpass_cache_path() -> Path:
    return Path(".gpx_to_markdown_cache.json")


def load_overpass_cache() -> Dict[str, Dict]:
    path = overpass_cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_overpass_cache(cache: Dict[str, Dict]) -> None:
    path = overpass_cache_path()
    path.write_text(json.dumps(cache), encoding="utf-8")


def overpass_query_text(
    bbox: Tuple[float, float, float, float],
    timeout_s: int,
    include_highways: bool,
    include_pois: bool,
) -> str:
    south, west, north, east = bbox
    selectors: List[str] = []
    if include_pois:
        selectors.extend(
            [
                'nwr[place~"^(city|town|village|hamlet)$"]',
                "nwr[natural=peak]",
                "nwr[amenity=place_of_worship][religion=christian]",
                "nwr[building=church]",
                "nwr[waterway=river]",
                "nwr[waterway=stream]",
                "nwr[waterway=waterfall]",
                "nwr[natural=waterfall]",
                "nwr[natural=water][water=lake]",
                "nwr[water=lake]",
                "nwr[natural=cave_entrance]",
                "nwr[natural=gorge]",
                "nwr[historic=ruins]",
                "nwr[building=ruins]",
                "nwr[ruins=yes]",
                "nwr[building=chapel]",
                "nwr[amenity=place_of_worship][place_of_worship=chapel]",
                "nwr[historic=monument]",
                "nwr[historic=memorial]",
                "nwr[memorial]",
                "nwr[landuse=farmyard]",
                "nwr[building=farm]",
                "nwr[amenity=farm]",
                "nwr[landuse=pasture]",
                "nwr[livestock]",
                "nwr[animal]",
                "nwr[landuse=forest]",
                "nwr[natural=wood]",
                "nwr[landuse=farmland]",
                "nwr[landuse=meadow]",
                "nwr[man_made=windmill]",
                'nwr[power=generator]["generator:source"=wind]',
                "nwr[power=line]",
                "nwr[power=minor_line]",
                "nwr[railway=rail]",
                "nwr[bridge=yes]",
                "nwr[man_made=bridge]",
                "nwr[tunnel=yes]",
            ]
        )
    if include_highways:
        selectors.extend(
            [
                "nwr[highway][name]",
                "nwr[highway][ref]",
            ]
        )
    if not selectors:
        return ""
    selector_lines = "\n".join(
        f"      {selector}({south},{west},{north},{east});" for selector in selectors
    )
    return f"""
    [out:json][timeout:{timeout_s}];
    (
{selector_lines}
    );
    out center tags;
    """


def overpass_query(
    bbox: Tuple[float, float, float, float],
    overpass_url: str,
    cache: Dict[str, Dict],
    timeout: int = 90,
    split_on_fail: bool = True,
    debug: bool = False,
    include_highways: bool = True,
    include_pois: bool = True,
) -> Optional[Dict]:
    query = overpass_query_text(bbox, timeout, include_highways, include_pois)
    if not query:
        return None
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if key in cache:
        if debug:
            click.echo("Overpass cache hit.", err=True)
        return cache[key]
    if debug:
        south, west, north, east = bbox
        click.echo(
            f"Overpass bbox: south={south:.6f} west={west:.6f} north={north:.6f} east={east:.6f}",
            err=True,
        )
    headers = {"User-Agent": "gpx-to-markdown/1.0"}
    data = request_json(
        overpass_url,
        params={"data": query},
        timeout=timeout,
        debug=debug,
        label="Overpass",
        headers=headers,
        method="post",
    )
    if data:
        if debug:
            count = len(data.get("elements", []))
            click.echo(f"Overpass returned {count} elements.", err=True)
        cache[key] = data
        return data
    elif debug:
        click.echo("Overpass request failed or returned no data.", err=True)
    if not split_on_fail:
        return None
    if debug:
        click.echo("Overpass split retry (2x2).", err=True)
    south, west, north, east = bbox
    mid_lat = (south + north) / 2.0
    mid_lon = (west + east) / 2.0
    tiles = [
        (south, west, mid_lat, mid_lon),
        (south, mid_lon, mid_lat, east),
        (mid_lat, west, north, mid_lon),
        (mid_lat, mid_lon, north, east),
    ]
    merged: Dict[str, Dict] = {"elements": []}
    seen = set()
    for tile in tiles:
        tile_data = overpass_query(
            tile,
            overpass_url,
            cache,
            timeout=timeout,
            split_on_fail=False,
            debug=debug,
            include_highways=include_highways,
            include_pois=include_pois,
        )
        if not tile_data:
            continue
        for element in tile_data.get("elements", []):
            element_id = (element.get("type"), element.get("id"))
            if element_id in seen:
                continue
            seen.add(element_id)
            merged["elements"].append(element)
    if merged["elements"]:
        if debug:
            click.echo(f"Overpass merged {len(merged['elements'])} elements.", err=True)
        cache[key] = merged
        return merged
    return None


def poi_label(tags: Dict[str, str]) -> Optional[str]:
    place = tags.get("place")
    if place in {"city", "town"}:
        return "la ville"
    if place in {"village", "hamlet"}:
        return "le village"
    if tags.get("natural") == "peak":
        return "le sommet"
    if tags.get("amenity") == "place_of_worship" or tags.get("building") == "church":
        return "l'église"
    if tags.get("waterway") == "river":
        return "la rivière"
    if tags.get("waterway") == "stream":
        return "le ruisseau"
    if tags.get("waterway") == "waterfall" or tags.get("natural") == "waterfall":
        return "la cascade"
    if tags.get("natural") == "water" and tags.get("water") == "lake":
        return "le lac"
    if tags.get("water") == "lake":
        return "le lac"
    if tags.get("natural") == "cave_entrance":
        return "la grotte"
    if tags.get("natural") == "gorge":
        return "la gorge"
    if tags.get("historic") == "ruins" or tags.get("building") == "ruins" or tags.get("ruins") == "yes":
        return "les ruines"
    if tags.get("building") == "chapel" or tags.get("place_of_worship") == "chapel":
        return "la chapelle"
    if tags.get("historic") in {"monument", "memorial"} or tags.get("memorial"):
        return "le monument"
    if tags.get("landuse") == "farmyard" or tags.get("building") == "farm" or tags.get("amenity") == "farm":
        return "la ferme"
    if tags.get("landuse") == "pasture" or tags.get("livestock") or tags.get("animal"):
        return "le pâturage"
    if tags.get("landuse") == "forest" or tags.get("natural") == "wood":
        return "la forêt"
    if tags.get("landuse") in {"farmland", "meadow"}:
        return "le champ"
    if tags.get("man_made") == "windmill" or (
        tags.get("power") == "generator" and tags.get("generator:source") == "wind"
    ):
        return "l'éolienne"
    if tags.get("power") in {"line", "minor_line"}:
        return "la ligne électrique"
    if tags.get("railway") == "rail":
        return "la voie ferrée"
    highway = tags.get("highway")
    if highway and (tags.get("name") or tags.get("ref")):
        if highway in {"path", "track", "footway", "bridleway", "cycleway"}:
            return "le chemin"
        return "la route"
    if tags.get("bridge") == "yes" or tags.get("man_made") == "bridge":
        return "le pont"
    if tags.get("tunnel") == "yes":
        return "le tunnel"
    return None


def extract_pois(
    match_polyline: List[Tuple[float, float]],
    match_distances: List[float],
    poi_radius_m: float,
    overpass_url: str,
    overpass_timeout: int = 90,
    overpass_split: bool = True,
    debug: bool = False,
) -> List[Tuple[float, str]]:
    if not match_polyline:
        return []
    lats = [lat for lat, _ in match_polyline]
    lons = [lon for _, lon in match_polyline]
    mean_lat = sum(lats) / len(lats)
    delta_lat = poi_radius_m / M_PER_DEG
    delta_lon = poi_radius_m / (M_PER_DEG * math.cos(math.radians(mean_lat)))
    bbox = (min(lats) - delta_lat, min(lons) - delta_lon, max(lats) + delta_lat, max(lons) + delta_lon)
    cache = load_overpass_cache()
    data = overpass_query(
        bbox,
        overpass_url,
        cache,
        timeout=overpass_timeout,
        split_on_fail=overpass_split,
        debug=debug,
        include_highways=False,
        include_pois=True,
    )
    if data:
        save_overpass_cache(cache)
    if not data:
        return []
    features: List[Tuple[float, str]] = []
    seen = set()
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        label = poi_label(tags)
        if not label:
            continue
        if "lat" in element and "lon" in element:
            lat = element["lat"]
            lon = element["lon"]
        elif "center" in element:
            lat = element["center"]["lat"]
            lon = element["center"]["lon"]
        else:
            continue
        distance = point_to_polyline_distance_m((lat, lon), match_polyline)
        if distance > poi_radius_m:
            continue
        name = tags.get("name") or tags.get("ref") or ""
        key = (label, name)
        if key in seen:
            continue
        seen.add(key)
        position = project_distance_on_polyline((lat, lon), match_polyline, match_distances)
        if name:
            description = f'Passer près de {label} "{name}"'
        else:
            description = f"Passer près de {label}"
        features.append((position, description))
    return features


def extract_overpass_step_names(
    match_polyline: List[Tuple[float, float]],
    match_distances: List[float],
    road_radius_m: float,
    overpass_url: str,
    overpass_timeout: int = 90,
    overpass_split: bool = True,
    debug: bool = False,
) -> List[StepName]:
    if not match_polyline:
        return []
    lats = [lat for lat, _ in match_polyline]
    lons = [lon for _, lon in match_polyline]
    mean_lat = sum(lats) / len(lats)
    delta_lat = road_radius_m / M_PER_DEG
    delta_lon = road_radius_m / (M_PER_DEG * math.cos(math.radians(mean_lat)))
    bbox = (min(lats) - delta_lat, min(lons) - delta_lon, max(lats) + delta_lat, max(lons) + delta_lon)
    cache = load_overpass_cache()
    data = overpass_query(
        bbox,
        overpass_url,
        cache,
        timeout=overpass_timeout,
        split_on_fail=overpass_split,
        debug=debug,
        include_highways=True,
        include_pois=False,
    )
    if data:
        save_overpass_cache(cache)
    if not data:
        return []
    positions_by_name: Dict[str, List[float]] = {}
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        if not tags.get("highway"):
            continue
        name = (tags.get("name") or tags.get("ref") or "").strip()
        if not name:
            continue
        if "lat" in element and "lon" in element:
            lat = element["lat"]
            lon = element["lon"]
        elif "center" in element:
            lat = element["center"]["lat"]
            lon = element["center"]["lon"]
        else:
            continue
        distance = point_to_polyline_distance_m((lat, lon), match_polyline)
        if distance > road_radius_m:
            continue
        position = project_distance_on_polyline((lat, lon), match_polyline, match_distances)
        positions_by_name.setdefault(name, []).append(position)
    steps: List[StepName] = []
    for name, positions in positions_by_name.items():
        start = min(positions)
        end = max(positions)
        steps.append(StepName(max(0.0, start - road_radius_m), end + road_radius_m, name))
    if debug:
        click.echo(
            f"Overpass road names: {len(positions_by_name)} (radius {road_radius_m:.0f} m).",
            err=True,
        )
    return sorted(steps, key=lambda step: step.start_m)


@click.command()
@click.argument("gpx_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--osrm-url", default="https://router.project-osrm.org", show_default=True)
@click.option("--osrm-profile", default="driving", show_default=True)
@click.option("--overpass-url", default="https://overpass-api.de/api/interpreter", show_default=True)
@click.option("--turn-angle", default=80.0, show_default=True, type=float)
@click.option("--min-grade", default=4.0, show_default=True, type=float)
@click.option("--climb-min-minutes", default=3.0, show_default=True, type=float)
@click.option("--elevation-smooth", default=5, show_default=True, type=int)
@click.option("--poi-radius", default=100.0, show_default=True, type=float)
@click.option("--road-radius", default=50.0, show_default=True, type=float)
@click.option("--overpass-timeout", default=90, show_default=True, type=int)
@click.option("--overpass-split/--no-overpass-split", default=True, show_default=True)
@click.option("--osrm-max-points", default=1000, show_default=True, type=int)
@click.option("--osrm-debug", is_flag=True, help="Print OSRM HTTP errors.")
@click.option("--overpass-debug", is_flag=True, help="Print Overpass debug logs.")
@click.option(
    "--name-source",
    type=click.Choice(["overpass", "osrm"]),
    default="overpass",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
def main(
    gpx_path: Path,
    osrm_url: str,
    osrm_profile: str,
    overpass_url: str,
    turn_angle: float,
    min_grade: float,
    climb_min_minutes: float,
    elevation_smooth: int,
    poi_radius: float,
    road_radius: float,
    overpass_timeout: int,
    overpass_split: bool,
    osrm_max_points: int,
    osrm_debug: bool,
    overpass_debug: bool,
    name_source: str,
    output: Optional[Path],
) -> None:
    points = load_gpx_points(gpx_path)
    raw_coords = [(p.lat, p.lon) for p in points]
    raw_distances = cumulative_distances(raw_coords)
    times = [p.time for p in points]

    if name_source == "osrm":
        match_data = osrm_match(points, osrm_url, osrm_profile, osrm_max_points, osrm_debug)
        match_polyline = parse_match_geometry(match_data) if match_data else None
        steps = parse_step_names(match_data) if match_data else []
        if match_polyline:
            match_distances = cumulative_distances(match_polyline)
            distance_scale = match_distances[-1] / raw_distances[-1] if raw_distances[-1] else 1.0
        else:
            click.echo("OSRM match failed; using raw geometry without road names.", err=True)
            match_polyline = raw_coords
            match_distances = raw_distances
            distance_scale = 1.0
    else:
        match_polyline = raw_coords
        match_distances = raw_distances
        distance_scale = 1.0
        steps = extract_overpass_step_names(
            match_polyline,
            match_distances,
            road_radius,
            overpass_url,
            overpass_timeout=overpass_timeout,
            overpass_split=overpass_split,
            debug=overpass_debug,
        )

    events: List[Event] = []
    for event in detect_turns(match_polyline, match_distances, turn_angle, steps):
        raw_distance = event.distance_m / distance_scale if distance_scale else event.distance_m
        events.append(
            Event(
                distance_m=raw_distance,
                description=event.description,
                range_start_m=event.range_start_m,
                range_end_m=event.range_end_m,
                time_start=event.time_start,
                time_end=event.time_end,
            )
        )
    events.extend(
        detect_climbs(
            points,
            raw_distances,
            min_grade,
            climb_min_minutes,
            elevation_smooth,
            steps,
            distance_scale,
        )
    )
    events.extend(detect_pauses(points, raw_distances, 5.0))

    poi_entries = extract_pois(
        match_polyline,
        match_distances,
        poi_radius,
        overpass_url,
        overpass_timeout=overpass_timeout,
        overpass_split=overpass_split,
        debug=overpass_debug,
    )
    for position, description in poi_entries:
        scaled_position = position / distance_scale if distance_scale else position
        events.append(Event(distance_m=scaled_position, description=description))

    events.sort(key=lambda e: e.distance_m)
    output_lines = [format_event(event, raw_distances, times) for event in events]
    content = "\n".join(output_lines)
    if output:
        output.write_text(content + "\n", encoding="utf-8")
    else:
        click.echo(content)


if __name__ == "__main__":
    main()
