#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
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
    elevations = smooth_elevations([p.ele for p in points], elevation_smooth)
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


def point_segment_distance_xy(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


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


def simplify_polyline_indices(polyline: List[Tuple[float, float]], tolerance_m: float) -> List[int]:
    if len(polyline) < 3 or tolerance_m <= 0:
        return list(range(len(polyline)))
    origin_lat = sum(lat for lat, _ in polyline) / len(polyline)
    xy = [to_xy(lat, lon, origin_lat) for lat, lon in polyline]
    keep = [False] * len(polyline)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(polyline) - 1)]
    while stack:
        start, end = stack.pop()
        ax, ay = xy[start]
        bx, by = xy[end]
        max_dist = 0.0
        index = None
        for i in range(start + 1, end):
            px, py = xy[i]
            dist = point_segment_distance_xy(px, py, ax, ay, bx, by)
            if dist > max_dist:
                max_dist = dist
                index = i
        if index is not None and max_dist > tolerance_m:
            keep[index] = True
            stack.append((start, index))
            stack.append((index, end))
    return [i for i, flag in enumerate(keep) if flag]


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x = x0
    y = y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def render_ascii(
    points: List[TrackPoint],
    width: int,
    height: int,
    labels: Optional[List[Tuple[float, float, str]]] = None,
    path_char: str = "#",
) -> str:
    if not points:
        return ""
    width = max(10, width)
    height = max(5, height)
    origin_lat = sum(p.lat for p in points) / len(points)
    xy = [to_xy(p.lat, p.lon, origin_lat) for p in points]
    min_x = min(x for x, _ in xy)
    max_x = max(x for x, _ in xy)
    min_y = min(y for _, y in xy)
    max_y = max(y for _, y in xy)
    span_x = max_x - min_x
    span_y = max_y - min_y
    pad = max(span_x, span_y) * 0.05
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x == 0:
        span_x = 1.0
    if span_y == 0:
        span_y = 1.0
    scale = min((width - 1) / span_x, (height - 1) / span_y)
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def to_grid(x: float, y: float) -> Tuple[int, int]:
        gx = int(round((x - min_x) * scale))
        gy = int(round((max_y - y) * scale))
        gx = max(0, min(width - 1, gx))
        gy = max(0, min(height - 1, gy))
        return gx, gy

    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        gx0, gy0 = to_grid(x0, y0)
        gx1, gy1 = to_grid(x1, y1)
        for gx, gy in bresenham_line(gx0, gy0, gx1, gy1):
            grid[gy][gx] = path_char

    start_x, start_y = to_grid(*xy[0])
    end_x, end_y = to_grid(*xy[-1])
    grid[start_y][start_x] = "S"
    grid[end_y][end_x] = "E"

    if labels:
        for lat, lon, name in labels:
            gx, gy = to_grid(*to_xy(lat, lon, origin_lat))
            if 0 <= gy < height:
                label = name[: width - gx]
                for offset, ch in enumerate(label):
                    if 0 <= gx + offset < width:
                        grid[gy][gx + offset] = ch

    return "\n".join("".join(row).rstrip() for row in grid)


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


def smooth_elevations(elevations: List[Optional[float]], window: int) -> List[Optional[float]]:
    smooth_window = max(1, window)
    half_window = smooth_window // 2
    if smooth_window <= 1:
        return elevations
    smoothed: List[Optional[float]] = []
    for i in range(len(elevations)):
        start = max(0, i - half_window)
        end = min(len(elevations), i + half_window + 1)
        window_vals = [v for v in elevations[start:end] if v is not None]
        if window_vals:
            smoothed.append(sum(window_vals) / len(window_vals))
        else:
            smoothed.append(None)
    return smoothed


def total_elevation_gain_loss(
    points: List[TrackPoint], elevation_smooth: int
) -> Tuple[float, float, Optional[float]]:
    elevations = smooth_elevations([p.ele for p in points], elevation_smooth)
    gain = 0.0
    loss = 0.0
    max_ele = None
    for e1, e2 in zip(elevations, elevations[1:]):
        if e1 is None or e2 is None:
            continue
        if max_ele is None or e2 > max_ele:
            max_ele = e2
        delta = e2 - e1
        if delta > 0:
            gain += delta
        elif delta < 0:
            loss += -delta
    return gain, loss, max_ele


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0 min"
    hours = int(seconds // 3600)
    minutes = int(round((seconds % 3600) / 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    if hours > 0:
        return f"{hours:d} h {minutes:02d} min"
    return f"{minutes:d} min"


def summarize_track(points: List[TrackPoint], distances: List[float], elevation_smooth: int) -> str:
    total_km = distances[-1] / 1000.0 if distances else 0.0
    gain, loss, max_ele = total_elevation_gain_loss(points, elevation_smooth)
    parts = [f"Distance {total_km:.1f} km", f"D+ {gain:.0f} m", f"D- {loss:.0f} m"]
    if max_ele is not None:
        parts.append(f"Alt max {max_ele:.0f} m")
    if points and points[0].time and points[-1].time:
        total_seconds = (points[-1].time - points[0].time).total_seconds()
        parts.append(f"Temps {format_duration(total_seconds)}")
    return " \u00b7 ".join(parts)


def turn_direction(description: str) -> Optional[str]:
    text = description.lower()
    if "gauche" in text:
        return "gauche"
    if "droite" in text:
        return "droite"
    return None


def cluster_turn_events(events: List[Event], cluster_radius_m: float, min_cluster_size: int = 3) -> List[Event]:
    if not events or cluster_radius_m <= 0:
        return events
    clustered: List[Event] = []
    i = 0
    while i < len(events):
        start = events[i]
        cluster = [start]
        j = i + 1
        while j < len(events) and events[j].distance_m - start.distance_m <= cluster_radius_m:
            cluster.append(events[j])
            j += 1
        if len(cluster) >= min_cluster_size:
            directions = [turn_direction(e.description) for e in cluster]
            left = sum(1 for d in directions if d == "gauche")
            right = sum(1 for d in directions if d == "droite")
            if left and right:
                detail = f"Enchaînement de {len(cluster)} virages"
            elif left:
                detail = f"Série de {len(cluster)} virages à gauche"
            elif right:
                detail = f"Série de {len(cluster)} virages à droite"
            else:
                detail = f"Enchaînement de {len(cluster)} virages"
            clustered.append(Event(distance_m=start.distance_m, description=detail))
        else:
            clustered.extend(cluster)
        i = j
    return clustered


def filter_turn_events(events: List[Event], min_spacing_m: float) -> List[Event]:
    if not events or min_spacing_m <= 0:
        return events
    filtered: List[Event] = []
    last_distance = None
    for event in events:
        if last_distance is None or event.distance_m - last_distance >= min_spacing_m:
            filtered.append(event)
            last_distance = event.distance_m
    return filtered


def format_event(
    event: Event,
    total_distances: List[float],
    times: List[Optional[dt.datetime]],
    obsidian: bool,
) -> str:
    if event.range_start_m is not None and event.range_end_m is not None:
        km_label = f"km {format_km(event.range_start_m)}–{format_km(event.range_end_m)}"
        time_label = format_time_range(event.time_start, event.time_end)
    else:
        km_label = f"km {format_km(event.distance_m)}"
        if event.time_start and event.time_end:
            time_label = format_time_range(event.time_start, event.time_end)
        else:
            time_label = format_time(time_at_distance(total_distances, times, event.distance_m))
    description = obsidianize_labels(event.description) if obsidian else event.description
    if time_label:
        return f"- [{km_label} | {time_label}] {description}"
    return f"- [{km_label}] {description}"


def obsidianize_labels(text: str) -> str:
    def replace(match: re.Match) -> str:
        label = match.group(1).strip()
        if not label:
            return match.group(0)
        if label.startswith("[[") and label.endswith("]]"):
            return label
        return f"[[{label}]]"

    return re.sub(r'"([^"]+)"', replace, text)


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


def extract_place_labels(
    points: List[TrackPoint],
    label_radius_m: float,
    overpass_url: str,
    overpass_timeout: int = 90,
    overpass_split: bool = True,
    debug: bool = False,
) -> List[Tuple[float, float, str]]:
    if not points:
        return []
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    mean_lat = sum(lats) / len(lats)
    delta_lat = label_radius_m / M_PER_DEG
    delta_lon = label_radius_m / (M_PER_DEG * math.cos(math.radians(mean_lat)))
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
    labels: List[Tuple[float, float, str]] = []
    seen = set()
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        place = tags.get("place")
        if place not in {"city", "town", "village", "hamlet"}:
            continue
        name = (tags.get("name") or "").strip()
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
        key = (round(lat, 6), round(lon, 6), name)
        if key in seen:
            continue
        seen.add(key)
        labels.append((lat, lon, name))
    return labels


def extract_poi_labels(
    points: List[TrackPoint],
    label_radius_m: float,
    overpass_url: str,
    overpass_timeout: int = 90,
    overpass_split: bool = True,
    debug: bool = False,
) -> List[Tuple[float, float, str]]:
    if not points:
        return []
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    mean_lat = sum(lats) / len(lats)
    delta_lat = label_radius_m / M_PER_DEG
    delta_lon = label_radius_m / (M_PER_DEG * math.cos(math.radians(mean_lat)))
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
    labels: List[Tuple[float, float, str]] = []
    seen = set()
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        if not poi_label(tags):
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
        key = (round(lat, 6), round(lon, 6), name)
        if key in seen:
            continue
        seen.add(key)
        labels.append((lat, lon, name))
    return labels


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
@click.option(
    "--verbosity",
    type=click.Choice(["human", "detailed"]),
    default="human",
    show_default=True,
)
@click.option("--simplify-tolerance", default=20.0, show_default=True, type=float)
@click.option("--turn-min-spacing", default=120.0, show_default=True, type=float)
@click.option("--turn-cluster-radius", default=200.0, show_default=True, type=float)
@click.option("--ascii/--no-ascii", default=False, show_default=True)
@click.option("--ascii-width", default=80, show_default=True, type=int)
@click.option("--ascii-height", default=25, show_default=True, type=int)
@click.option("--ascii-labels/--no-ascii-labels", default=True, show_default=True)
@click.option("--ascii-poi-labels/--no-ascii-poi-labels", default=True, show_default=True)
@click.option("--ascii-label-radius", default=2000.0, show_default=True, type=float)
@click.option("--obsidian", is_flag=True, help="Wrap quoted labels in [[...]] for Obsidian.")
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
    verbosity: str,
    simplify_tolerance: float,
    turn_min_spacing: float,
    turn_cluster_radius: float,
    ascii: bool,
    ascii_width: int,
    ascii_height: int,
    ascii_labels: bool,
    ascii_poi_labels: bool,
    ascii_label_radius: float,
    obsidian: bool,
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
    turn_polyline = match_polyline
    turn_distances = match_distances
    if verbosity == "human":
        indices = simplify_polyline_indices(match_polyline, simplify_tolerance)
        turn_polyline = [match_polyline[i] for i in indices]
        turn_distances = [match_distances[i] for i in indices]

    scaled_turns: List[Event] = []
    for event in detect_turns(turn_polyline, turn_distances, turn_angle, steps):
        raw_distance = event.distance_m / distance_scale if distance_scale else event.distance_m
        scaled_turns.append(
            Event(
                distance_m=raw_distance,
                description=event.description,
                range_start_m=event.range_start_m,
                range_end_m=event.range_end_m,
                time_start=event.time_start,
                time_end=event.time_end,
            )
        )
    if verbosity == "human":
        scaled_turns = cluster_turn_events(scaled_turns, turn_cluster_radius)
        scaled_turns = filter_turn_events(scaled_turns, turn_min_spacing)
    events.extend(scaled_turns)
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
    output_lines = [format_event(event, raw_distances, times, obsidian) for event in events]
    content_sections: List[str] = []
    if ascii:
        labels = []
        if ascii_labels:
            labels.extend(
                extract_place_labels(
                    points,
                    ascii_label_radius,
                    overpass_url,
                    overpass_timeout=overpass_timeout,
                    overpass_split=overpass_split,
                    debug=overpass_debug,
                )
            )
        if ascii_poi_labels:
            labels.extend(
                extract_poi_labels(
                    points,
                    ascii_label_radius,
                    overpass_url,
                    overpass_timeout=overpass_timeout,
                    overpass_split=overpass_split,
                    debug=overpass_debug,
                )
            )
        content_sections.append(
            render_ascii(points, ascii_width, ascii_height, labels=labels or None)
        )
    if verbosity == "human":
        content_sections.append(summarize_track(points, raw_distances, elevation_smooth))
    content_sections.append("\n".join(output_lines))
    content = "\n\n".join(section for section in content_sections if section)
    if output:
        output.write_text(content + "\n", encoding="utf-8")
    else:
        click.echo(content)


if __name__ == "__main__":
    main()
