"""
Optional spatially uniform wind for the dispersion model.

The wind is a compass direction (degrees, the bearing the wind blows FROM,
0 = from north, 90 = from east) and a speed [m/s], settable per hour of the
simulated day.  It is off by default.  SUMO/grid axes are x = east,
y = north, so the transport velocity components are

    u_x = -speed * sin(direction),   u_y = -speed * cos(direction).

"""

import math
import re

CALM = (0.0, 0.0)   # (direction_deg, speed_m_s), no wind case, default


def wind_components(direction_deg, speed):
    """Transport velocity (u_x, u_y) [m/s] for a wind blowing from `direction_deg`."""
    rad = math.radians(direction_deg)
    ux, uy = -speed * math.sin(rad), -speed * math.cos(rad)
    # avoid -0.0 and 1e-16 residue for the cardinal directions
    return (0.0 if abs(ux) < 1e-12 else ux, 0.0 if abs(uy) < 1e-12 else uy)


class WindSchedule:
    """Wind per hour of day: {hour: (direction_deg, speed)}; unlisted hours are calm."""

    def __init__(self, hourly=None):
        self.hourly = {}
        for hour, (direction, speed) in (hourly or {}).items():
            if not 0 <= int(hour) <= 23:
                raise ValueError(f"hour must be in 0..23, got {hour}")
            if speed < 0:
                raise ValueError(f"wind speed must be >= 0, got {speed}")
            self.hourly[int(hour)] = (float(direction), float(speed))

    @classmethod
    def constant(cls, direction_deg, speed):
        return cls({h: (direction_deg, speed) for h in range(24)})

    def at(self, sim_time_s):
        """(u_x, u_y) [m/s] at simulation time `sim_time_s` [s]."""
        hour = int(sim_time_s // 3600) % 24
        return wind_components(*self.hourly.get(hour, CALM))

    @property
    def active(self):
        return any(speed > 0 for _, speed in self.hourly.values())


def parse_wind_spec(spec):
    """
    Parse a command-line wind specification.

      "DIR:SPEED"                      constant all day, e.g. "315:10"
      "H=DIR:SPEED;H-H=DIR:SPEED;..."  per hour or hour range, e.g. "8-17=315:10;18=100:3"
    """
    if spec is None or not spec.strip():
        return WindSchedule()
    spec = spec.strip()
    if "=" not in spec:
        direction, speed = _parse_dir_speed(spec)
        return WindSchedule.constant(direction, speed)
    hourly = {}
    for part in spec.split(";"):
        hours, value = part.split("=")
        first, _, last = hours.strip().partition("-")
        for hour in range(int(first), int(last or first) + 1):
            hourly[hour] = _parse_dir_speed(value)
    return WindSchedule(hourly)


def _parse_dir_speed(text):
    if not re.fullmatch(r"\s*-?[\d.]+\s*:\s*[\d.]+\s*", text):
        raise ValueError(f"expected DIR:SPEED, got {text!r}")
    direction, speed = text.split(":")
    return float(direction), float(speed)
