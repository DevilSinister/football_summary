from dataclasses import dataclass
import re
from typing import Tuple


RGBColor = Tuple[int, int, int]


def parse_hex_color(value: str) -> RGBColor:
    """Convert a user-facing #RRGGBB color into an RGB tuple."""
    if not isinstance(value, str) or not re.fullmatch(r"#?[0-9a-fA-F]{6}", value.strip()):
        raise ValueError(f"Invalid kit color {value!r}; expected #RRGGBB")
    normalized = value.strip().lstrip("#")
    return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))


@dataclass(frozen=True)
class TeamConfig:
    team_id: int
    name: str
    primary_color_rgb: RGBColor

    def __post_init__(self) -> None:
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be 1 or 2")
        if not self.name.strip():
            raise ValueError("team name cannot be empty")
        if len(self.primary_color_rgb) != 3 or any(
            channel < 0 or channel > 255 for channel in self.primary_color_rgb
        ):
            raise ValueError("primary_color_rgb must contain three values from 0 to 255")

    @property
    def primary_color_bgr(self) -> RGBColor:
        red, green, blue = self.primary_color_rgb
        return blue, green, red


@dataclass(frozen=True)
class MatchConfig:
    team_a: TeamConfig
    team_b: TeamConfig

    @classmethod
    def from_user_input(
        cls,
        team_a_name: str,
        team_a_color: str,
        team_b_name: str,
        team_b_color: str,
    ) -> "MatchConfig":
        return cls(
            team_a=TeamConfig(1, team_a_name.strip(), parse_hex_color(team_a_color)),
            team_b=TeamConfig(2, team_b_name.strip(), parse_hex_color(team_b_color)),
        )

    @classmethod
    def from_detected_colors(
        cls,
        team_a_name: str,
        team_a_color_rgb: RGBColor,
        team_b_name: str,
        team_b_color_rgb: RGBColor,
    ) -> "MatchConfig":
        """Build configuration from colors calculated by the video pipeline."""
        return cls(
            team_a=TeamConfig(1, team_a_name.strip(), tuple(team_a_color_rgb)),
            team_b=TeamConfig(2, team_b_name.strip(), tuple(team_b_color_rgb)),
        )

    @property
    def teams(self) -> Tuple[TeamConfig, TeamConfig]:
        return self.team_a, self.team_b
