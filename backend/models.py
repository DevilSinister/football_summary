from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(150), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    matches: Mapped[list["Match"]] = relationship(back_populates="user")
    logs: Mapped[list["UserLog"]] = relationship(back_populates="user")


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    primary_color: Mapped[str | None] = mapped_column(String(7), nullable=True)

    players: Mapped[list["Player"]] = relationship(back_populates="team")
    plays: Mapped[list["Play"]] = relationship(back_populates="team")


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("team_id", "jersey_no", name="uq_player_team_jersey"),)

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    jersey_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"), nullable=True)

    team: Mapped[Team | None] = relationship(back_populates="players")
    occurrences: Mapped[list["Occurrence"]] = relationship(back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_date: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, unique=True)
    source_video: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="completed")

    user: Mapped[User | None] = relationship(back_populates="matches")
    plays: Mapped[list["Play"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )


class Play(Base):
    __tablename__ = "plays"

    play_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.team_id"), nullable=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.match_id"), nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0)

    team: Mapped[Team | None] = relationship(back_populates="plays")
    match: Mapped[Match] = relationship(back_populates="plays")
    occurrences: Mapped[list["Occurrence"]] = relationship(back_populates="play")


class Event(Base):
    __tablename__ = "events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.match_id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    match: Mapped[Match] = relationship(back_populates="events")
    occurrences: Mapped[list["Occurrence"]] = relationship(back_populates="event")


class Occurrence(Base):
    __tablename__ = "occurrences"

    occurrence_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.event_id"), nullable=False)
    play_id: Mapped[int | None] = mapped_column(ForeignKey("plays.play_id"), nullable=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.player_id"), nullable=True)
    start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detected_jersey_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detected_player_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    detected_team_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    team_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    jersey_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    name_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    event: Mapped[Event] = relationship(back_populates="occurrences")
    play: Mapped[Play | None] = relationship(back_populates="occurrences")
    player: Mapped[Player | None] = relationship(back_populates="occurrences")


class UserLog(Base):
    __tablename__ = "user_logs"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    ip_address: Mapped[str | None] = mapped_column(String(50), nullable=True)

    user: Mapped[User] = relationship(back_populates="logs")
