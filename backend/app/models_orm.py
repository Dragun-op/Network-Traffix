"""
ORM tables. incidents is the hot path table the API serves from;
model_versions and metric_snapshots exist so /api/metrics has something
real to report instead of a stub.
"""
import datetime as dt

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    trained_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    algorithm: Mapped[str] = mapped_column(String)
    threshold: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    incidents: Mapped[list["Incident"]] = relationship(back_populates="model_version")


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        CheckConstraint("severity in ('low','medium','high','critical')", name="ck_severity"),
        CheckConstraint("status in ('new','investigating','resolved')", name="ck_status"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "INC-XXXXXX"
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    src_ip: Mapped[str] = mapped_column(String)
    dst_ip: Mapped[str] = mapped_column(String)
    protocol: Mapped[str] = mapped_column(String)
    attack_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    confidence: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="new")
    packet_count: Mapped[int] = mapped_column(Integer)
    # cached at creation time -- SHAP is too slow to recompute per GET request
    explanation: Mapped[list] = mapped_column(JSON, default=list)

    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="incidents")


class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    threshold: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    false_positive_rate: Mapped[float] = mapped_column(Float)
