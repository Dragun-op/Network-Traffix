import datetime as dt

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import Base, init_db, make_engine
from app.models_orm import Incident, ModelVersion


@pytest.fixture
def db_session():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_tables_created(db_session):
    assert db_session.query(Incident).count() == 0
    assert db_session.query(ModelVersion).count() == 0


def test_insert_and_read_incident(db_session):
    inc = Incident(
        id="INC-000001",
        timestamp=dt.datetime.now(dt.timezone.utc),
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="TCP",
        attack_type="Port Scan",
        severity="high",
        confidence=88,
        status="new",
        packet_count=1200,
        explanation=[{"feature": "SYN Flag Count", "contribution": 100}],
    )
    db_session.add(inc)
    db_session.commit()

    fetched = db_session.get(Incident, "INC-000001")
    assert fetched is not None
    assert fetched.severity == "high"
    assert fetched.explanation[0]["feature"] == "SYN Flag Count"


def test_model_version_relationship(db_session):
    mv = ModelVersion(
        trained_at=dt.datetime.now(dt.timezone.utc),
        algorithm="XGBoost",
        threshold=0.5,
        metrics={"precision": 0.9, "recall": 0.85, "false_positive_rate": 0.03},
    )
    db_session.add(mv)
    db_session.commit()

    inc = Incident(
        id="INC-000002",
        timestamp=dt.datetime.now(dt.timezone.utc),
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="UDP",
        attack_type="DDoS",
        severity="critical",
        confidence=97,
        status="new",
        packet_count=50000,
        explanation=[],
        model_version_id=mv.id,
    )
    db_session.add(inc)
    db_session.commit()

    assert inc.model_version.algorithm == "XGBoost"
