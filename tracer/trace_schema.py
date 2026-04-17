"""SQLite trace schema using SQLAlchemy ORM.

Defines 4 tables: sessions, steps, ground_truth, injections.
All agent actions are traced to SQLite — never lose data.
"""

import logging

from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text,
    ForeignKey, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SASession

logger = logging.getLogger(__name__)

Base = declarative_base()


class SessionRecord(Base):
    """One row per agent execution session."""
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    task_id = Column(String, index=True)
    model = Column(String)
    domain = Column(String)
    injection_type = Column(String, nullable=True)
    injection_stage = Column(String, nullable=True)
    started_at = Column(Float)
    ended_at = Column(Float, nullable=True)
    final_answer = Column(Text, nullable=True)
    ground_truth_answer = Column(Text, nullable=True)
    final_correct = Column(Boolean, nullable=True)


class StepRecord(Base):
    """One row per step in the agent loop."""
    __tablename__ = "steps"

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), index=True)
    turn_id = Column(Integer, default=0)
    step_number = Column(Integer)
    step_type = Column(String)  # thought | action | observation | retrieval | memory_write | final
    content = Column(Text, nullable=True)
    tool_name = Column(String, nullable=True)
    tool_params_raw = Column(Text, nullable=True)       # JSON
    tool_params_validated = Column(Text, nullable=True)  # JSON
    tool_result = Column(Text, nullable=True)            # JSON
    param_errors = Column(Text, nullable=True)           # JSON
    retrieval_query = Column(Text, nullable=True)
    retrieval_results = Column(Text, nullable=True)      # JSON
    memory_state = Column(Text, nullable=True)           # JSON
    token_count = Column(Integer, default=0)
    timestamp = Column(Float)


class GroundTruthRecord(Base):
    """Ground truth for each task."""
    __tablename__ = "ground_truth"

    task_id = Column(String, primary_key=True)
    domain = Column(String)
    query = Column(Text)
    correct_tool = Column(String)
    correct_params = Column(Text)   # JSON
    correct_answer = Column(Text)
    difficulty = Column(String)     # easy | medium | hard


class InjectionRecord(Base):
    """Error injection log."""
    __tablename__ = "injections"

    injection_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), index=True)
    injection_type = Column(String)
    target_stage = Column(String)
    original_value = Column(Text, nullable=True)
    injected_value = Column(Text, nullable=True)
    turn_id = Column(Integer, nullable=True)
    step_number = Column(Integer, nullable=True)
    timestamp = Column(Float)


def create_tables(db_path: str) -> SASession:
    """Create all tables if they don't exist and return a session."""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    logger.info("Trace DB tables created at %s", db_path)
    return Session()
