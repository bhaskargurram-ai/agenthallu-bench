"""Trace logger — wraps every agent action and logs to SQLite.

Must be passed into agent, tool_executor, rag_retriever, memory_manager.
All methods are synchronous.
"""

import json
import logging
import threading
import time
from typing import Any, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session as SASession

from tracer.trace_schema import (
    Base, SessionRecord, StepRecord, GroundTruthRecord, InjectionRecord,
)

logger = logging.getLogger(__name__)


class TraceLogger:
    """Logs every pipeline stage to SQLite. Thread-safe via lock."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self._SessionFactory = sessionmaker(bind=self._engine)
        self._session: Optional[SASession] = None
        self._lock = threading.Lock()
        logger.info("TraceLogger created for %s", db_path)

    def init_db(self) -> None:
        """Create tables if they don't exist."""
        Base.metadata.create_all(self._engine)
        self._session = self._SessionFactory()
        logger.info("Trace DB initialized at %s", self.db_path)

    @property
    def session(self) -> SASession:
        if self._session is None:
            self.init_db()
        return self._session

    def start_session(
        self,
        session_id: str,
        task_id: str,
        model: str,
        domain: str,
        injection_type: Optional[str] = None,
        injection_stage: Optional[str] = None,
    ) -> None:
        """Begin a new agent execution session."""
        with self._lock:
            record = SessionRecord(
                session_id=session_id,
                task_id=task_id,
                model=model,
                domain=domain,
                injection_type=injection_type,
                injection_stage=injection_stage,
                started_at=time.time(),
            )
            self.session.add(record)
            self.session.commit()
        logger.info("Session started: %s (task=%s, model=%s)", session_id, task_id, model)

    def end_session(
        self,
        session_id: str,
        final_answer: str,
        ground_truth: str,
        correct: bool,
    ) -> None:
        """End an agent execution session."""
        with self._lock:
            record = self.session.get(SessionRecord, session_id)
            if record:
                record.ended_at = time.time()
                record.final_answer = final_answer
                record.ground_truth_answer = ground_truth
                record.final_correct = correct
                self.session.commit()
        logger.info("Session ended: %s (correct=%s)", session_id, correct)

    def log_step(
        self,
        session_id: str,
        turn_id: int,
        step_number: int,
        step_type: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_params_raw: Optional[dict] = None,
        tool_params_validated: Optional[dict] = None,
        tool_result: Any = None,
        param_errors: Optional[list] = None,
        retrieval_query: Optional[str] = None,
        retrieval_results: Optional[list] = None,
        memory_state: Optional[dict] = None,
        token_count: int = 0,
    ) -> None:
        """Log a single step in the agent loop."""
        with self._lock:
            record = StepRecord(
                session_id=session_id,
                turn_id=turn_id,
                step_number=step_number,
                step_type=step_type,
                content=content,
                tool_name=tool_name,
                tool_params_raw=json.dumps(tool_params_raw, default=str) if tool_params_raw else None,
                tool_params_validated=json.dumps(tool_params_validated, default=str) if tool_params_validated else None,
                tool_result=json.dumps(tool_result, default=str) if tool_result is not None else None,
                param_errors=json.dumps(param_errors) if param_errors else None,
                retrieval_query=retrieval_query,
                retrieval_results=json.dumps(retrieval_results, default=str) if retrieval_results else None,
                memory_state=json.dumps(memory_state, default=str) if memory_state else None,
                token_count=token_count,
                timestamp=time.time(),
            )
            self.session.add(record)
            self.session.commit()
        logger.info(
            "Step logged: session=%s step=%d type=%s",
            session_id, step_number, step_type,
        )

    def log_retrieval(
        self,
        session_id: str,
        turn_id: int,
        query: str,
        results: list,
        scores: list,
    ) -> None:
        """Log a RAG retrieval step."""
        with self._lock:
            step_count = (
                self.session.query(StepRecord)
                .filter_by(session_id=session_id)
                .count()
            )
        self.log_step(
            session_id=session_id,
            turn_id=turn_id,
            step_number=step_count + 1,
            step_type="retrieval",
            content=f"RAG retrieval for: {query[:200]}",
            retrieval_query=query,
            retrieval_results=[
                {"text": r.get("text", "")[:300], "score": s}
                for r, s in zip(results, scores)
            ],
        )

    def log_memory_write(
        self,
        session_id: str,
        turn_id: int,
        key: str,
        value: Any,
        source_step: int,
    ) -> None:
        """Log a memory write operation."""
        with self._lock:
            step_count = (
                self.session.query(StepRecord)
                .filter_by(session_id=session_id)
                .count()
            )
        self.log_step(
            session_id=session_id,
            turn_id=turn_id,
            step_number=step_count + 1,
            step_type="memory_write",
            content=f"Memory write: {key}={json.dumps(value, default=str)[:200]}",
            memory_state={key: value},
        )

    def log_injection(
        self,
        session_id: str,
        injection_type: str,
        target_stage: str,
        original: Any,
        injected: Any,
        turn_id: int,
        step_number: int = 0,
    ) -> None:
        """Log an error injection event."""
        with self._lock:
            record = InjectionRecord(
                session_id=session_id,
                injection_type=injection_type,
                target_stage=target_stage,
                original_value=json.dumps(original, default=str),
                injected_value=json.dumps(injected, default=str),
                turn_id=turn_id,
                step_number=step_number,
                timestamp=time.time(),
            )
            self.session.add(record)
            self.session.commit()
        logger.info(
            "Injection logged: session=%s type=%s stage=%s",
            session_id, injection_type, target_stage,
        )

    def get_session_trace(self, session_id: str) -> dict:
        """Get full trace as a nested dict."""
        with self._lock:
            return self._get_session_trace_unlocked(session_id)

    def _get_session_trace_unlocked(self, session_id: str) -> dict:
        """Internal unlocked version."""
        sess = self.session.get(SessionRecord, session_id)
        if not sess:
            return {}
        steps = (
            self.session.query(StepRecord)
            .filter_by(session_id=session_id)
            .order_by(StepRecord.step_number)
            .all()
        )
        injections = (
            self.session.query(InjectionRecord)
            .filter_by(session_id=session_id)
            .all()
        )
        return {
            "session_id": sess.session_id,
            "task_id": sess.task_id,
            "model": sess.model,
            "domain": sess.domain,
            "injection_type": sess.injection_type,
            "started_at": sess.started_at,
            "ended_at": sess.ended_at,
            "final_answer": sess.final_answer,
            "ground_truth_answer": sess.ground_truth_answer,
            "final_correct": sess.final_correct,
            "steps": [
                {
                    "step_id": s.step_id,
                    "turn_id": s.turn_id,
                    "step_number": s.step_number,
                    "step_type": s.step_type,
                    "content": s.content,
                    "tool_name": s.tool_name,
                    "tool_params_raw": s.tool_params_raw,
                    "tool_params_validated": s.tool_params_validated,
                    "tool_result": s.tool_result,
                    "param_errors": s.param_errors,
                    "retrieval_query": s.retrieval_query,
                    "retrieval_results": s.retrieval_results,
                    "memory_state": s.memory_state,
                    "token_count": s.token_count,
                    "timestamp": s.timestamp,
                }
                for s in steps
            ],
            "injections": [
                {
                    "injection_id": inj.injection_id,
                    "injection_type": inj.injection_type,
                    "target_stage": inj.target_stage,
                    "original_value": inj.original_value,
                    "injected_value": inj.injected_value,
                    "turn_id": inj.turn_id,
                    "step_number": inj.step_number,
                }
                for inj in injections
            ],
        }

    def export_traces(self, output_path: str) -> None:
        """Export all traces to JSON."""
        with self._lock:
            sessions = self.session.query(SessionRecord).all()
            all_traces = []
            for sess in sessions:
                all_traces.append(self._get_session_trace_unlocked(sess.session_id))
        with open(output_path, "w") as f:
            json.dump(all_traces, f, indent=2, default=str)
        logger.info("Exported %d traces to %s", len(all_traces), output_path)

    def get_step_count(self, session_id: str) -> int:
        """Get the number of steps in a session."""
        with self._lock:
            return (
                self.session.query(StepRecord)
                .filter_by(session_id=session_id)
                .count()
            )

    def close(self) -> None:
        """Close the database session."""
        if self._session:
            self._session.close()
            self._session = None
