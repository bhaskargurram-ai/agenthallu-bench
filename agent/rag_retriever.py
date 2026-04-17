"""ChromaDB RAG retriever with synthetic domain documents.

Creates 3 collections (weather_docs, calendar_docs, medical_docs)
with 50 synthetic documents each. Uses sentence-transformers for embedding.
"""

import logging
import random
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, RAG_CHUNK_SIZE, RAG_TOP_K, RANDOM_SEED

logger = logging.getLogger(__name__)

# ── Synthetic Document Generators ─────────────────────────────────────────────

CITIES = [
    "New York", "London", "Tokyo", "Paris", "Sydney", "Mumbai", "Berlin",
    "Toronto", "São Paulo", "Cairo", "Seoul", "Moscow", "Dubai", "Singapore",
    "Bangkok", "Rome", "Istanbul", "Beijing", "Mexico City", "Lagos",
]

WEATHER_TEMPLATES = [
    "Weather station {city} ({station_id}): Located at elevation {elev}m. Average annual temperature {avg_temp}°C. Annual rainfall {rain}mm. Climate zone: {zone}. Station operational since {year}.",
    "Climate report for {city}: The {season} season typically brings {condition} weather with temperatures ranging from {low}°C to {high}°C. Humidity averages {humidity}% during this period. Wind speeds average {wind} km/h.",
    "Historical weather analysis for {city}: Over the past {n_years} years, {city} has experienced a {trend} trend in average temperatures. Extreme weather events occur approximately {events} times per year. The hottest recorded temperature was {max_temp}°C on {hot_date}.",
    "Weather API documentation: The get_weather endpoint accepts city (string), date (YYYY-MM-DD), and unit (celsius|fahrenheit). Returns temperature, humidity, and condition. Rate limit: {limit} requests/minute.",
    "Forecast methodology for {city}: Our {n_day}-day forecast model uses {model} with {accuracy}% accuracy for day-1 predictions. Accuracy decreases by approximately {decay}% per additional day. Hourly forecasts are available for the next 48 hours.",
]

CALENDAR_TEMPLATES = [
    "Calendar system documentation: Events require title (string), date (YYYY-MM-DD), time (HH:MM), and duration_minutes (integer, 1-1440). Attendees is an optional list of email addresses. Events are stored in UTC and converted to local time for display.",
    "Calendar best practices: When scheduling meetings, consider time zones of all attendees. Default meeting duration is {dur} minutes. Back-to-back meetings should have a {buffer}-minute buffer. Maximum attendees per event: {max_att}.",
    "Event management guide: To create recurring events, use the create_event endpoint with recurrence rules. Events can be modified up to {mod_hours} hours before start time. Cancellation notifications are sent to all attendees by default.",
    "Calendar integration: The calendar API supports {n_cals} concurrent calendars per user. Primary calendar is the default. Events can be color-coded with {n_colors} available colors. Calendar sharing supports read, write, and admin permissions.",
    "Scheduling conflicts: The system detects scheduling conflicts for events with overlapping times. Conflict resolution priority: {priority}. Double-booking is allowed for calendars with overlap_allowed=true. Maximum {max_events} events per day.",
]

MEDICAL_TEMPLATES = [
    "Patient record system: Records are identified by patient_id (format: P###). Available fields: name, dob, conditions, medications, allergies, lab_results, vitals, insurance. Access requires HIPAA authorization level {level}.",
    "Drug interaction database: Contains {n_interactions} known interactions. Severity levels: none, low, moderate, high, critical. The check_drug_interaction endpoint compares two drugs and returns interaction status, severity, and clinical details.",
    "Medical reference for {condition}: {condition} is a {type} condition affecting approximately {prevalence}% of the population. Common medications include {meds}. Regular monitoring of {markers} is recommended every {interval} months.",
    "Appointment scheduling: Appointments require patient_id, doctor_id, date (YYYY-MM-DD), and appointment_type (checkup|follow_up|specialist|emergency). {type_desc}. Standard appointment duration: {dur} minutes.",
    "Clinical guidelines for {drug}: Recommended dosage for adults: {dose}. Contraindications: {contra}. Common side effects: {sides}. Drug class: {drug_class}. Monitoring: {monitoring} levels every {freq} weeks.",
]


def _generate_weather_docs(rng: random.Random) -> list[tuple[str, dict]]:
    """Generate 50 synthetic weather documents."""
    docs = []
    seasons = ["spring", "summer", "autumn", "winter"]
    conditions = ["warm and sunny", "cold and rainy", "mild and partly cloudy", "hot and humid", "cold and snowy"]
    zones = ["tropical", "temperate", "arid", "continental", "polar"]
    models = ["ECMWF HRES", "GFS ensemble", "NAM regional", "ICON-EU"]

    for i in range(50):
        city = CITIES[i % len(CITIES)]
        template = WEATHER_TEMPLATES[i % len(WEATHER_TEMPLATES)]
        text = template.format(
            city=city,
            station_id=f"WS-{rng.randint(1000,9999)}",
            elev=rng.randint(0, 2500),
            avg_temp=rng.randint(5, 30),
            rain=rng.randint(200, 2000),
            zone=rng.choice(zones),
            year=rng.randint(1950, 2010),
            season=rng.choice(seasons),
            condition=rng.choice(conditions),
            low=rng.randint(-10, 15),
            high=rng.randint(20, 45),
            humidity=rng.randint(30, 90),
            wind=rng.randint(5, 40),
            n_years=rng.randint(10, 50),
            trend=rng.choice(["warming", "cooling", "stable"]),
            events=rng.randint(1, 15),
            max_temp=rng.randint(35, 50),
            hot_date=f"20{rng.randint(10,23):02d}-{rng.randint(6,8):02d}-{rng.randint(1,28):02d}",
            limit=rng.choice([60, 100, 200]),
            n_day=rng.choice([3, 5, 7, 10, 14]),
            model=rng.choice(models),
            accuracy=rng.randint(80, 98),
            decay=rng.randint(3, 8),
        )
        docs.append((text, {"domain": "weather", "doc_id": f"weather_{i:03d}", "city": city}))
    return docs


def _generate_calendar_docs(rng: random.Random) -> list[tuple[str, dict]]:
    """Generate 50 synthetic calendar documents."""
    docs = []
    priorities = ["first-come-first-served", "manager priority", "manual resolution"]

    for i in range(50):
        template = CALENDAR_TEMPLATES[i % len(CALENDAR_TEMPLATES)]
        text = template.format(
            dur=rng.choice([15, 30, 45, 60]),
            buffer=rng.choice([5, 10, 15]),
            max_att=rng.choice([20, 50, 100, 200]),
            mod_hours=rng.choice([1, 2, 24]),
            n_cals=rng.choice([5, 10, 20]),
            n_colors=rng.choice([8, 12, 16]),
            priority=rng.choice(priorities),
            max_events=rng.choice([20, 50, 100]),
        )
        docs.append((text, {"domain": "calendar", "doc_id": f"calendar_{i:03d}"}))
    return docs


def _generate_medical_docs(rng: random.Random) -> list[tuple[str, dict]]:
    """Generate 50 synthetic medical documents."""
    docs = []
    conditions = ["hypertension", "diabetes type 2", "asthma", "migraine", "arthritis",
                   "anxiety disorder", "high cholesterol", "GERD", "hypothyroidism", "osteoporosis"]
    drugs = ["metformin", "lisinopril", "atorvastatin", "albuterol", "sertraline",
             "omeprazole", "levothyroxine", "ibuprofen", "amlodipine", "metoprolol"]
    drug_classes = ["biguanide", "ACE inhibitor", "statin", "bronchodilator", "SSRI",
                     "PPI", "thyroid hormone", "NSAID", "calcium channel blocker", "beta blocker"]
    types = ["chronic", "acute", "autoimmune", "metabolic", "inflammatory"]
    levels = ["1 (basic)", "2 (clinical)", "3 (full access)"]

    for i in range(50):
        template = MEDICAL_TEMPLATES[i % len(MEDICAL_TEMPLATES)]
        cond = conditions[i % len(conditions)]
        drug = drugs[i % len(drugs)]
        text = template.format(
            condition=cond,
            type=rng.choice(types),
            prevalence=rng.randint(1, 25),
            meds=", ".join(rng.sample(drugs, 3)),
            markers=rng.choice(["blood pressure", "blood glucose", "cholesterol", "lung function", "liver enzymes"]),
            interval=rng.choice([3, 6, 12]),
            n_interactions=rng.randint(5000, 20000),
            level=rng.choice(levels),
            drug=drug,
            dose=f"{rng.choice([5, 10, 20, 25, 50, 100])}mg {rng.choice(['daily', 'twice daily', 'as needed'])}",
            contra=", ".join(rng.sample(["pregnancy", "liver disease", "kidney failure", "heart failure", "allergy to drug"], 2)),
            sides=", ".join(rng.sample(["nausea", "headache", "dizziness", "fatigue", "dry mouth", "insomnia"], 3)),
            drug_class=drug_classes[i % len(drug_classes)],
            monitoring=rng.choice(["creatinine", "liver enzymes", "blood glucose", "electrolytes"]),
            freq=rng.choice([4, 8, 12, 26]),
            type_desc=rng.choice([
                "Checkups are routine wellness visits",
                "Follow-ups review treatment progress",
                "Specialist referrals require primary care authorization",
                "Emergency appointments bypass scheduling rules",
            ]),
            dur=rng.choice([15, 20, 30, 45, 60]),
        )
        docs.append((text, {"domain": "medical", "doc_id": f"medical_{i:03d}"}))
    return docs


# ── RAG Retriever ─────────────────────────────────────────────────────────────

class RAGRetriever:
    """ChromaDB-backed retriever with 3 domain collections."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, reset: bool = False, tracer=None):
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False,  # in-memory for speed; set True + persist_directory for disk
        ))
        self.collections: dict[str, chromadb.Collection] = {}
        self._initialized = False
        self.tracer = tracer  # Optional TraceLogger instance
        logger.info("RAGRetriever created (persist_dir=%s)", persist_dir)

    def initialize(self) -> None:
        """Create collections and load synthetic documents."""
        if self._initialized:
            return

        rng = random.Random(RANDOM_SEED)

        domain_generators = {
            "weather_docs": _generate_weather_docs,
            "calendar_docs": _generate_calendar_docs,
            "medical_docs": _generate_medical_docs,
        }

        for collection_name, generator in domain_generators.items():
            # Delete if exists to avoid duplicates
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            docs = generator(rng)
            texts = [d[0] for d in docs]
            metadatas = [d[1] for d in docs]
            ids = [m["doc_id"] for m in metadatas]

            collection.add(documents=texts, metadatas=metadatas, ids=ids)
            self.collections[collection_name] = collection
            logger.info("Loaded %d docs into %s", len(docs), collection_name)

        self._initialized = True
        logger.info("RAGRetriever initialized with 3 collections (150 total docs)")

    def retrieve(
        self, query: str, domain: str, top_k: int = RAG_TOP_K,
        session_id: str = None, turn_id: int = 0,
    ) -> list[dict]:
        """Retrieve top_k documents from the domain collection.

        Returns list of {text, metadata, score}.
        """
        if not self._initialized:
            self.initialize()

        collection_name = f"{domain}_docs"
        if collection_name not in self.collections:
            logger.warning("Unknown domain for retrieval: %s", domain)
            return []

        collection = self.collections[collection_name]

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, 50),
            )
        except Exception as e:
            logger.error("Retrieval error: %s", str(e))
            return []

        retrieved = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for text, meta, dist in zip(docs, metas, dists):
                # ChromaDB returns distances; convert to similarity score
                score = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)
                retrieved.append({"text": text, "metadata": meta, "score": round(score, 4)})

        logger.info(
            "Retrieved %d docs for query='%s' domain=%s",
            len(retrieved), query[:50], domain,
        )

        # Trace retrieval if tracer is configured
        if self.tracer and session_id:
            scores = [r["score"] for r in retrieved]
            self.tracer.log_retrieval(session_id, turn_id, query, retrieved, scores)

        return retrieved
