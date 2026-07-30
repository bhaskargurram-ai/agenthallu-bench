"""Citation tracker + 'should-cite' opportunity finder for AgentProp-Bench.

Uses the free OpenAlex API (no key; be polite with a mailto) and, best-effort,
Semantic Scholar. Three commands:
  track      -> append today's cited_by_count to analysis/citation_log.csv
  citers     -> list works currently citing the paper (title, year, venue)
  candidates -> recent (2025-26) on-topic works NOT yet citing it (outreach list)

Usage:
  ./venv/bin/python analysis/citation_tracker.py track
  ./venv/bin/python analysis/citation_tracker.py citers
  ./venv/bin/python analysis/citation_tracker.py candidates
"""
import sys, os, csv, json, time, datetime, urllib.request, urllib.parse

MAILTO = os.getenv("OPENALEX_MAILTO", "research@example.com")  # set yours to raise rate limits
ARXIV_ID = "2604.16706"
OA_WORK_IDS = ["W7155074625", "W7155247005"]  # both indexed records (dedup counts)
LOG = os.path.join(os.path.dirname(__file__), "citation_log.csv")
TOPICS = ["LLM agent evaluation benchmark", "LLM-as-a-judge reliability",
          "tool-use hallucination function calling", "agent error propagation"]


def _get(url):
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}mailto={MAILTO}",
                                 headers={"User-Agent": f"agentprop-cite/1.0 (mailto:{MAILTO})"})
    return json.load(urllib.request.urlopen(req, timeout=40))


def _oa_count():
    total_citers = set()
    for wid in OA_WORK_IDS:
        try:
            w = _get(f"https://api.openalex.org/works/{wid}")
            for c in _get(w["cited_by_api_url"] + "&per-page=200").get("results", []):
                total_citers.add(c["id"])
        except Exception as e:
            print(f"  (openalex {wid}: {e})")
        time.sleep(1)
    return total_citers


def track():
    citers = _oa_count()
    row = [datetime.date.today().isoformat(), "openalex", len(citers)]
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "source", "cited_by_count"])
        w.writerow(row)
    print(f"logged {row} -> {LOG}")
    print("NOTE: OpenAlex lags; check Google Scholar for the fastest count "
          "(set up a Scholar alert on the paper).")


def citers():
    seen = _oa_count()
    print(f"OpenAlex citers (deduped across records): {len(seen)}")
    for wid in OA_WORK_IDS:
        try:
            w = _get(f"https://api.openalex.org/works/{wid}")
            for c in _get(w["cited_by_api_url"] + "&per-page=50").get("results", []):
                host = (c.get("primary_location") or {}).get("source") or {}
                print(f"  - {c.get('publication_year')} | {host.get('display_name','?')} | {c.get('title','')[:70]}")
        except Exception as e:
            print(f"  ({wid}: {e})")
        time.sleep(1)


def candidates():
    print("Recent (2025-2026) on-topic works = outreach candidates (verify they don't already cite you):")
    for t in TOPICS:
        try:
            q = urllib.parse.quote(t)
            r = _get(f"https://api.openalex.org/works?search={q}"
                     f"&filter=from_publication_date:2025-01-01,type:article"
                     f"&sort=cited_by_count:desc&per-page=8")
            print(f"\n## {t}")
            for w in r.get("results", []):
                oaid = w["id"].split("/")[-1]
                print(f"  [{w.get('cited_by_count')}c] {w.get('publication_year')} | "
                      f"{w.get('title','')[:66]} | {oaid}")
        except Exception as e:
            print(f"  ({t}: {e})")
        time.sleep(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "track"
    {"track": track, "citers": citers, "candidates": candidates}.get(cmd, track)()
