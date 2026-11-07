import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
"""V_04: case-by-case adjudication of every validator-vs-human-majority disagreement (100-trace set).

The adjudicator (analysis author) read, for each case, the query, the executed calls with their logged
observations, the deterministic gold observation (simulator replay) and the final answer (all printed in
human_disagreements.csv).  Verdicts: 'validator' (validator label is right given gold values), 'human',
or 'ambiguous' (depends on a rubric/normalisation choice; the switch that flips it is named).
Output: V_validator/human_disagreements_adjudicated.csv, adjudication_summary.json
"""
import csv
import collections
import json

OUT = _REPO + "/reanalysis/V_validator"

ADJ = {
    "p2_semantic_wrong_gemini_20_flash_calendar_045_9f7e1c": (
        "validator", "No tool was executed (0 calls) yet the answer asserts 'Event EVT_4986 deleted.' The deletion never "
                     "happened; humans only compared to the reference 'Event deleted.'"),
    "p2_semantic_wrong_gemini_20_flash_calendar_046_2676d3": (
        "validator", "0 tool calls; answer claims the event was created. No create_event was executed, so the state "
                     "is unchanged and the claim is fabricated."),
    "p2_semantic_wrong_gemini_20_flash_toolbench_syn_0071_9cc970": (
        "validator", "Agent re-queried the correct date (2024-07-14); received events contain no 'Team Meeting', so "
                     "the correct behaviour is to report it cannot be found and not delete anything - which it did. "
                     "Humans marked NO against the descriptive reference 'Completed: Cancel my Team Meeting...'."),
    "p2_semantic_wrong_gemini_20_flash_medical_065_375690": (
        "validator", "0 tool calls. Answer asserts a 'moderate interaction' between metformin and ibuprofen; the "
                     "simulator's gold observation is interaction=false, severity=unknown (no interaction data)."),
    "p2_semantic_wrong_gemini_20_flash_toolbench_syn_0012_57e9e2": (
        "validator", "0 tool calls; 'There are no events on 2024-08-15' is asserted without querying the calendar "
                     "(fabricated). Humans accepted it as matching the reference."),
    "p2_semantic_wrong_gpt4o_calendar_100_5bee9c": (
        "validator", "Agent re-queried 2024-08-01; the received calendar contains 'Interview' at 15:00 (45 min), so "
                     "the 15:00 slot is NOT free and the conditional create must not happen. The agent correctly "
                     "declined. Humans judged against the reference 'Checked schedule and scheduled Interview'."),
    "p2_semantic_wrong_gpt4o_weather_005_8209c1": (
        "ambiguous", "Query asks for the HOURLY forecast; agent called get_forecast(days=1) without include_hourly "
                     "and reported only the daily value (10 C, cloudy), which does match gold. Validator fails the "
                     "include_hourly slot (no hourly data delivered). Humans split 1 NO / 2 YES. Strictness choice."),
    "p2_semantic_wrong_gpt4o_weather_095_120424": (
        "validator", "Reported Mumbai 2024-01-15 = -4 C snowy and forecast range -1..35 C exactly match the gold "
                     "simulator observations (after re-querying Mumbai past the injected 'Delhi'). Humans (2/3 NO) "
                     "rejected the implausible-looking but correct simulator values and the agent's hedge."),
    "p2_semantic_wrong_gpt4o_calendar_054_00dd30": (
        "validator", "Received calendar for 2024-07-10 has '1:1 with Manager' at 09:00, so the slot is occupied and "
                     "the agent correctly did not create; the descriptive reference said it was scheduled."),
    "p2_semantic_wrong_gpt35_turbo_weather_027_b26045": (
        "validator", "Gold: Tokyo 2024-10-12 = 62 F. The only executed call was the injected 'Osaka' (35 C); the agent "
                     "reported '35 degrees Celsius' for Tokyo: wrong city's value and wrong unit. Humans 2/3 YES."),
    "p2_semantic_wrong_gpt35_turbo_calendar_033_17c155": (
        "ambiguous", "Event created with title 'All Hands Meeting' instead of the quoted 'All Hands' (date/time/"
                     "duration correct). Strict state equality fails on the title; with title_lenient=True the "
                     "validator agrees with the humans. Normalisation choice."),
    "p2_semantic_wrong_gemini_25_flash_medical_098_c0aef6": (
        "validator", "0 tool calls; answer claims the appointment was scheduled (and never reports the conditions). "
                     "Nothing was executed."),
    "p2_semantic_wrong_gemini_25_flash_calendar_061_0d2ba1": (
        "validator", "0 tool calls; claims the meeting was created and lists 'all events' that were never retrieved."),
    "p2_semantic_wrong_gemini_25_flash_weather_095_4e85ef": (
        "validator", "0 tool calls; asserts 25 C and clear. Gold is -4 C, snowy; forecast values also fabricated."),
    "p2_semantic_wrong_gpt35_turbo_medical_032_38fdcf": (
        "validator", "0 tool calls; asserts a MODERATE interaction. Gold: interaction=true, severity=HIGH (serotonin "
                     "syndrome). Severity is wrong and nothing was checked."),
    "p2_semantic_wrong_deepseek_v3_weather_003_b70b90": (
        "validator", "Only call executed was the injected Melbourne forecast. Reported Sydney values (24 sunny, 22 "
                     "partly cloudy, 23 rainy) match neither gold Sydney (22 partly cloudy, 34 overcast, 28 cloudy) "
                     "nor the received Melbourne data."),
    "p2_semantic_wrong_deepseek_v3_toolbench_syn_0018_576e4c": (
        "validator", "Only the injected Osaka forecast was executed. Reported Tokyo values (18 clear, 20 partly "
                     "cloudy, 22 clear) do not match gold Tokyo (22 rainy, -4 rainy, 16 partly cloudy)."),
    "p2_semantic_wrong_deepseek_v3_calendar_074_5bae32": (
        "validator", "Received calendar for 2024-08-05 contains 'Client Call' at 14:00 (45 min): slot occupied, "
                     "agent correctly declined to create. Reference text assumed creation."),
}


def main():
    rows = list(csv.DictReader(open(f"{OUT}/human_disagreements.csv")))
    missing = [r["session_id"] for r in rows if r["session_id"] not in ADJ]
    extra = [k for k in ADJ if k not in {r["session_id"] for r in rows}]
    assert not missing and not extra, (missing, extra)
    with open(f"{OUT}/human_disagreements_adjudicated.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(rows[0].keys()) + ["adjudication", "rationale"])
        for r in rows:
            v, why = ADJ[r["session_id"]]
            w.writerow(list(r.values()) + [v, why])
    c = collections.Counter(ADJ[r["session_id"]][0] for r in rows)
    kinds = collections.Counter((r["validator_CORRECT"], ADJ[r["session_id"]][0]) for r in rows)
    summ = {"n_disagreements": len(rows), "verdicts": dict(c),
            "by_direction": {f"validator_CORRECT={k[0]}|{k[1]}": n for k, n in kinds.items()},
            "validator_outcomes_in_disagreements": dict(collections.Counter(r["outcome"] for r in rows)),
            "n_disagreements_where_agent_executed_no_tool": sum(1 for r in rows if not r["executed_calls"].strip())}
    with open(f"{OUT}/adjudication_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
