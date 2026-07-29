# Case Studies (Appendix)

## High-EPS Surprising (Frontier Model Fully Propagates Error) — Example 1

- **Session ID**: `p2_type_mismatch_gpt4o_calendar_100_43c932`
- **Model**: gpt4o
- **Domain**: calendar
- **Injection type**: p2_type_mismatch
- **Ground truth**: Checked schedule and scheduled 'Interview' on 2024-08-01.
- **Final answer**: The 'Interview' event has been successfully created for 2024-08-01 at 15:00 for 45 minutes.
- **Final correct**: 0
- **Injection target_stage**: parameter_generation
- **Injection detail**: 

### Trace steps
```
[thought] Thought: I need to check the events scheduled for 2024-08-01 to see if the 15:00 slot is free. Action: get_events Action Input: {"date": "2024-08-01"}
[action] get_events: Action: get_events
[observation] {"error": "Parameter 'date' must be string, got int"}
[thought] Thought: It seems there was an error with the input format. I need to ensure the date is provided as a string. Action: get_events Action Input: {"date": "2024-08-01"}
[action] get_events: Action: get_events
[observation] {"date": "2024-08-01", "calendar_id": "primary", "events": [{"title": "1:1 Meeting", "time": "16:00", "duration": 90, "event_id": "EVT_MOCK_2395"}, {"title": "Team Standup", "time": "11:00", "duration
[thought] Thought: The 15:00 slot is free since there are no events scheduled at that time. I can proceed to create an 'Interview' event for 45 minutes starting at 15:00. Action: create_event Action Input: {"da
[action] create_event: Action: create_event
[observation] {"error": "Missing required parameter: 'duration_minutes'; Unknown parameter: 'duration'"}
[thought] Thought: I need to correct the parameter name from 'duration' to 'duration_minutes' to successfully create the event. Action: create_event Action Input: {"date": "2024-08-01", "time": "15:00", "durati
[action] create_event: Action: create_event
[observation] {"event_id": "EVT_0002", "status": "created"}
```

## High-EPS Surprising (Frontier Model Fully Propagates Error) — Example 2

- **Session ID**: `p2_type_mismatch_gpt4o_medical_098_724f8b`
- **Model**: gpt4o
- **Domain**: medical
- **Injection type**: p2_type_mismatch
- **Ground truth**: Retrieved conditions and scheduled appointment for P004.
- **Final answer**: The emergency appointment for patient P004 (David Lee) with doctor D002 has been successfully scheduled for 2024-11-01. The appointment ID is APPT_27560.
- **Final correct**: 0
- **Injection target_stage**: parameter_generation
- **Injection detail**: 

### Trace steps
```
[thought] Thought: I need to first retrieve the medical record of patient P004 to understand their conditions before scheduling an emergency appointment. Action: get_patient_record Action Input: {"patient_id": 
[action] get_patient_record: Action: get_patient_record
[observation] {"error": "Parameter 'patient_id' must be string, got int"}
[thought] Thought: It seems there was an error with the input type. I need to ensure the patient ID is a string. Action: get_patient_record Action Input: {"patient_id": "P004"}
[action] get_patient_record: Action: get_patient_record
[observation] {"name": "David Lee", "dob": "1968-01-30", "conditions": ["arthritis", "high cholesterol"], "medications": ["ibuprofen", "atorvastatin"], "patient_id": "P004"}
[thought] Thought: Now that I have the patient's conditions, I can proceed to schedule an emergency appointment with doctor D002 on the specified date. Action: schedule_appointment Action Input: {"doctor_id": "
[action] schedule_appointment: Action: schedule_appointment
[observation] {"error": "Missing required parameter: 'appointment_type'; Unknown parameter: 'type'"}
[thought] Thought: I need to correct the parameter name to 'appointment_type' instead of 'type' for scheduling the appointment. Action: schedule_appointment Action Input: {"doctor_id": "D002", "patient_id": "P0
[action] schedule_appointment: Action: schedule_appointment
[observation] {"appointment_id": "APPT_27560", "patient_id": "P004", "doctor_id": "D002", "date": "2024-11-01", "appointment_type": "emergency", "status": "scheduled"}
```

## Self-Correction (Agent Recovers Despite Bad Parameter) — Example 1

- **Session ID**: `p2_type_mismatch_gpt4o_mini_weather_097_a05e74`
- **Model**: gpt4o_mini
- **Domain**: weather
- **Injection type**: p2_type_mismatch
- **Ground truth**: Current vs historical weather for Toronto.
- **Final answer**: Today's weather in Toronto (34°C, cloudy) is significantly warmer than the historical average temperature from January 1 to January 7, 2023, which is approximately 12.14°C.
- **Final correct**: 1
- **Injection target_stage**: parameter_generation
- **Injection detail**: 

### Trace steps
```
[thought] Thought: I need to first get today's weather in Toronto for the date 2024-07-08, and then retrieve the historical weather data for Toronto from January 1, 2023, to January 7, 2023, to compare the two.
[action] get_weather: Action: get_weather
[observation] {"error": "Missing required parameter: 'city'; Missing required parameter: 'date'; Unknown parameter: 'param'"}
[thought] Thought: It seems I used the wrong parameter format for the tools. I need to specify the city and date correctly for both the current weather and historical weather data requests.  Action: get_weather
[action] get_weather: Action: get_weather
[observation] {"city": "Toronto", "date": "2024-07-08", "temperature": 34, "unit": "celsius", "humidity": 27, "condition": "cloudy"}
[thought] Thought: I have obtained today's weather in Toronto for July 8, 2024, which is 34°C with 27% humidity and cloudy conditions. Now, I need to retrieve the historical weather data for Toronto from Januar
[action] get_historical: Action: get_historical
[observation] {"city": "Toronto", "period": "2023-01-01 to 2023-01-07", "data": [{"date": "2023-01-01", "temperature": -6}, {"date": "2023-01-02", "temperature": 12}, {"date": "2023-01-03", "temperature": -6}, {"da
[thought] Thought: I have the historical weather data for Toronto from January 1 to January 7, 2023. The temperatures for those days are as follows: - January 1: -6°C - January 2: 12°C - January 3: -6°C - Janua
[final] Today's weather in Toronto (34°C, cloudy) is significantly warmer than the historical average temperature from January 1 to January 7, 2023, which is approximately 12.14°C.
```

## Self-Correction (Agent Recovers Despite Bad Parameter) — Example 2

- **Session ID**: `p2_type_mismatch_gpt4o_mini_calendar_081_f64e4a`
- **Model**: gpt4o_mini
- **Domain**: calendar
- **Injection type**: p2_type_mismatch
- **Ground truth**: Created 'Training Session' and listed all events on 2024-07-02.
- **Final answer**: The 'Training Session' meeting has been successfully created on July 2, 2024, at 16:00 for 30 minutes. The events for that day include the 'Training Session' from 16:00 to 16:30.
- **Final correct**: 1
- **Injection target_stage**: parameter_generation
- **Injection detail**: 

### Trace steps
```
[thought] Thought: I need to create a calendar event for a 'Training Session' on July 2, 2024, at 16:00 for 30 minutes with the specified participants. After creating the event, I will retrieve all events for t
[action] create_event: Action: create_event
[observation] {"error": "Missing required parameter: 'duration_minutes'; Parameter 'title' must be string, got int; Unknown parameter: 'duration'; Unknown parameter: 'participants'"}
[thought] Thought: It seems I made an error in the parameters for creating the event. I need to ensure I use the correct parameters, specifically using 'duration_minutes' instead of 'duration' and 'attendees' i
[action] create_event: Action: create_event
[observation] {"event_id": "EVT_0002", "status": "created"}
[thought] Thought: The 'Training Session' meeting has been successfully created. Now, I will retrieve all events for July 2, 2024, to confirm the creation and see any other scheduled events.  Action: get_events
[action] get_events: Action: get_events
[observation] {"date": "2024-07-02", "calendar_id": "primary", "events": [{"title": "Training Session", "time": "16:00", "duration": 30, "event_id": "EVT_0002"}]}
[thought] Final Answer: The 'Training Session' meeting has been successfully created on July 2, 2024, at 16:00 for 30 minutes. The events for that day include the 'Training Session' from 16:00 to 16:30.
[final] The 'Training Session' meeting has been successfully created on July 2, 2024, at 16:00 for 30 minutes. The events for that day include the 'Training Session' from 16:00 to 16:30.
```

## Cross-Agent Amplification (Error Propagates Through Multi-Agent Pipeline) — Example 1

- **Session ID**: `p4_wrong_subtask_gpt4o_mini_calendar_100_4f2ad5`
- **Model**: gpt4o_mini
- **Domain**: calendar
- **Injection type**: p4_wrong_subtask
- **Ground truth**: Checked schedule and scheduled 'Interview' on 2024-08-01.
- **Final answer**: The 45-minute 'Interview' event has been successfully created for 2024-08-01 at 15:00, as the slot was free.
- **Final correct**: 0
- **Injection target_stage**: planning
- **Injection detail**: 

### Trace steps
```
[thought] Agent1 (Planner): Plan: 1. Use tool get_events with params: {"date": "2024-08-01"} 2. Check if the 15:00 slot is free based on the retrieved events. 3. If the slot is free, use tool create_event with 
[thought] Agent2 (Executor): Thought: I need to retrieve all events for the date 2024-08-01 to check if the 15:00 slot is free. Action: get_events Action Input: {"date": "2024-08-01"}
[action] get_events: Action: get_events
[observation] {"date": "2024-08-01", "calendar_id": "primary", "events": [{"title": "1:1 Meeting", "time": "16:00", "duration": 90, "event_id": "EVT_MOCK_2395"}, {"title": "Team Standup", "time": "11:00", "duration
[thought] Agent2 (Executor): Thought: The retrieved events for 2024-08-01 show a "1:1 Meeting" at 16:00 and a "Team Standup" at 11:00. There are no events scheduled at 15:00, so the slot is free. Action: create
[action] create_event: Action: create_event
[observation] {"error": "Missing required parameter: 'duration_minutes'; Unknown parameter: 'duration'"}
[thought] Agent2 (Executor): Thought: The error indicates that the parameter for duration should be specified as 'duration_minutes' instead of 'duration'. I will correct this and try to create the event again. 
[action] create_event: Action: create_event
[observation] {"event_id": "EVT_0002", "status": "created"}
[thought] Agent2 (Executor): Result: The 45-minute 'Interview' event has been successfully created for 2024-08-01 at 15:00.
[final] Agent3 (Synthesizer): Final Answer: The 45-minute 'Interview' event has been successfully created for 2024-08-01 at 15:00, as the slot was free.
```

## Cross-Agent Amplification (Error Propagates Through Multi-Agent Pipeline) — Example 2

- **Session ID**: `p4_wrong_subtask_gpt4o_mini_toolbench_syn_0058_68519a`
- **Model**: gpt4o_mini
- **Domain**: weather
- **Injection type**: p4_wrong_subtask
- **Ground truth**: Completed: Is it going to rain in Berlin tomorrow?
- **Final answer**: No, it is not going to rain in Berlin tomorrow; the forecast indicates cloudy weather with a temperature of 22°C.
- **Final correct**: 0
- **Injection target_stage**: planning
- **Injection detail**: 

### Trace steps
```
[thought] Agent1 (Planner): Plan: 1. Use tool get_forecast with params: {"city": "Berlin", "days": 1} Goal: Determine if it will rain in Berlin tomorrow.
[thought] Agent2 (Executor): Thought: I need to get the weather forecast for Berlin for tomorrow to determine if it will rain.  Action: get_forecast Action Input: {"city": "Berlin", "days": 1}
[action] get_forecast: Action: get_forecast
[observation] {"city": "Berlin", "days": 1, "forecast": [{"date": "2026-04-16", "temperature": 22, "condition": "cloudy"}]}
[thought] Agent2 (Executor): Result: The weather forecast for Berlin tomorrow (April 16, 2026) indicates a temperature of 22°C and a condition of "cloudy." There is no indication of rain.
[final] Agent3 (Synthesizer): Final Answer: No, it is not going to rain in Berlin tomorrow; the forecast indicates cloudy weather with a temperature of 22°C.
```

## Interceptor Caught a Hallucination — Example 1

- **Session ID**: `int_output_only_gpt4o_mini_weather_097_e5f304`
- **Model**: gpt4o_mini
- **Domain**: weather
- **Injection type**: interceptor_output_only
- **Ground truth**: Current vs historical weather for Toronto.
- **Final answer**: I cannot reliably answer this question.
- **Final correct**: 0

### Trace steps
```
[thought] Thought: I need to first get today's weather in Toronto for the date 2024-07-08, and then retrieve the historical weather data for Toronto from January 1, 2023, to January 7, 2023, to compare the two.
[action] get_weather: Action: get_weather
[observation] {"error": "Missing required parameter: 'city'; Missing required parameter: 'date'; Unknown parameter: 'param'"}
[thought] Thought: It seems I used the wrong parameter format for the weather tool. I need to specify the city and date correctly.  Action: get_weather Action Input: {"city": "Toronto", "date": "2024-07-08"}
[action] get_weather: Action: get_weather
[observation] {"city": "Toronto", "date": "2024-07-08", "temperature": 34, "unit": "celsius", "humidity": 27, "condition": "cloudy"}
[thought] Thought: I have obtained today's weather in Toronto for July 8, 2024, which is 34°C with 27% humidity and cloudy conditions. Now, I will retrieve the historical weather data for Toronto from January 1
[action] get_historical: Action: get_historical
[observation] {"city": "Toronto", "period": "2023-01-01 to 2023-01-07", "data": [{"date": "2023-01-01", "temperature": -6}, {"date": "2023-01-02", "temperature": 12}, {"date": "2023-01-03", "temperature": -6}, {"da
[thought] Thought: I have retrieved the historical weather data for Toronto from January 1 to January 7, 2023. The temperatures for those days are as follows: - January 1: -6°C - January 2: 12°C - January 3: -6
[action] Calculate: Action: Calculate
[observation] {"error": "Unknown tool: 'Calculate'"}
```

## Interceptor Caught a Hallucination — Example 2

- **Session ID**: `int_output_only_gpt4o_mini_medical_098_fed83d`
- **Model**: gpt4o_mini
- **Domain**: medical
- **Injection type**: interceptor_output_only
- **Ground truth**: Retrieved conditions and scheduled appointment for P004.
- **Final answer**: I cannot reliably answer this question.
- **Final correct**: 0

### Trace steps
```
[thought] Thought: I need to first retrieve the medical record of patient P004 to check their conditions before scheduling an emergency appointment with doctor D002.  Action: get_patient_record Action Input: {"
[action] get_patient_record: Action: get_patient_record
[observation] {"error": "Missing required parameter: 'patient_id'; Unknown parameter: 'param'"}
[thought] Thought: It seems I used the wrong parameter name when trying to retrieve the patient record. I will correct it and try again using the correct parameter name for the patient ID.  Action: get_patient_
[action] get_patient_record: Action: get_patient_record
[observation] {"name": "David Lee", "dob": "1968-01-30", "conditions": ["arthritis", "high cholesterol"], "medications": ["ibuprofen", "atorvastatin"], "patient_id": "P004"}
[thought] Thought: I have successfully retrieved the medical record for patient P004, who has conditions of arthritis and high cholesterol. Now, I will proceed to schedule an emergency appointment with doctor D
[action] schedule_appointment: Action: schedule_appointment
[observation] {"error": "Missing required parameter: 'appointment_type'"}
[thought] Thought: It appears that I need to specify the type of appointment when scheduling. I will include the appointment type as "emergency" and try scheduling the appointment again.  Action: schedule_appoi
[action] schedule_appointment: Action: schedule_appointment
[observation] {"appointment_id": "APPT_27560", "patient_id": "P004", "doctor_id": "D002", "date": "2024-11-01", "appointment_type": "emergency", "status": "scheduled"}
```

## Interceptor False Positive — Example 1

- **Session ID**: `int_output_only_gpt4o_mini_weather_097_e5f304`
- **Model**: gpt4o_mini
- **Domain**: weather
- **Injection type**: interceptor_output_only
- **Ground truth**: Current vs historical weather for Toronto.
- **Final answer**: I cannot reliably answer this question.
- **Final correct**: 0

### Trace steps
```
[thought] Thought: I need to first get today's weather in Toronto for the date 2024-07-08, and then retrieve the historical weather data for Toronto from January 1, 2023, to January 7, 2023, to compare the two.
[action] get_weather: Action: get_weather
[observation] {"error": "Missing required parameter: 'city'; Missing required parameter: 'date'; Unknown parameter: 'param'"}
[thought] Thought: It seems I used the wrong parameter format for the weather tool. I need to specify the city and date correctly.  Action: get_weather Action Input: {"city": "Toronto", "date": "2024-07-08"}
[action] get_weather: Action: get_weather
[observation] {"city": "Toronto", "date": "2024-07-08", "temperature": 34, "unit": "celsius", "humidity": 27, "condition": "cloudy"}
[thought] Thought: I have obtained today's weather in Toronto for July 8, 2024, which is 34°C with 27% humidity and cloudy conditions. Now, I will retrieve the historical weather data for Toronto from January 1
[action] get_historical: Action: get_historical
[observation] {"city": "Toronto", "period": "2023-01-01 to 2023-01-07", "data": [{"date": "2023-01-01", "temperature": -6}, {"date": "2023-01-02", "temperature": 12}, {"date": "2023-01-03", "temperature": -6}, {"da
[thought] Thought: I have retrieved the historical weather data for Toronto from January 1 to January 7, 2023. The temperatures for those days are as follows: - January 1: -6°C - January 2: 12°C - January 3: -6
[action] Calculate: Action: Calculate
[observation] {"error": "Unknown tool: 'Calculate'"}
```
