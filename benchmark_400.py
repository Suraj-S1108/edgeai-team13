
"""
Mission 3: Compare 400 individual predictions against one batch of 400.

Usage:
python benchmark_400.py http://localhost:8000

Or from the laptop:
python benchmark_400.py https://YOUR-CODESPACE-ADDRESS
"""

import json
import random
import sys
import time
import urllib.request


if len(sys.argv) < 2:
    print("Usage: python benchmark_400.py API_BASE_URL")
    sys.exit(1)

BASE_URL = sys.argv[1].rstrip("/")
random.seed(42)


def send_request(endpoint, payload):
    url = BASE_URL + endpoint
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


# Generate 400 reproducible, valid sensor readings.
readings = []

for _ in range(400):
    readings.append({
        "battery_v": round(random.uniform(3.2, 4.2), 2),
        "temp_c": round(random.uniform(10, 40), 2),
        "rings_per_day": random.randint(0, 30),
        "wifi_rssi_dbm": random.randint(-90, -30),
        "age_months": random.randint(0, 36)
    })


print("\n========== 400 PREDICTION COMPARISON ==========")
print("Readings generated:", len(readings))

# Check API availability.
health_request = urllib.request.urlopen(
    BASE_URL + "/health", timeout=30
)
print("API health:", health_request.status)
health_request.close()


# 1. Measure 400 individual HTTP requests.
print("\nRunning 400 individual /predict requests...")

individual_predictions = []
individual_start = time.perf_counter()

for index, reading in enumerate(readings, start=1):
    response = send_request("/predict", reading)
    individual_predictions.append(response["days_left"])

    if index % 50 == 0:
        print(f"Completed {index}/400 individual requests")


individual_time_ms = (
    time.perf_counter() - individual_start
) * 1000


# 2. Measure one batch request containing all 400 readings.
print("\nRunning one /predict_batch request with 400 readings...")

batch_start = time.perf_counter()

batch_response = send_request(
    "/predict_batch",
    {"rows": readings}
)

batch_time_ms = (
    time.perf_counter() - batch_start
) * 1000

batch_predictions = batch_response["days_left"]


# 3. Compare predictions.
if len(batch_predictions) != 400:
    print("ERROR: Expected 400 batch predictions.")
    print("Received:", len(batch_predictions))
    sys.exit(1)

differences = [
    abs(individual - batch)
    for individual, batch in zip(
        individual_predictions, batch_predictions
    )
]

max_difference = max(differences)

# 4. Calculate speedup.
speedup = individual_time_ms / batch_time_ms

print("\n========== RESULTS ==========")
print("Individual requests:", len(individual_predictions))
print("Batch predictions:", len(batch_predictions))

print(
    "400 individual requests total time:",
    round(individual_time_ms, 3), "ms"
)

print(
    "One batch request total time:",
    round(batch_time_ms, 3), "ms"
)

print("Batch speedup:", round(speedup, 2), "x")
print(
    "Maximum prediction difference:",
    round(max_difference, 6), "days"
)

if max_difference <= 0.01:
    print("SUCCESS: Batch and individual predictions match.")
else:
    print("WARNING: Batch and individual predictions differ.")

print("\n========== TEST COMPLETED ==========")
