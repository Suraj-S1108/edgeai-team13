
import json
import random
import sys
import time
import urllib.request
import urllib.error

# Use the public Codespace port 8000 URL.
BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else ""

if not BASE_URL:
    print("Usage: python test_batch.py <API_BASE_URL>")
    sys.exit(1)

random.seed(42)


def send_request(endpoint, payload):
    """Send a JSON POST request and return the response."""
    url = BASE_URL + endpoint
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# Generate 50 valid and reproducible sensor readings.
readings = []

for _ in range(50):
    readings.append({
        "battery_v": round(random.uniform(3.2, 4.2), 2),
        "temp_c": round(random.uniform(10, 40), 2),
        "rings_per_day": random.randint(0, 30),
        "wifi_rssi_dbm": random.randint(-90, -30),
        "age_months": random.randint(0, 36)
    })


print("\n--- 50-READING BATCH TEST ---")
print("Generated readings:", len(readings))

# 1. Send all 50 readings in one batch request.
print("\nSending batch request...")

batch_response = send_request(
    "/predict_batch",
    {"rows": readings}
)

batch_predictions = batch_response["days_left"]

print("Batch request successful.")
print("Batch predictions received:", len(batch_predictions))
print("Batch timings:", batch_response["timings_ms"])


# 2. Send the same readings individually.
print("\nSending 50 individual requests...")

individual_predictions = []
individual_start = time.perf_counter()

for index, reading in enumerate(readings, start=1):
    response = send_request("/predict", reading)
    individual_predictions.append(response["days_left"])

    if index % 10 == 0:
        print(f"Completed {index}/50 individual requests")

individual_elapsed = (time.perf_counter() - individual_start) * 1000


# 3. Compare batch and individual predictions.
differences = [
    abs(batch - individual)
    for batch, individual in zip(
        batch_predictions, individual_predictions
    )
]

max_difference = max(differences)
matching = all(diff <= 0.01 for diff in differences)

print("\n--- COMPARISON RESULTS ---")
print("Batch predictions:", len(batch_predictions))
print("Individual predictions:", len(individual_predictions))
print("Maximum prediction difference:", round(max_difference, 4), "days")
print("Individual requests total time:", round(individual_elapsed, 3), "ms")

if matching:
    print("\nSUCCESS: All 50 batch predictions match individual predictions.")
else:
    print("\nERROR: Some batch and individual predictions differ.")

    for i, diff in enumerate(differences):
        if diff > 0.01:
            print(
                f"Reading {i + 1}: "
                f"batch={batch_predictions[i]}, "
                f"individual={individual_predictions[i]}, "
                f"difference={diff}"
            )

print("\n--- TEST COMPLETED ---")
