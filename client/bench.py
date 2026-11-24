
"""
client/bench.py -- time your API from the outside.

Run from your LAPTOP:
    python client/bench.py --url https://<your-codespace>-8000.app.github.dev --n 200

Run inside the Codespace:
    python client/bench.py --url http://localhost:8000 --n 200

Disable persistent connections:
    python client/bench.py --url http://localhost:8000 --n 200 --no-keepalive

It records:
  - First request separately (cold request)
  - p50, p95 and p99 round-trip latency
  - p50, p95 and p99 for every server timing stage
  - Network + framework time
  - Progress every 10 requests
  - Results saved to a timestamped CSV

Needs:
    pip install requests numpy
"""

import argparse
import csv
import time

import numpy as np
import requests


def reading(r):
    return {
        "battery_v": round(r.uniform(3.4, 4.2), 3),
        "temp_c": round(r.uniform(10, 45), 1),
        "rings_per_day": int(r.integers(0, 30)),
        "wifi_rssi_dbm": int(r.integers(-88, -45)),
        "age_months": int(r.integers(0, 37)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-keepalive", action="store_true")
    a = ap.parse_args()

    if a.n < 2:
        ap.error("--n must be at least 2 to calculate percentiles excluding the first request")

    url = a.url.rstrip("/")
    r = np.random.default_rng(1)

    print("Checking API health...", flush=True)

    try:
        health = requests.get(url + "/health", timeout=(10, 30))
        health.raise_for_status()
        print("health:", health.json(), flush=True)
    except requests.RequestException as exc:
        print("Could not connect to API:", exc, flush=True)
        return

    # Use one session for connection-pool management.
    # Connection: close tells the server/client not to keep TCP connections alive.
    session = requests.Session()

    if a.no_keepalive:
        session.headers.update({"Connection": "close"})

    rows = []
    errors = 0
    start_time = time.perf_counter()

    print(
        f"\nStarting benchmark: {a.n} requests | "
        f"keep-alive: {not a.no_keepalive}",
        flush=True,
    )

    try:
        for i in range(a.n):
            if i % 10 == 0:
                print(
                    f"Progress: {i}/{a.n} requests completed",
                    flush=True,
                )

            t0 = time.perf_counter()

            try:
                resp = session.post(
                    url + "/predict",
                    json=reading(r),
                    timeout=(10, 30),
                )

                rt = (time.perf_counter() - t0) * 1000

                if resp.status_code != 200:
                    print(
                        f"Request {i} failed: "
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        flush=True,
                    )
                    errors += 1
                    resp.close()
                    continue

                data = resp.json()
                tm = data.get("timings_ms", {})

                row = {
                    "i": i,
                    "round_trip": rt,
                    **tm,
                }

                if "server_total" in tm:
                    row["network_and_framework"] = (
                        rt - tm["server_total"]
                    )

                rows.append(row)
                resp.close()

            except requests.RequestException as exc:
                errors += 1
                print(
                    f"\nRequest {i} connection/timeout error: {exc}",
                    flush=True,
                )
                continue

            except ValueError as exc:
                errors += 1
                print(
                    f"\nRequest {i} returned invalid JSON: {exc}",
                    flush=True,
                )
                continue

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", flush=True)

    finally:
        session.close()

    elapsed = time.perf_counter() - start_time

    print("\n========== BENCHMARK SUMMARY ==========", flush=True)
    print(f"Successful requests: {len(rows)}")
    print(f"Failed requests: {errors}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    if not rows:
        print("No successful prediction requests. No CSV created.")
        return

    print(
        f"\nFirst request: {rows[0]['round_trip']:.3f} ms round trip",
        flush=True,
    )

    # Exclude the first request from steady-state percentile calculations.
    steady_rows = rows[1:]

    if not steady_rows:
        print("Not enough successful requests for percentiles.")
        return

    keys = [
        k for k in rows[0]
        if k != "i"
    ]

    print(
        f"\n{'stage':24s} {'p50 ms':>9s} "
        f"{'p95 ms':>9s} {'p99 ms':>9s} "
        f"   (requests after first: {len(steady_rows)})"
    )

    for key in keys:
        values = np.array([
            row[key]
            for row in steady_rows
            if key in row
        ])

        if values.size:
            print(
                f"{key:24s} "
                f"{np.percentile(values, 50):9.3f} "
                f"{np.percentile(values, 95):9.3f} "
                f"{np.percentile(values, 99):9.3f}"
            )

    # Save all successful measurements, including the first request.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"bench_{timestamp}.csv"

    csv_keys = list(dict.fromkeys(
        key
        for row in rows
        for key in row
    ))

    with open(name, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark results: {name}", flush=True)
    print("========== BENCHMARK COMPLETED ==========", flush=True)


if __name__ == "__main__":
    main()
