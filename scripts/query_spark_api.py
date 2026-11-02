import json
import urllib.request

base = "http://127.0.0.1:4040"
try:
    with urllib.request.urlopen(f"{base}/api/v1/applications") as resp:
        apps = json.loads(resp.read().decode("utf-8"))
        print("=== SPARK REST API APPLICATIONS ===")
        print(json.dumps(apps, indent=2))
        app_id = apps[0]["id"]

    with urllib.request.urlopen(f"{base}/api/v1/applications/{app_id}/jobs") as resp:
        jobs = json.loads(resp.read().decode("utf-8"))
        print(f"\n=== JOBS COMPLETED ({len(jobs)}) ===")
        for j in jobs:
            print(f"Job ID: {j['jobId']}, Name: {j.get('name')}, Status: {j.get('status')}, Stages: {j.get('stageIds')}")

    with urllib.request.urlopen(f"{base}/api/v1/applications/{app_id}/stages") as resp:
        stages = json.loads(resp.read().decode("utf-8"))
        print(f"\n=== STAGES COMPLETED ({len(stages)}) ===")
        for s in stages:
            print(f"Stage ID: {s['stageId']}, Name: {s.get('name')}, Status: {s.get('status')}, Tasks: {s.get('numCompleteTasks')}/{s.get('numTasks')}")

    with urllib.request.urlopen(f"{base}/api/v1/applications/{app_id}/environment") as resp:
        env = json.loads(resp.read().decode("utf-8"))
        print("\n=== SPARK ENVIRONMENT PROPERTIES ===")
        spark_props = {k: v for k, v in env.get("sparkProperties", []) if "spark." in k}
        for k in sorted(spark_props.keys())[:12]:
            print(f"{k}: {spark_props[k]}")

    with urllib.request.urlopen(f"{base}/api/v1/applications/{app_id}/executors") as resp:
        execs = json.loads(resp.read().decode("utf-8"))
        print(f"\n=== EXECUTORS ({len(execs)}) ===")
        for ex in execs:
            print(f"ID: {ex['id']}, HostPort: {ex['hostPort']}, Cores: {ex.get('totalCores')}, Memory: {ex.get('maxMemory') / (1024*1024):.1f} MB")

except Exception as e:
    print("Error querying Spark REST API:", e)
