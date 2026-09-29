import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


process = subprocess.Popen([sys.executable, "/work/jasmine_local.py", "serve", "--port", "8765"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            urllib.request.urlopen("http://127.0.0.1:8765/", timeout=1)
        except urllib.error.HTTPError as exc:
            if exc.code == 501:
                break
        except urllib.error.URLError:
            time.sleep(0.1)
    body = open("/work/examples/support.json", "rb").read()
    request = urllib.request.Request("http://127.0.0.1:8765/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
        assert response.status == 200
        assert result["answers"]["department"]["choice"] == "billing"
        print(json.dumps({"http_status": response.status, "department": result["answers"]["department"]["choice"]}))
    bad = urllib.request.Request("http://127.0.0.1:8765/v1/systemone", data=b"{}", headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(bad, timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
        print(json.dumps({"bad_request_status": exc.code}))
finally:
    process.terminate()
    process.wait(timeout=5)
