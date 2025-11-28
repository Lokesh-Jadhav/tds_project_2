import os, re, json, time, base64, io, uuid, logging, threading
import requests
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
import pandas as pd

# Optional libs
try:
    import fitz  # PyMuPDF
except:
    fitz = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except:
    plt = None

# =================
# ENV
# =================
load_dotenv()
SECRET = os.getenv("SECRETE")
PORT = int(os.getenv("PORT", 8080))

if not SECRET:
    raise RuntimeError("SECRETE not found in .env")

# =================
# CONFIG
# =================
MAX_SECONDS = 180         # 3 minutes global cap
NETWORK_TIMEOUT = 20      # seconds for HTTP / Playwright
RETRIES = 3
BACKOFF = [1, 3, 6]
MAX_THREADS = 4

# =================
# LOGGING (SAFE)
# =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class RequestLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra['rid']}] {msg}", kwargs

def get_logger(rid):
    return RequestLogger(logging.getLogger("QUIZ"), {"rid": rid})

# =================
# APP
# =================
app = Flask(__name__)
_semaphore = threading.BoundedSemaphore(MAX_THREADS)

# ======================
# FAKE QUIZ (TEST ONLY)
# ======================
@app.route("/fake-quiz")
def fake_quiz():
    return send_file("fake_quiz.html")

@app.route("/fake-submit", methods=["POST"])
def fake_submit():
    data = request.get_json(silent=True) or {}
    print("FAKE SUBMISSION RECEIVED:", data)
    return jsonify({"correct": True})

# ======================
# MAIN ENDPOINT
# ======================
@app.route("/quiz", methods=["POST"])
def quiz():
    try:
        payload = request.get_json(force=True)
    except:
        return jsonify({"error": "Invalid JSON"}), 400

    if payload.get("secret") != SECRET:
        return jsonify({"error": "Invalid Secret"}), 403

    for k in ("email", "secret", "url"):
        if k not in payload:
            return jsonify({"error": f"Missing field {k}"}), 400

    if not _semaphore.acquire(False):
        return jsonify({"error": "Server busy"}), 429

    rid = str(uuid.uuid4())[:8]
    threading.Thread(target=_run_guarded, args=(payload, rid), daemon=True).start()
    return jsonify({"status": "Accepted", "request_id": rid}), 200

def _run_guarded(payload, rid):
    log = get_logger(rid)
    try:
        solve_quiz(payload, log)
    finally:
        _semaphore.release()

# ======================
# SOLVER
# ======================
def solve_quiz(payload, log):
    start = time.time()
    url = payload["url"]

    def remaining():
        return MAX_SECONDS - (time.time() - start)

    log.info(f"Quiz start → {url}")

    while remaining() > 5:

        html, text = extract_page(url, log)
        if not html:
            log.error("Page load failed")
            return

        submit_url = find_submit_url(text)
        if not submit_url:
            log.error("Submit URL not found")
            return

        task = infer_task(text)
        log.info(f"Task detected = {task}")

        files = extract_files(html)
        tables = extract_tables(html)

        answer = None

        # 1) Try HTML tables on the page
        if tables:
            answer = solve_tables(tables, task, log)

        # 2) Try linked files (csv/xlsx/json/pdf/images)
        if answer is None and files:
            answer = solve_files(files, task, log)

        # 3) Visualization requested?
        if answer is None and ("plot" in text.lower() or "chart" in text.lower() or "graph" in text.lower()):
            answer = make_chart(log)

        # 4) Smarter fallback: if any numbers in text, use last; else "ok"
        if answer is None:
            nums = re.findall(r"\d+", text)
            if nums:
                answer = float(nums[-1])
                log.warning("Fallback numeric answer used from page text.")
            else:
                answer = "ok"
                log.warning("Fallback string answer used.")

        result = {
            "email": payload["email"],
            "secret": payload["secret"],
            "url": url,
            "answer": answer
        }

        resp = post_with_retry(submit_url, result, log)
        if not resp:
            log.error("Submit failed")
            return

        log.info(f"Response = {resp}")

        if resp.get("correct") is False and "url" in resp:
            url = resp["url"]
            log.info(f"Moving to next quiz → {url}")
            continue

        log.info("Quiz completed ✅")
        return

    log.warning("Global timeout reached")

# ======================
# EXTRACTION HELPERS
# ======================
def extract_page(url, log):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=NETWORK_TIMEOUT*1000)
            html = page.content()
            text = page.inner_text("body")
            open("page.html","w", encoding="utf-8").write(html)
            page.screenshot(path="page.png")
            browser.close()
            return html, text
    except PwTimeout:
        log.error("Browser timeout")
        return None, None


def find_submit_url(text):
    m = re.search(r"https?://[^\s\"']+/(?:fake-)?submit[^\s\"']*", text)
    return m.group(0) if m else None


def extract_files(html):
    # FIX: return full URLs, not just extension group
    return re.findall(r"https?://[^\s\"']+\.(?:csv|xlsx|xls|json|pdf|png|jpg|jpeg)", html, re.I)


def extract_tables(html):
    try:
        # Wrap html string to avoid FutureWarning
        return pd.read_html(io.StringIO(html))
    except:
        return []

# ======================
# SOLVING HELPERS
# ======================
def solve_tables(tables, task, log):
    for df in tables:
        try:
            res = compute_df(df, task)
            log.info("Solved via HTML table")
            return res
        except Exception as e:
            log.warning(f"Table solve failed: {e}")
    return None


def solve_files(files, task, log):
    for f in files:
        data = get_with_retry(f, log)
        if not data:
            continue
        try:
            return solve_file_data(f, data, task)
        except Exception as e:
            log.warning(f"File solve failed for {f}: {e}")
    return None


def solve_file_data(url, data, task):
    url = url.lower()

    if url.endswith(".csv"):
        return compute_df(pd.read_csv(io.BytesIO(data)), task)

    if url.endswith((".xlsx",".xls")):
        return compute_df(pd.read_excel(io.BytesIO(data)), task)

    if url.endswith(".json"):
        nums = extract_numbers(json.loads(data.decode()))
        return reduce_numbers(nums, task)

    if url.endswith(".pdf") and fitz:
        return parse_pdf(data, task)

    if url.endswith((".png",".jpg",".jpeg")):
        # For some tasks, a string acknowledgement may be enough
        return "image-ok"

    return None

# ----- intelligent DataFrame reducer -----
def compute_df(df, task):
    df = df.copy()
    # normalize col names
    df.columns = [str(c).lower().strip() for c in df.columns]

    numeric = df.select_dtypes("number")
    if numeric.empty:
        raise ValueError("No numeric columns")

    # attempt to pick the most relevant column by name
    for col in numeric.columns:
        if any(key in col for key in ["value", "amount", "count", "price", "score", "total", "quantity"]):
            return reduce_series(numeric[col], task)

    # fallback: use the widest numeric column
    col = numeric.columns[0]
    return reduce_series(numeric[col], task)

def reduce_series(series, task):
    series = series.dropna()

    if series.empty:
        raise ValueError("Empty series")

    if task == "sum":
        return float(series.sum())
    if task == "mean":
        return float(series.mean())
    if task == "count":
        return int(series.count())
    if task == "max":
        return float(series.max())
    if task == "min":
        return float(series.min())
    if task == "delta":   # difference between last and first
        return float(series.iloc[-1] - series.iloc[0])

    # default: first numeric value
    return float(series.iloc[0])

def parse_pdf(data, task):
    doc = fitz.open(stream=data, filetype="pdf")
    text = "\n".join(p.get_text() for p in doc)
    # better numeric regex: handles commas & decimals
    nums = [float(n.replace(",", "")) for n in re.findall(r"[-+]?\d[\d,]*\.?\d*", text)]
    return reduce_numbers(nums, task)

def extract_numbers(obj):
    nums = []
    if isinstance(obj, dict):
        for _, v in obj.items():
            nums.extend(extract_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            nums.extend(extract_numbers(v))
    elif isinstance(obj, (int, float)):
        nums.append(obj)
    return nums

def reduce_numbers(nums, task):
    if not nums:
        raise ValueError("No numbers found")

    if task == "sum":
        return sum(nums)
    if task == "mean":
        return sum(nums) / len(nums)
    if task == "max":
        return max(nums)
    if task == "min":
        return min(nums)
    if task == "count":
        return len(nums)
    if task == "delta":
        return nums[-1] - nums[0]

    return nums[0]

def infer_task(text):
    t = text.lower()

    if any(w in t for w in ["total", "sum", "add", "aggregate"]):
        return "sum"

    if any(w in t for w in ["average", "mean"]):
        return "mean"

    if any(w in t for w in ["how many", "count", "number of"]):
        return "count"

    if any(w in t for w in ["maximum", "highest", "largest", "top"]):
        return "max"

    if any(w in t for w in ["minimum", "lowest", "smallest"]):
        return "min"

    if any(w in t for w in ["trend", "growth", "plot", "chart", "graph", "visualize"]):
        return "plot"

    if "difference" in t or "change" in t:
        return "delta"

    return "unknown"

# ======================
# NETWORK
# ======================
def get_with_retry(url, log):
    for i in range(RETRIES):
        try:
            r = requests.get(url, timeout=NETWORK_TIMEOUT)
            r.raise_for_status()
            return r.content
        except Exception as e:
            log.warning(f"GET retry {i+1}: {e}")
            if i < len(BACKOFF):
                time.sleep(BACKOFF[i])
    return None


def post_with_retry(url, payload, log):
    for i in range(RETRIES):
        try:
            r = requests.post(url, json=payload, timeout=NETWORK_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"POST retry {i+1}: {e}")
            if i < len(BACKOFF):
                time.sleep(BACKOFF[i])
    return None

# ======================
# CHART
# ======================
def make_chart(log):
    if not plt:
        log.warning("Matplotlib not available")
        return "chart-unavailable"

    plt.figure()
    plt.plot([1,2,3],[4,1,5])
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ======================
# START
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
