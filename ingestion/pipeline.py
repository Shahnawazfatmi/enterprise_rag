"""
Nightly ingestion pipeline.

Stages (each runs as its own process):
    scraping -> cleaning -> comparison -> vector_update
    -> manifest_update -> baseline_promotion

Reliability features:
- A lock file stops two runs from writing to the database at once.
- latest_run.json says "running" (with the current stage) while a run
  is in progress, so a run killed by Windows / power loss is detected
  and reported by the next run instead of looking like a success.
- Stage output is streamed into the log line by line as it happens.
- Every stage has a timeout.
- Child processes run in UTF-8 mode (no cp1252 crashes on Windows).
- Optional failure alerts to a webhook (ALERT_WEBHOOK_URL).
- The baseline only moves forward after the vector update succeeded,
  so a failed run is simply retried by the next run.
"""

from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    ALERT_WEBHOOK_URL,
    BASELINE_DIR,
    CHANGES_FILE,
    COMPARISON_DIR,
    NEW_STRUCTURED_DIR,
    PIPELINE_LOG_DIR,
    RAW_DIR,
    STAGE_TIMEOUTS_SECONDS,
)

INGESTION_DIR = BASE_DIR / "ingestion"

STATUS_FILE = PIPELINE_LOG_DIR / "latest_run.json"
HISTORY_FILE = PIPELINE_LOG_DIR / "run_history.jsonl"
LOCK_FILE = PIPELINE_LOG_DIR / "pipeline.lock"

# A lock older than this is considered stale even if its PID is reused.
LOCK_MAX_AGE_SECONDS = 6 * 60 * 60

PIPELINE_NAME = "consiva-ingestion-pipeline"

STAGE_SCRIPTS = [
    ("scraping", "scraper.py"),
    ("cleaning", "master_clean.py"),
    ("comparison", "compare.py"),
    ("vector_update", "vector_updater.py"),
]

# Child output that is pure noise in the pipeline log.
NOISY_LINE = re.compile(r"HTTP Request: (HEAD|GET) https://huggingface\.co|it/s\]$")
ERROR_LINE = re.compile(r"\b(ERROR|CRITICAL|Traceback)\b")


# ============================================================
# LOGGING
# ============================================================

PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            PIPELINE_LOG_DIR / "pipeline.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, data: dict) -> None:
    temp_file = path.with_suffix(".tmp")
    temp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_file.replace(path)


def pid_alive(pid: int) -> bool:
    """Check a process without signalling it (os.kill(pid, 0) kills on Windows)."""

    if pid <= 0:
        return False

    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)

        if not handle:
            return False

        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == still_active

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def send_alert(message: str) -> None:
    """Post a failure message to ALERT_WEBHOOK_URL (Slack/Teams/generic). Never raises."""

    if not ALERT_WEBHOOK_URL:
        return

    try:
        request = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=json.dumps({"text": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=15).close()
    except Exception as error:
        logger.warning(f"Could not send alert: {error}")


# ============================================================
# LOCK + STATUS
# ============================================================

def acquire_lock() -> None:
    if LOCK_FILE.exists():
        try:
            lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            lock = {}

        age = time.time() - LOCK_FILE.stat().st_mtime

        if pid_alive(int(lock.get("pid", 0))) and age < LOCK_MAX_AGE_SECONDS:
            raise RuntimeError(
                f"Another pipeline run is in progress (pid {lock.get('pid')}, "
                f"started {lock.get('started_at')})."
            )

        logger.warning(f"Removing stale lock from pid {lock.get('pid')}.")

    write_json_atomic(LOCK_FILE, {"pid": os.getpid(), "started_at": utc_now()})


def release_lock() -> None:
    try:
        lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        if lock.get("pid") == os.getpid():
            LOCK_FILE.unlink()
    except (OSError, json.JSONDecodeError):
        pass


def append_history(status: dict) -> None:
    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(status, ensure_ascii=False) + "\n")


def report_interrupted_run() -> None:
    """If the last run never finished (killed, sleep, power loss), record it now."""

    if not STATUS_FILE.exists():
        return

    try:
        previous = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if previous.get("status") != "running" or pid_alive(int(previous.get("pid", 0))):
        return

    previous.update({
        "status": "interrupted",
        "completed_at": None,
        "error": "Process ended without finishing (killed, sleep, or power loss).",
    })
    append_history(previous)

    message = (
        f"Consiva pipeline run started {previous.get('started_at')} was interrupted "
        f"during stage '{previous.get('current_stage')}'."
    )
    logger.error(message)
    send_alert(message)


def create_status(status: str, started_at: str, **fields) -> dict:
    return {
        "pipeline": PIPELINE_NAME,
        "status": status,
        "pid": os.getpid(),
        "started_at": started_at,
        "completed_at": fields.get("completed_at"),
        "current_stage": fields.get("current_stage"),
        "failed_stage": fields.get("failed_stage"),
        "error": fields.get("error"),
        "changes": fields.get("changes") or {},
        "duration_seconds": fields.get("duration_seconds"),
    }


# ============================================================
# STAGE EXECUTION
# ============================================================

def run_stage(stage_name: str, script_name: str) -> None:
    """Run one stage script, streaming its output into the log."""

    logger.info("=" * 70)
    logger.info(f"STARTING STAGE: {stage_name.upper()}")
    logger.info("=" * 70)

    script_path = INGESTION_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Required stage script not found: {script_path}")

    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }

    timeout = STAGE_TIMEOUTS_SECONDS.get(stage_name, 30 * 60)
    start_time = time.time()

    process = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )

    timed_out = threading.Event()

    def kill_on_timeout():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, kill_on_timeout)
    timer.start()

    try:
        for line in process.stdout:
            line = line.rstrip()
            if not line or NOISY_LINE.search(line):
                continue
            level = logging.ERROR if ERROR_LINE.search(line) else logging.INFO
            logger.log(level, f"[{stage_name}] {line}")
        return_code = process.wait()
    finally:
        timer.cancel()

    duration = time.time() - start_time

    if timed_out.is_set():
        raise TimeoutError(f"Stage '{stage_name}' timed out after {timeout} seconds.")

    if return_code != 0:
        logger.error(f"STAGE FAILED: {stage_name} (exit code={return_code}, duration={duration:.2f}s)")
        raise RuntimeError(f"Pipeline stage '{stage_name}' failed (exit code {return_code}).")

    logger.info(f"STAGE COMPLETED: {stage_name} ({duration:.2f}s)")


# ============================================================
# CHANGES
# ============================================================

def load_changes() -> dict:
    if not CHANGES_FILE.exists():
        raise FileNotFoundError(f"Comparison file not found: {CHANGES_FILE}")

    changes = json.loads(CHANGES_FILE.read_text(encoding="utf-8"))

    if not isinstance(changes.get("pages"), dict):
        raise ValueError("Invalid changes.json: 'pages' must be an object.")

    return changes


def summarize(changes: dict) -> dict:
    pages = changes["pages"]
    return {
        status: pages.get(status) if isinstance(pages.get(status), list) else []
        for status in ("new", "changed", "unchanged", "deleted")
    }


def get_successful_page_paths(changes: dict) -> list[Path]:
    """NEW and CHANGED page files in new_structured (used after vector DB success)."""

    paths = []
    details = changes.get("details", {})

    for status in ("new", "changed"):
        for page_id in changes["pages"].get(status, []):
            filename = (details.get(page_id, {}).get("new") or {}).get("file_name")
            if not filename:
                logger.warning(f"No file name recorded for page: {page_id}")
                continue

            file_path = NEW_STRUCTURED_DIR / filename

            if file_path.exists():
                paths.append(file_path)
            else:
                logger.warning(f"Changed page file not found: {file_path}")

    return paths


def update_manifest(successful_pages: list[Path], deleted_page_ids: list[str]) -> None:
    from ingestion.manifest import apply_changes

    logger.info("Updating manifest...")
    apply_changes(successful_pages=successful_pages, deleted_page_ids=deleted_page_ids)
    logger.info("Manifest update completed.")


def promote_baseline() -> None:
    from ingestion.promote import promote_baseline as promote

    logger.info("Promoting baseline...")
    promote()
    logger.info("Baseline promotion completed.")


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(skip_scrape: bool = False) -> None:

    started_at = utc_now()
    start_time = time.time()
    current_stage = "startup"

    logger.info("")
    logger.info("#" * 70)
    logger.info(f"STARTING {PIPELINE_NAME.upper()} (pid {os.getpid()})")
    logger.info("#" * 70)

    report_interrupted_run()
    acquire_lock()

    def mark(stage: str) -> None:
        nonlocal current_stage
        current_stage = stage
        write_json_atomic(STATUS_FILE, create_status("running", started_at, current_stage=stage))

    try:
        for directory in (RAW_DIR, NEW_STRUCTURED_DIR, BASELINE_DIR, COMPARISON_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        for stage_name, script_name in STAGE_SCRIPTS:
            if skip_scrape and stage_name == "scraping":
                logger.info("Skipping scraping (--skip-scrape): using existing data/raw")
                continue

            mark(stage_name)
            run_stage(stage_name, script_name)

            if stage_name == "comparison":
                changes = load_changes()
                change_summary = summarize(changes)
                logger.info(
                    "Change summary | "
                    + ", ".join(f"{key.upper()}={len(value)}" for key, value in change_summary.items())
                )

        # Everything below runs only after the vector update succeeded.
        mark("manifest_update")
        update_manifest(
            successful_pages=get_successful_page_paths(changes),
            deleted_page_ids=list(change_summary["deleted"]),
        )

        mark("baseline_promotion")
        promote_baseline()

        status = create_status(
            "success",
            started_at,
            completed_at=utc_now(),
            changes={key: len(value) for key, value in change_summary.items()},
            duration_seconds=round(time.time() - start_time, 2),
        )
        write_json_atomic(STATUS_FILE, status)
        append_history(status)

        logger.info("#" * 70)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY ({status['duration_seconds']}s)")
        logger.info("#" * 70)

    except BaseException as error:
        cancelled = isinstance(error, KeyboardInterrupt)
        status = create_status(
            "cancelled" if cancelled else "failed",
            started_at,
            completed_at=utc_now(),
            failed_stage=current_stage,
            error=str(error) or type(error).__name__,
            duration_seconds=round(time.time() - start_time, 2),
        )
        write_json_atomic(STATUS_FILE, status)
        append_history(status)

        logger.error("#" * 70)
        logger.error(f"PIPELINE {status['status'].upper()} at stage '{current_stage}': {status['error']}")
        logger.error("#" * 70)

        if not cancelled:
            send_alert(f"Consiva pipeline failed at stage '{current_stage}': {status['error']}")

        raise

    finally:
        release_lock()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consiva ingestion pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="re-process the last scrape in data/raw without calling Firecrawl",
    )
    arguments = parser.parse_args()

    try:
        run_pipeline(skip_scrape=arguments.skip_scrape)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        sys.exit(1)
