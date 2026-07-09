"""
Daily batch scheduler.

A background thread checks the configured schedule (settings table:
schedule_enabled, schedule_time as local HH:MM) and kicks off one batch
pass per day at that time. Manual "Run now" batches are unaffected.
"""

import threading
from datetime import date, datetime

from . import db, pipeline

_thread: threading.Thread | None = None
_stop = threading.Event()
_last_run_day: date | None = None


def _due(now: datetime, schedule_time: str) -> bool:
    try:
        hh, mm = (int(x) for x in schedule_time.strip().split(":"))
    except ValueError:
        return False
    return (now.hour, now.minute) >= (hh, mm)


def _loop() -> None:
    global _last_run_day
    while not _stop.is_set():
        try:
            settings = db.get_settings()
            if settings.get("schedule_enabled", "0") == "1":
                now = datetime.now()
                if _last_run_day != now.date() and _due(now, settings.get("schedule_time", "02:00")):
                    _last_run_day = now.date()
                    print(f"[scheduler] starting scheduled pass at {now:%H:%M}")
                    pipeline.run_batch(trigger="scheduled")
        except Exception as e:
            print(f"[scheduler] error: {e}")
        _stop.wait(timeout=30)


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, daemon=True, name="qc-scheduler")
    _thread.start()


def stop() -> None:
    _stop.set()


def next_run_description() -> str:
    settings = db.get_settings()
    if settings.get("schedule_enabled", "0") != "1":
        return "disabled"
    t = settings.get("schedule_time", "02:00")
    if _last_run_day == date.today():
        return f"tomorrow at {t}"
    now = datetime.now()
    return f"today at {t}" if not _due(now, t) else f"tomorrow at {t}"
