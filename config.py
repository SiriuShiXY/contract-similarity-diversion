import datetime

# Panel construction parameters.
OIS_TENORS = [
    ("OIS1W", 7),
    ("OIS2W", 14),
    ("OIS3W", 21),
    ("OIS1M", 30),
    ("OIS2M", 60),
    ("OIS3M", 90),
    ("OIS4M", 120),
    ("OIS5M", 150),
    ("OIS6M", 180),
    ("OIS8M", 240),
    ("OIS9M", 270),
    ("OIS10M", 300),
    ("OIS11M", 330),
    ("OIS12M", 360),
    ("OIS2Y", 720),
    ("OIS3Y", 1080),
    ("OIS4Y", 1440),
]
HOURS_PER_YEAR = 252 * 6.5
MIN_TAU_H = 1 / 60
CHAR_WINDOW = (9 * 60 + 55, 15 * 60 + 45)
WINSOR_Q = (0.0025, 0.9975)
ATM_DTE = (7, 120)
ATM_MIN_TRADES = 10
TARGET_DAYS = 30
CALENDAR_START = datetime.date(2015, 1, 1)
CALENDAR_END = datetime.date(2030, 12, 31)

EVENT_SPEC = {
    "2016-08-30": {
        "day0": datetime.date(2016, 8, 30),
        "pre_end": datetime.date(2016, 8, 30),
        "start": datetime.date(2016, 8, 30),
        "treated": ["SPY"],
        "control": ["QQQ", "IWM"],
        "control_treated": [],
        "new_wd": [3],
        "n_e": 1,
        "cut": False,
        "label": "2016: SPY adds Wednesday",
        "short": "2016",
        "tab": "SPY 2016 (Wed)",
    },
    "2018-02-16": {
        "day0": datetime.date(2018, 2, 16),
        "pre_end": datetime.date(2018, 2, 16),
        "start": datetime.date(2018, 2, 16),
        "treated": ["SPY"],
        "control": ["QQQ", "IWM"],
        "control_treated": [],
        "new_wd": [1],
        "n_e": 2,
        "cut": False,
        "label": "2018: SPY adds Monday",
        "short": "2018",
        "tab": "SPY 2018 (Mon)",
    },
    "2021-04-27": {
        "day0": datetime.date(2021, 4, 27),
        "pre_end": datetime.date(2021, 4, 23),
        "start": datetime.date(2021, 4, 23),
        "treated": ["QQQ"],
        "control": ["SPY", "IWM"],
        "control_treated": ["SPY"],
        "new_wd": [1, 3],
        "n_e": 1,
        "cut": False,
        "label": "2021: QQQ adds Mon/Wed",
        "short": "2021 QQQ",
        "tab": "QQQ 2021 (Mon, Wed)",
    },
    "2021-10-08": {
        "day0": datetime.date(2021, 10, 8),
        "pre_end": datetime.date(2021, 10, 5),
        "start": datetime.date(2021, 10, 5),
        "treated": ["IWM"],
        "control": ["SPY", "QQQ"],
        "control_treated": ["SPY", "QQQ"],
        "new_wd": [1, 3],
        "n_e": 1,
        "cut": False,
        "label": "2021: IWM adds Mon/Wed",
        "short": "2021 IWM",
        "tab": "IWM 2021 (Mon, Wed)",
    },
    "2022-11-16": {
        "day0": datetime.date(2022, 11, 16),
        "pre_end": datetime.date(2022, 11, 14),
        "start": datetime.date(2022, 11, 14),
        "treated": ["SPY", "QQQ"],
        "control": ["IWM"],
        "control_treated": [],
        "new_wd": [2, 4],
        "n_e": 3,
        "cut": True,
        "label": "2022: SPY, QQQ add Tue/Thu",
        "short": "2022",
        "tab": "SPY/QQQ 2022 (Tue, Thu)",
    },
    "2024-04-18": {
        "day0": datetime.date(2024, 4, 18),
        "pre_end": datetime.date(2024, 4, 16),
        "start": datetime.date(2024, 4, 16),
        "treated": ["IWM"],
        "control": ["SPY", "QQQ"],
        "control_treated": ["SPY", "QQQ"],
        "new_wd": [2, 4],
        "n_e": 3,
        "cut": False,
        "label": "2024: IWM adds Tue/Thu",
        "short": "2024 IWM",
        "tab": "IWM 2024 (Tue, Thu)",
    },
}

# Held-out events never enter fitting or initialization of prediction parameters.
OOS_TRAIN_EVENTS = (
    "2016-08-30",
    "2018-02-16",
    "2021-04-27",
    "2021-10-08",
)
OOS_TEST_EVENTS = ("2022-11-16", "2024-04-18")

SETTINGS = {"lambda_draws": 999, "structural_draws": 499, "full_panel_workers": 8}

# All path defaults and resolution live in Run_All.py.
import json as _json
import os as _os
from pathlib import Path as _Path
import runpy as _runpy
from Run_All import runtime_paths

PATHS = runtime_paths()
_context = _os.environ.get("SIMDIV_PIPELINE_CONTEXT")
if _context:
    _configured = _Path(_json.loads(_Path(_context).read_text())["config"])
    if _configured.resolve() != _Path(__file__).resolve():
        _custom = _runpy.run_path(str(_configured))
        EVENT_SPEC = _custom["EVENT_SPEC"]
        SETTINGS = _custom["SETTINGS"]
        for _name in (
            "OIS_TENORS",
            "HOURS_PER_YEAR",
            "MIN_TAU_H",
            "CHAR_WINDOW",
            "WINSOR_Q",
            "ATM_DTE",
            "ATM_MIN_TRADES",
            "TARGET_DAYS",
            "CALENDAR_START",
            "CALENDAR_END",
            "OOS_TRAIN_EVENTS",
            "OOS_TEST_EVENTS",
        ):
            if _name in _custom:
                globals()[_name] = _custom[_name]
