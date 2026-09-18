import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button

import requests
import numpy as np
import matplotlib.pyplot as plt
import eccodes


BASE = "https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc-eps/p/T_2M/r"
OPER_BASE = "https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/T_2M/r"
CACHE = "cache"

MEMBERS = [f"{i:02d}" for i in range(1, 21)]

AIRPORTS = {
    "EDDM": 298420,
    "LIMC": 411180,
    "EGLC": 323946,
    "LFPB": 268742,
}

AIRPORT_NAMES = {
    "EDDM": "Munich",
    "LIMC": "Milan Malpensa",
    "EGLC": "London City",
    "LFPB": "Paris Le Bourget",
}

MAX_WORKERS = 10
CHECK_SECONDS = 30
MAX_FORECAST_HOUR = 14
REQUEST_TIMEOUT = (8, 30)

AIRPORT_TZ = {
    "EDDM": "Europe/Berlin",
    "LIMC": "Europe/Rome",
    "EGLC": "Europe/London",
    "LFPB": "Europe/Paris",
}

LOCAL_TZ = datetime.now().astimezone().tzinfo
LAST_EXTRACTION = None
LAST_AVAILABLE_UTC = None
LAST_REFRESH_UTC = None
LAST_PULL_UTC = None
CURRENT_TIME_TEXT = None
LIVE_TIMER = None
RECENT_RUNS = []
VIEW_INDEX = 0

os.makedirs(CACHE, exist_ok=True)

FIG = None
AXES = None
HOVER_ANNOTATION = None
HOVER_LINE = None
CURRENT_DATA = {}
CURRENT_AVG = {}
CURRENT_OPER = {}
CURRENT_RUN = None
CURRENT_OPER_RUN = None


def get(url):
    return requests.get(
        url,
        timeout=(15, 60),
        headers={"User-Agent": "ICON-D2-RUC-EPS Meteogram"},
    )


def newest_run():
    r = get(BASE + "/")
    r.raise_for_status()

    runs = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", r.text)

    if not runs:
        raise RuntimeError("Could not find a DWD model run.")

    return sorted(set(runs))[-1]


def available_hours(run):
    global LAST_AVAILABLE_UTC, LAST_REFRESH_UTC
    url = f"{BASE}/{run}/e/01/s/"
    r = get(url)
    r.raise_for_status()
    LAST_REFRESH_UTC = datetime.now(timezone.utc)

    hours = sorted(set(int(x) for x in re.findall(r"PT(\d{3})H00M\.grib2", r.text)))

    if 0 in hours:
        try:
            h = requests.head(
                file_url(run, "01", 0),
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "ICON-D2-RUC-EPS Meteogram"},
                allow_redirects=True,
            )
            lm = h.headers.get("Last-Modified")
            if lm:
                LAST_AVAILABLE_UTC = parsedate_to_datetime(lm).astimezone(timezone.utc)
        except Exception:
            pass

    return hours


def file_url(run, member, hour):
    filename = f"PT{hour:03d}H00M.grib2"
    return f"{BASE}/{run}/e/{member}/s/{filename}"


def oper_file_url(run, hour):
    filename = f"PT{hour:03d}H00M.grib2"
    return f"{OPER_BASE}/{run}/s/{filename}"


def cache_file(run, member, hour):
    safe_run = run.replace(":", "")
    return os.path.join(
        CACHE,
        f"{safe_run}_{member}_{hour:03d}.grib2",
    )


def oper_cache_file(run, hour):
    safe_run = run.replace(":", "")
    return os.path.join(
        CACHE,
        f"oper_{safe_run}_{hour:03d}.grib2",
    )


def read_values(filename):
    result = {}
    with open(filename, "rb") as f:
        handle = eccodes.codes_grib_new_from_file(f)
        if handle is None:
            raise RuntimeError("Could not read GRIB file.")
        try:
            for airport, index in AIRPORTS.items():
                value = eccodes.codes_get_double_element(handle, "values", index) - 273.15
                result[airport] = float(value) if np.isfinite(value) and abs(value) < 100 else np.nan
        finally:
            eccodes.codes_release(handle)
    return result


def _download_to_cache(url, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 100000:
        try:
            return read_values(filename)
        except Exception:
            try:
                os.remove(filename)
            except OSError:
                pass

    for attempt in range(3):
        try:
            r = get(url)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            temporary = filename + ".tmp"
            with open(temporary, "wb") as f:
                f.write(r.content)
            os.replace(temporary, filename)
            return read_values(filename)
        except Exception as e:
            if attempt == 2:
                print(f"  download failed: {e}")
                return None
            time.sleep(1 + attempt)


def download_member(run, member, hour):
    values = _download_to_cache(
        file_url(run, member, hour),
        cache_file(run, member, hour),
    )
    return member, hour, values


def download_oper(run, hour):
    values = _download_to_cache(
        oper_file_url(run, hour),
        oper_cache_file(run, hour),
    )
    return "OPER", hour, values



def fetch_hour(eps_run, oper_run, hour):
    results = {}
    oper = None
    print(f"Downloading hour +{hour:02d} ({len(MEMBERS)} EPS + OPER)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        jobs = [executor.submit(download_member, eps_run, member, hour) for member in MEMBERS]
        jobs.append(executor.submit(download_oper, oper_run, hour))

        for job in as_completed(jobs):
            who, returned_hour, values = job.result()
            if who == "OPER":
                oper = values
            elif values is not None:
                results[who] = values

    print(f"  Received {len(results)}/20 EPS members" + (" + OPER" if oper is not None else ""))
    return results, oper



def run_dt(run):
    return datetime.strptime(run, "%Y-%m-%dT%H:%M").replace(tzinfo=ZoneInfo("UTC"))


def local_valid_time(run, hour, airport):
    utc_dt = run_dt(run) + timedelta(hours=hour)
    return utc_dt.astimezone(ZoneInfo(AIRPORT_TZ[airport]))


def hover_move(event):
    global HOVER_ANNOTATION, HOVER_LINE

    if FIG is None or event.inaxes not in AXES or event.xdata is None or CURRENT_RUN is None:
        if HOVER_ANNOTATION is not None:
            HOVER_ANNOTATION.set_visible(False)
        if HOVER_LINE is not None:
            HOVER_LINE.set_visible(False)
        return

    ax = event.inaxes
    airport = list(AIRPORTS.keys())[list(AXES).index(ax)]

    all_hours = sorted({
        hour
        for member in MEMBERS
        for hour in CURRENT_DATA.get(member, {})
        if airport in CURRENT_DATA.get(member, {}).get(hour, {})
    })
    if not all_hours:
        return

    nearest_hour = min(all_hours, key=lambda h: abs(h - event.xdata))
    local_dt = (run_dt(CURRENT_RUN) + timedelta(hours=nearest_hour)).astimezone(
        ZoneInfo(AIRPORT_TZ[airport])
    )

    # Sort members by temperature, highest first, while retaining member IDs.
    member_values = []
    for member in MEMBERS:
        value = CURRENT_DATA.get(member, {}).get(nearest_hour, {}).get(airport, np.nan)
        member_values.append((member, value))

    member_values.sort(
        key=lambda item: (-item[1] if np.isfinite(item[1]) else float("inf"))
    )

    rows = [
        f"{member}: {value:5.1f} °C" if np.isfinite(value) else f"{member}:   —"
        for member, value in member_values
    ]

    valid_vals = [v for _, v in member_values if np.isfinite(v)]
    ens_min = min(valid_vals) if valid_vals else np.nan
    ens_max = max(valid_vals) if valid_vals else np.nan
    avg = CURRENT_AVG.get(nearest_hour, {}).get(airport, np.nan)
    oper = CURRENT_OPER.get(nearest_hour, {}).get(airport, np.nan)

    def fmt(v):
        return f"{v:5.1f} °C" if np.isfinite(v) else "  —"

    member_lines = "\n".join(
        f"{rows[i]:<15}   {rows[i + 10]}"
        for i in range(10)
    )

    text = (
        f"{airport} — {local_dt.strftime('%a %d %b %H:%M')} local  (+{nearest_hour:02d}h)\n"
        f"{member_lines}\n\n"
        f"MIN : {fmt(ens_min)}      MAX : {fmt(ens_max)}\n"
        f"AVG : {fmt(avg)}      OPER: {fmt(oper)}"
    )

    if HOVER_ANNOTATION is not None:
        HOVER_ANNOTATION.remove()
    if HOVER_LINE is not None:
        HOVER_LINE.remove()

    x_frac = (nearest_hour - ax.get_xlim()[0]) / max(ax.get_xlim()[1] - ax.get_xlim()[0], 1)
    ha = "left" if x_frac > 0.62 else "right"
    xytext = (0.02, 0.98) if ha == "left" else (0.98, 0.98)

    HOVER_LINE = ax.axvline(
        nearest_hour,
        linestyle="--",
        linewidth=0.8,
        alpha=0.55,
    )
    HOVER_ANNOTATION = ax.annotate(
        text,
        xy=(nearest_hour, ax.get_ylim()[1]),
        xycoords="data",
        xytext=xytext,
        textcoords="axes fraction",
        ha=ha,
        va="top",
        fontsize=7.5,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.38",
            facecolor="white",
            edgecolor="black",
            alpha=0.78,
        ),
    )
    FIG.canvas.draw_idle()




def make_plot(run, oper_run, data, oper_data):
    global FIG, AXES, HOVER_ANNOTATION, HOVER_LINE
    global CURRENT_DATA, CURRENT_RUN, CURRENT_OPER_RUN, CURRENT_OPER, CURRENT_AVG, LAST_EXTRACTION, LIVE_TIMER

    CURRENT_DATA = data
    CURRENT_RUN = run
    CURRENT_OPER_RUN = oper_run
    CURRENT_OPER = oper_data
    LAST_EXTRACTION = datetime.now(timezone.utc)


    if FIG is None:
        FIG, AXES = plt.subplots(
            2, 2,
            figsize=(15.2, 7.55),
            squeeze=False,
        )
        AXES = AXES.ravel()
        FIG.canvas.mpl_connect("motion_notify_event", hover_move)
        try:
            FIG.canvas.manager.set_window_title("DWD ICON-D2 RUC-EPS Meteogram")
        except Exception:
            pass
        plt.show(block=False)
        LIVE_TIMER = FIG.canvas.new_timer(interval=1000)
        LIVE_TIMER.single_shot = False
        LIVE_TIMER.add_callback(update_live_status)
        LIVE_TIMER.start()

    else:
        for ax in AXES:
            ax.clear()

    if HOVER_ANNOTATION is not None:
        try: HOVER_ANNOTATION.remove()
        except Exception: pass
        HOVER_ANNOTATION = None
    if HOVER_LINE is not None:
        try: HOVER_LINE.remove()
        except Exception: pass
        HOVER_LINE = None

    all_hours = sorted(set(h for member in MEMBERS for h in data.get(member, {})))

    CURRENT_AVG = {}
    for hour in all_hours:
        CURRENT_AVG[hour] = {}
        for airport in AIRPORTS:
            vals = [
                data.get(member, {}).get(hour, {}).get(airport, np.nan)
                for member in MEMBERS
            ]
            vals = [v for v in vals if np.isfinite(v)]
            CURRENT_AVG[hour][airport] = float(np.mean(vals)) if vals else np.nan

    for ax, airport in zip(AXES, AIRPORTS):
        for member in MEMBERS:
            hours = sorted(h for h in data.get(member, {}) if airport in data[member][h])
            if hours:
                ax.plot(
                    hours,
                    [data[member][h][airport] for h in hours],
                    linewidth=0.75,
                    alpha=0.48,
                    label=member,
                )

        if CURRENT_AVG:
            avg_hours = sorted(CURRENT_AVG)
            ax.plot(
                avg_hours,
                [CURRENT_AVG[h].get(airport, np.nan) for h in avg_hours],
                linewidth=2.8,
                label="AVG",
            )

        if oper_data:
            oper_hours = sorted(oper_data)
            ax.plot(
                oper_hours,
                [oper_data[h].get(airport, np.nan) for h in oper_hours],
                linewidth=2.8,
                linestyle="--",
                label="OPER",
            )

        ax.set_title(
            f"{airport} — {AIRPORT_NAMES[airport]}",
            fontsize=11.5,
            fontweight="bold",
            pad=3,
        )
        ax.set_ylabel("°C", fontsize=8.5)
        ax.set_xlabel("", labelpad=2)
        ax.grid(True, alpha=0.20)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

        if all_hours:
            ax.set_xlim(-0.1, max(1, max(all_hours) + 0.1))
            ax.set_xticks(all_hours)

            tz = ZoneInfo(AIRPORT_TZ[airport])
            labels2 = []
            for h in all_hours:
                local_dt = (run_dt(run) + timedelta(hours=h)).astimezone(tz)
                # Only show date when the local calendar date changes.
                if h == all_hours[0] or local_dt.date() != (
                    run_dt(run) + timedelta(hours=all_hours[all_hours.index(h) - 1])
                ).astimezone(tz).date():
                    labels2.append(local_dt.strftime("%d %b\n%H:%M"))
                else:
                    labels2.append(local_dt.strftime("%H:%M"))
            ax.set_xticklabels(labels2, fontsize=7.5)

    for lg in list(FIG.legends):
        lg.remove()

    handles, labels = AXES[0].get_legend_handles_labels()
    if handles:
        legend = FIG.legend(
            handles,
            labels,
            title="20 Members — AVG / OPER",
            ncol=11,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            fontsize=6.9,
            title_fontsize=8.5,
            frameon=True,
            handlelength=1.6,
            columnspacing=0.65,
            borderpad=0.35,
        )
        for txt, lab in zip(legend.get_texts(), labels):
            if lab in ("AVG", "OPER"):
                txt.set_fontweight("bold")
                txt.set_fontsize(9.0)

    eps_z = run_dt(run)
    oper_z = run_dt(oper_run)

    if LAST_AVAILABLE_UTC is not None:
        avail_text = LAST_AVAILABLE_UTC.strftime("%H:%MZ")
    else:
        avail_text = "not reported yet"

        FIG.suptitle(
        "DWD ICON-D2 RUC-EPS — 20 Ensemble Members",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    FIG.text(
        0.01, 0.970,
        f"{eps_z.strftime('%HZ')} • available from {avail_text}",
        ha="left", va="top", fontsize=8.4, fontweight="bold",
    )
    FIG.text(
        0.99, 0.970,
        f"EPS initialized {eps_z.strftime('%d %b %Y %H:%M UTC')}",
        ha="right", va="top", fontsize=8.4,
    )
    FIG.text(
        0.5, 0.945,
        "Local Time",
        ha="center", va="top", fontsize=8.6, fontweight="bold",
    )

    FIG.subplots_adjust(
        left=0.042,
        right=0.992,
        top=0.815,
        bottom=0.075,
        wspace=0.10,
        hspace=0.22,
    )
    FIG.canvas.draw_idle()
    FIG.canvas.flush_events()
    plt.pause(0.01)




def main():
    global LAST_PULL_UTC
    plt.ion()

    data = {member: {} for member in MEMBERS}
    oper_data = {}
    current_run = None
    current_oper_run = None
    processed_hours = set()

    print("ICON-D2 RUC-EPS meteogram")
    print("Starting...")

    while True:
        try:
            eps_run = newest_run()
            oper_run = newest_run_from_base(OPER_BASE)

            if eps_run != current_run or oper_run != current_oper_run:
                print()
                print("======================================")
                print("NEW MODEL RUN:", eps_run)
                print("OPER RUN:", oper_run)
                print("======================================")
                current_run = eps_run
                current_oper_run = oper_run
                data = {member: {} for member in MEMBERS}
                oper_data = {}
                processed_hours = set()

            hours = [h for h in available_hours(eps_run) if h <= MAX_FORECAST_HOUR]
            if hours:
                print("Available forecast hours:", hours)

            # Always make the first available hour appear before continuing.
            for hour in hours:
                if hour in processed_hours:
                    continue

                results, oper_values = fetch_hour(eps_run, oper_run, hour)
                for member, values in results.items():
                    data[member][hour] = values
                if oper_values is not None:
                    oper_data[hour] = oper_values

                processed_hours.add(hour)
                make_plot(eps_run, oper_run, data, oper_data)

            for _ in range(CHECK_SECONDS):
                if FIG is not None and not plt.fignum_exists(FIG.number):
                    return
                plt.pause(1)

        except KeyboardInterrupt:
            print()
            print("Stopped.")
            break
        except Exception as e:
            print()
            print("Error:", e)
            print("Retrying in 30 seconds...")
            for _ in range(30):
                plt.pause(1)


def newest_run_from_base(base):
    r = get(base + "/")
    r.raise_for_status()
    runs = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", r.text)
    if not runs:
        raise RuntimeError("Could not find a DWD model run.")
    return sorted(set(runs))[-1]


if __name__ == "__main__":
    main()
