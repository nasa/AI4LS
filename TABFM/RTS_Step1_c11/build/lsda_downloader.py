#!/usr/bin/env python3
"""
04_lsda_pull_via_zips3.py

Production LSDA bulk downloader using the reverse-engineered NLSP API.

Architecture (verified 2026-04-29 via DevTools):
  1. Per experiment: GET recordtables HTML with session cookie
  2. Parse out (file_path, s3_file_guid) pairs (no JSON API exists anymore)
  3. Group CUIDs by folder
  4. For each folder: POST to /data/fsapi/api/v1/zips3/ with the JSON payload
  5. Save returned ZIP to 03_raw_downloads/{exp_id}/{folder}.zip
  6. Optional --unzip: extract in place
  7. Resume-able via sha256 manifest

USAGE:
  # Inventory only (HTML scrape, no downloads):
  python3 04_lsda_pull_via_zips3.py --inventory-only

  # Bulk download for real (overnight run):
  python3 04_lsda_pull_via_zips3.py --bulk

  # Resume an interrupted bulk download:
  python3 04_lsda_pull_via_zips3.py --bulk --resume

  # Bulk download + auto-unzip:
  python3 04_lsda_pull_via_zips3.py --bulk --unzip

  # Restrict to one campaign:
  python3 04_lsda_pull_via_zips3.py --bulk --campaign 1

  # Test on one experiment only (good for verifying cookie):
  python3 04_lsda_pull_via_zips3.py --bulk --only-exp MR080G

PRE-FLIGHT REQUIREMENTS:
  Run from bedrest_tfm/ project root.
  Catalog must exist: 02_scrapes/nlsp_catalog_bedrest_floor_*.json (with _key)
  Cookie must exist: 02_scrapes/cookie.txt
    File must contain the raw Cookie header value (everything after `-b '`
    in your DevTools cURL copy, ending before the closing quote).
    Example file content (single line):
      SMSESSION=tyDVFY...; _ga=GA1.1.253...; _csrf=0Ncheth...; lastAccess=...

REQUIREMENTS: requests (no other deps; uses regex instead of BeautifulSoup).
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip3 install requests --break-system-packages")
    sys.exit(1)


# === CONFIGURATION ===
ALIAS = "lsdapub"
RECORD_TYPE = "lsda_experiment"

RECORDTABLES_URL = (
    "https://nlsp.nasa.gov/query/web/api/v1/recordtables/"
    "{alias}/{record_type}/{key}/?linkedtablepages=lsda_dataset:1:1000"
)
ZIPS3_URL = "https://nlsp.nasa.gov/data/fsapi/api/v1/zips3/"
VIEW_URL = "https://nlsp.nasa.gov/view/{alias}/{record_type}/{key}"
PRIME_URL = "https://nlsp.nasa.gov/explore/lsdahome"

SLEEP_RECORDTABLES = 0.5     # between recordtables HTML fetches
SLEEP_DOWNLOAD = 1.0          # between zips3 download POSTs
DOWNLOAD_TIMEOUT = 600        # 10 minutes per ZIP (some are large)
MAX_FILE_SIZE_MB = 5000       # safety brake per ZIP
RETRY_COUNT = 1               # 1 retry per failed request
MAX_CONSECUTIVE_FAILURES = 5  # circuit breaker: abort after N back-to-back fails
MIN_FREE_DISK_GB = 20         # pre-flight warn threshold
INVENTORY_SCHEMA_VERSION = "1.1"
SCRIPT_VERSION = "v7 (2026-04-29)"
FLOOR_SCOPE_CAMPAIGNS = [1, 3, 11]

# Known alias map: alias text in folder name -> canonical UTMB campaign number.
# Sourced from MR080G catalog correlation between mission_id list and payload_id list:
#   missions: [..., "UTMB Campaign 11", ...]   payload_id: [..., "CFT70", ...]
# Add new entries here as they're verified. Do NOT guess — verify against catalog
# metadata or NASA documentation before adding an entry.
CAMPAIGN_ALIASES = {
    "CFT70": 11,
}

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36")

# === DIRECTORIES ===
DIR_SCRAPES = Path("02_scrapes")
DIR_RAW = Path("03_raw_downloads")
DIR_MANIFESTS = Path("04_acquisition_manifests")

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")

# === CAMPAIGN MATCHING ===
TARGET_CAMPAIGNS = ["UTMB Campaign 1", "UTMB Campaign 3", "UTMB Campaign 11"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def mkdirs():
    for d in [DIR_SCRAPES, DIR_RAW, DIR_MANIFESTS]:
        d.mkdir(parents=True, exist_ok=True)


def safe_filename(name):
    if not name:
        return "unnamed"
    return re.sub(r"[^\w\-.]+", "_", name)[:200]


# ============================================================
# COOKIE LOADING
# ============================================================

def load_cookie_header():
    """Load raw Cookie header value from 02_scrapes/cookie.txt."""
    p = DIR_SCRAPES / "cookie.txt"
    if not p.exists():
        log(f"ERROR: Cookie file not found at {p}")
        log("Create it by extracting the cookies from your DevTools cURL copy.")
        log("Example:")
        log("  echo 'SMSESSION=...; _ga=...; _csrf=...; lastAccess=...' > 02_scrapes/cookie.txt")
        sys.exit(1)
    raw = p.read_text().strip()
    if not raw:
        log(f"ERROR: Cookie file is empty: {p}")
        sys.exit(1)
    if "SMSESSION" not in raw:
        log(f"WARNING: Cookie file does not contain SMSESSION. Continuing but downloads will likely fail.")
    return raw


def cookie_age_minutes():
    """Return age of cookie file in minutes, or None if not available."""
    p = DIR_SCRAPES / "cookie.txt"
    if not p.exists():
        return None
    age_sec = time.time() - p.stat().st_mtime
    return age_sec / 60


def preflight_cookie(session, cookie_header, sample_key, sample_exp_id):
    """Probe one recordtables fetch with the cookie. Returns (ok_bool, message)."""
    log("PRE-FLIGHT: Cookie validation")
    age = cookie_age_minutes()
    if age is not None:
        log(f"  Cookie file age: {age:.0f} minutes")
        if age > 240:
            log(f"  WARNING: Cookie is over 4 hours old. May expire mid-run.")
    url = RECORDTABLES_URL.format(alias=ALIAS, record_type=RECORD_TYPE, key=sample_key)
    referer = VIEW_URL.format(alias=ALIAS, record_type=RECORD_TYPE, key=sample_key)
    headers = {
        "Accept": "text/html, */*; q=0.01",
        "Cookie": cookie_header,
        "Referer": referer,
        "User-Agent": USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        r = session.get(url, headers=headers, timeout=20)
    except requests.exceptions.RequestException as e:
        return False, f"network error: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    if "lsda_dataset_file_checkbox" not in r.text:
        return False, "response does not contain file checkboxes (cookie likely expired)"
    n_checkboxes = r.text.count("lsda_dataset_file_checkbox")
    return True, f"OK -- {sample_exp_id} returned {n_checkboxes // 2} file refs"


def preflight_disk():
    """Check free disk space in the downloads directory."""
    log("PRE-FLIGHT: Disk space")
    try:
        usage = shutil.disk_usage(DIR_RAW)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        log(f"  Free: {free_gb:.1f} GB  |  Total: {total_gb:.1f} GB")
        if free_gb < MIN_FREE_DISK_GB:
            log(f"  WARNING: Less than {MIN_FREE_DISK_GB} GB free. Bulk run may exhaust disk.")
            return False
        return True
    except OSError as e:
        log(f"  WARNING: could not check disk usage: {e}")
        return True  # don't block


# ============================================================
# CATALOG LOADING
# ============================================================

def load_catalog():
    """Load the most recent filtered catalog."""
    candidates = sorted(DIR_SCRAPES.glob("nlsp_catalog_bedrest_floor_*.json"))
    if not candidates:
        log(f"ERROR: No catalog found in {DIR_SCRAPES}/")
        log("Run script 03 (any version) first to generate the catalog.")
        sys.exit(1)
    latest = candidates[-1]
    log(f"Loaded catalog: {latest.name}")
    records = json.loads(latest.read_text())

    # Validate _key presence
    n_with_key = sum(1 for r in records if r.get("_key"))
    log(f"  Records: {len(records)}  |  with _key: {n_with_key}")
    if n_with_key == 0:
        log("ERROR: No records have _key. Re-run script 03_lsda_bulk_pull_v3.py to scrape fresh.")
        sys.exit(1)
    return [r for r in records if r.get("_key")]


def filter_to_campaign(records, campaign):
    if not campaign:
        return records
    target = f"UTMB Campaign {campaign}"
    out = []
    for r in records:
        miss_text = r.get("missions_text") or ""
        miss_list = r.get("missions") or []
        miss_str = "; ".join(str(m) for m in miss_list) if isinstance(miss_list, list) else str(miss_list)
        if target in f"{miss_text} {miss_str}":
            out.append(r)
    log(f"Filtered to {target}: {len(out)} experiments")
    return out


# ============================================================
# RECORDTABLES HTML FETCH + PARSE
# ============================================================

INPUT_PATTERN = re.compile(
    r'<input[^>]*class="[^"]*lsda_dataset_file_checkbox[^"]*"[^>]*>',
    re.IGNORECASE,
)
FILE_PATH_PATTERN = re.compile(r'file_path="([^"]+)"')
S3_GUID_PATTERN = re.compile(r's3_file_guid="([^"]+)"')


def fetch_recordtables(session, key, exp_id, cookie_header):
    """GET the recordtables HTML for one experiment. Returns HTML string or None on failure."""
    url = RECORDTABLES_URL.format(alias=ALIAS, record_type=RECORD_TYPE, key=key)
    referer = VIEW_URL.format(alias=ALIAS, record_type=RECORD_TYPE, key=key)
    headers = {
        "Accept": "text/html, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": cookie_header,
        "Referer": referer,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }
    for attempt in range(RETRY_COUNT + 1):
        try:
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
            log(f"    recordtables HTTP {r.status_code} for {exp_id} (attempt {attempt+1})")
        except requests.exceptions.RequestException as e:
            log(f"    recordtables error for {exp_id} (attempt {attempt+1}): {e}")
        if attempt < RETRY_COUNT:
            time.sleep(2)
    return None


def parse_recordtables_html(html):
    """Extract folder->CUID mapping from recordtables HTML.
    Returns dict: {folder_name: [cuid1, cuid2, ...]}"""
    folders = defaultdict(list)
    for input_tag in INPUT_PATTERN.findall(html):
        fp_m = FILE_PATH_PATTERN.search(input_tag)
        sg_m = S3_GUID_PATTERN.search(input_tag)
        if fp_m and sg_m:
            folders[fp_m.group(1)].append(sg_m.group(1))
    return dict(folders)


# ============================================================
# DOWNLOAD VIA ZIPS3
# ============================================================

ZIP_MAGIC = b"PK\x03\x04"


def write_experiment_inventory(exp_id, exp_dir, expected_folders):
    """Write a self-describing _inventory.json into the experiment's download dir.

    Shows: experiment ID, last update timestamp, every dataset folder this experiment
    has, file counts, on-disk ZIP/dir presence, ZIP sha256 + size, file-type breakdown
    from inside the ZIP, campaign tagging (explicit + alias-inferred), and floor-scope
    membership. Re-written on every successful download in this experiment (so partial-
    state inventories are always current).
    """
    inv_path = exp_dir / "_inventory.json"
    folders_state = []
    for folder_name, num_files in expected_folders:
        zip_path = exp_dir / f"{safe_filename(folder_name)}.zip"
        extracted_dir = exp_dir / folder_name

        zip_exists = zip_path.exists()
        zip_size = zip_path.stat().st_size if zip_exists else 0
        zip_valid = zipfile.is_zipfile(zip_path) if zip_exists else False

        # Compute sha256 only for small ZIPs (keeps this cheap during bulk runs)
        sha = ""
        if zip_exists and zip_valid and zip_size < 50 * 1024 * 1024:
            try:
                h = hashlib.sha256()
                with open(zip_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                sha = h.hexdigest()
            except OSError:
                pass

        # File-type breakdown from inside the ZIP (no filesystem walk needed)
        file_counts_by_ext = zip_file_extension_counts(zip_path) if zip_valid else {}

        # Campaign tagging
        explicit_campaigns, inferred_campaigns = detect_campaigns_in_folder_name(folder_name)
        all_campaigns = sorted(set(explicit_campaigns + [i["campaign"] for i in inferred_campaigns]))
        in_floor_scope = bool(set(all_campaigns) & set(FLOOR_SCOPE_CAMPAIGNS))

        folders_state.append({
            "folder": folder_name,
            "num_files_expected": num_files,
            "zip_present": zip_exists,
            "zip_valid": zip_valid,
            "zip_size_bytes": zip_size,
            "zip_sha256": sha,
            "extracted_dir_present": extracted_dir.exists() and extracted_dir.is_dir(),
            "file_counts_by_ext": file_counts_by_ext,
            "campaigns_explicit": explicit_campaigns,
            "campaigns_inferred": inferred_campaigns,
            "campaigns_all": all_campaigns,
            "in_floor_scope": in_floor_scope,
        })

    # Experiment-level rollups
    all_explicit = sorted(set(c for f in folders_state for c in f["campaigns_explicit"]))
    all_aliases_used = sorted(set(i["source"] for f in folders_state for i in f["campaigns_inferred"]))
    folders_in_scope = [f for f in folders_state if f["in_floor_scope"]]

    # Aggregate file-type counts across all folders
    total_by_ext = {}
    for f in folders_state:
        for ext, count in f["file_counts_by_ext"].items():
            total_by_ext[ext] = total_by_ext.get(ext, 0) + count

    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "experiment_id": exp_id,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "generated_by": SCRIPT_VERSION,
        "campaign_floor_scope": FLOOR_SCOPE_CAMPAIGNS,
        "experiment_campaigns_explicit": all_explicit,
        "experiment_campaign_aliases_used": all_aliases_used,
        "num_folders_total": len(expected_folders),
        "num_folders_with_valid_zip": sum(1 for f in folders_state if f["zip_valid"]),
        "num_folders_extracted": sum(1 for f in folders_state if f["extracted_dir_present"]),
        "num_folders_in_floor_scope": len(folders_in_scope),
        "total_files_expected": sum(f["num_files_expected"] for f in folders_state),
        "total_zip_bytes": sum(f["zip_size_bytes"] for f in folders_state),
        "total_files_by_ext": total_by_ext,
        "folders": folders_state,
    }

    inv_path.write_text(json.dumps(inventory, indent=2))


def detect_campaigns_in_folder_name(folder_name):
    """Return (explicit_campaigns, inferred_campaigns_with_provenance) for a folder name.

    explicit_campaigns: sorted list of ints from explicit 'Campaign_N' / 'CampaignN' patterns.
    inferred_campaigns_with_provenance: list of dicts {"campaign": int, "source": "alias:NAME"}.

    Note: explicit and inferred are kept SEPARATE so downstream code can decide whether
    to trust alias-based inference. CAMPAIGN_ALIASES is the only source of inferred mappings.
    """
    explicit = set()
    inferred = []

    # Explicit pattern: Campaign_1, Campaign 1, Campaign1, campaign_01, etc.
    for m in re.finditer(r"campaign[_\s]?(\d{1,2})", folder_name, re.IGNORECASE):
        explicit.add(int(m.group(1)))

    # Aliases (e.g., CFT70 -> Campaign 11)
    for alias, campaign in CAMPAIGN_ALIASES.items():
        if alias in folder_name:
            inferred.append({"campaign": campaign, "source": f"alias:{alias}"})

    return sorted(explicit), inferred


def zip_file_extension_counts(zip_path):
    """Return dict of file-extension -> count from a ZIP's central directory.
    Cheap (no extraction needed). Skips directory entries and __MACOSX cruft."""
    counts = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                if "__MACOSX" in name or name.startswith("."):
                    continue
                ext = Path(name).suffix.lower() or "(no_ext)"
                counts[ext] = counts.get(ext, 0) + 1
    except (zipfile.BadZipFile, OSError):
        pass
    return counts


def download_folder_zip(session, folder_name, cuids, exp_id, key, cookie_header, out_path):
    """POST to zips3/ and stream the ZIP to out_path. Returns (success_bool, bytes_written, sha256)."""
    payload = {"s3FilesGUID": [{"folder": folder_name, "__s3_file_guid": cuids}]}
    referer = VIEW_URL.format(alias=ALIAS, record_type=RECORD_TYPE, key=key)
    headers = {
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Cookie": cookie_header,
        "Origin": "https://nlsp.nasa.gov",
        "Referer": referer,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    }

    def attempt_one():
        """Single download attempt. Returns (success, size, sha) or (False, 0, None) on any failure."""
        try:
            r = session.post(ZIPS3_URL, json=payload, headers=headers,
                             stream=True, timeout=DOWNLOAD_TIMEOUT)
        except requests.exceptions.RequestException as e:
            log(f"    POST exception: {e}")
            return (False, 0, None)

        if r.status_code != 200:
            log(f"    POST returned HTTP {r.status_code}")
            return (False, 0, None)

        content_type = r.headers.get("Content-Type", "")
        if "stream" not in content_type and "zip" not in content_type:
            log(f"    Unexpected Content-Type: {content_type}")
            return (False, 0, None)

        h = hashlib.sha256()
        size = 0
        first_chunk = True
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if first_chunk:
                    if not chunk.startswith(ZIP_MAGIC):
                        log(f"    Response is not a ZIP (first 4 bytes: {chunk[:4]!r})")
                        f.close()
                        out_path.unlink(missing_ok=True)
                        return (False, 0, None)
                    first_chunk = False
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    log(f"    ABORT: ZIP exceeds {MAX_FILE_SIZE_MB} MB")
                    f.close()
                    out_path.unlink(missing_ok=True)
                    return (False, 0, None)
                h.update(chunk)
                f.write(chunk)
        return (True, size, h.hexdigest())

    for attempt in range(RETRY_COUNT + 1):
        if attempt > 0:
            log(f"    Retry attempt {attempt+1}/{RETRY_COUNT+1}")
            time.sleep(3)
        result = attempt_one()
        if result[0]:
            return result
    return (False, 0, None)


# ============================================================
# INVENTORY PHASE
# ============================================================

def build_inventory(session, catalog, cookie_header):
    """Build complete inventory of (exp_id, folder, num_files, cuids) across all experiments."""
    log("=" * 60)
    log("INVENTORY: Fetching recordtables for all experiments")
    log("=" * 60)
    inventory = []
    failures = []
    for i, rec in enumerate(catalog, start=1):
        exp_id = rec.get("experiment_id", "unknown")
        key = rec.get("_key")
        log(f"  [{i}/{len(catalog)}] {exp_id} (key={key[:8]}...)")
        html = fetch_recordtables(session, key, exp_id, cookie_header)
        if not html:
            failures.append({"experiment_id": exp_id, "key": key, "reason": "recordtables_fetch_failed"})
            continue
        folders = parse_recordtables_html(html)
        if not folders:
            failures.append({"experiment_id": exp_id, "key": key, "reason": "no_file_checkboxes_in_html"})
            log(f"    WARN: 0 folders extracted")
            continue
        for folder_name, cuids in folders.items():
            inventory.append({
                "experiment_id": exp_id,
                "key": key,
                "folder": folder_name,
                "num_files": len(cuids),
                "cuids": cuids,
            })
        log(f"    Found {len(folders)} folders, {sum(len(v) for v in folders.values())} files")
        time.sleep(SLEEP_RECORDTABLES)

    # Save inventory
    inv_path = DIR_MANIFESTS / f"dataset_inventory_{TIMESTAMP}.csv"
    with inv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "key", "folder", "num_files"])
        writer.writeheader()
        for item in inventory:
            writer.writerow({k: v for k, v in item.items() if k != "cuids"})
    log(f"\nInventory saved: {inv_path}")
    log(f"  Total folders to download: {len(inventory)}")
    log(f"  Total files across all folders: {sum(i['num_files'] for i in inventory)}")
    log(f"  Failures: {len(failures)}")
    if failures:
        fail_path = DIR_MANIFESTS / f"inventory_failures_{TIMESTAMP}.json"
        fail_path.write_text(json.dumps(failures, indent=2))
        log(f"  Failures logged: {fail_path}")
    return inventory


# ============================================================
# BULK DOWNLOAD PHASE
# ============================================================

def bulk_download(session, inventory, cookie_header, resume=False):
    log("=" * 60)
    log("BULK DOWNLOAD")
    log("=" * 60)

    manifest_path = DIR_MANIFESTS / f"download_manifest_{TIMESTAMP}.csv"
    log(f"Manifest: {manifest_path}")

    fields = ["experiment_id", "folder", "num_files", "http_status",
              "local_path", "sha256", "size_bytes", "fetch_timestamp", "notes"]

    n_ok = 0
    n_fail = 0
    n_skip = 0
    consecutive_failures = 0
    started = time.time()
    total_bytes = 0

    # Pre-compute per-experiment folder lists for inventory writing
    exp_folders = defaultdict(list)
    for item in inventory:
        exp_folders[item["experiment_id"]].append((item["folder"], item["num_files"]))

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, item in enumerate(inventory, start=1):
            exp_id = item["experiment_id"]
            folder = item["folder"]
            cuids = item["cuids"]
            key = item["key"]
            exp_dir = DIR_RAW / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            out_path = exp_dir / f"{safe_filename(folder)}.zip"

            log(f"[{i}/{len(inventory)}] {exp_id} | {folder} ({len(cuids)} files)")

            # Determine action: skip via resume, or attempt download.
            # Both paths fall through to the same post-processing block below.
            action = None  # one of: "skipped", "downloaded", "failed"

            if resume and out_path.exists() and out_path.stat().st_size > 0:
                if zipfile.is_zipfile(out_path):
                    log(f"    SKIP (resume, valid ZIP): {out_path.name} ({out_path.stat().st_size:,} bytes)")
                    writer.writerow({
                        "experiment_id": exp_id, "folder": folder, "num_files": len(cuids),
                        "http_status": "skipped_resume",
                        "local_path": str(out_path), "sha256": "",
                        "size_bytes": out_path.stat().st_size,
                        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                        "notes": "skipped via --resume (ZIP integrity verified)",
                    })
                    n_skip += 1
                    action = "skipped"
                else:
                    log(f"    Existing ZIP is corrupt, re-downloading: {out_path.name}")
                    out_path.unlink()

            if action is None:
                # Either no --resume, or no existing ZIP, or it was corrupt — attempt download
                success, size, sha = download_folder_zip(
                    session, folder, cuids, exp_id, key, cookie_header, out_path
                )
                if success:
                    n_ok += 1
                    total_bytes += size
                    log(f"    OK {out_path.name} | {size:,} bytes")
                    writer.writerow({
                        "experiment_id": exp_id, "folder": folder, "num_files": len(cuids),
                        "http_status": 200,
                        "local_path": str(out_path), "sha256": sha, "size_bytes": size,
                        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                        "notes": "",
                    })
                    action = "downloaded"
                else:
                    n_fail += 1
                    log(f"    FAIL")
                    writer.writerow({
                        "experiment_id": exp_id, "folder": folder, "num_files": len(cuids),
                        "http_status": "failed",
                        "local_path": "", "sha256": "", "size_bytes": 0,
                        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                        "notes": "all retries exhausted",
                    })
                    action = "failed"

            f.flush()  # keep manifest fresh in case of interrupt

            # Circuit breaker counter — only failures count; skip and download both reset
            if action == "failed":
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            # Write/update _inventory.json after EVERY folder (skip OR download OR fail)
            # so partial state is always captured, even on pure-resume runs
            try:
                write_experiment_inventory(exp_id, exp_dir, exp_folders[exp_id])
            except Exception as e:
                log(f"    WARN: failed to write _inventory.json for {exp_id}: {e}")

            # Circuit breaker: abort if too many back-to-back failures (likely cookie expired)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log("=" * 60)
                log(f"CIRCUIT BREAKER: {MAX_CONSECUTIVE_FAILURES} consecutive download failures.")
                log("Likely cause: session cookie expired. Aborting.")
                log("Recovery:")
                log("  1. Re-extract cookie from browser DevTools (cURL copy)")
                log("  2. Update 02_scrapes/cookie.txt")
                log(f"  3. Re-run with --resume to continue from current state")
                log("=" * 60)
                break

            # Polite sleep only if we actually made a network call
            if action != "skipped":
                time.sleep(SLEEP_DOWNLOAD)

            if i % 10 == 0:
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(inventory) - i) / rate if rate > 0 else 0
                log(f"  Progress: {i}/{len(inventory)} | OK {n_ok} | FAIL {n_fail} | "
                    f"SKIP {n_skip} | {total_bytes / 1024**2:.1f} MB | "
                    f"elapsed {elapsed/60:.1f}min | ETA {eta/60:.1f}min")

    log("=" * 60)
    log(f"BULK COMPLETE")
    log(f"  OK: {n_ok}  |  Failed: {n_fail}  |  Skipped: {n_skip}")
    log(f"  Total bytes: {total_bytes:,} ({total_bytes / 1024**2:.1f} MB)")
    log(f"  Manifest: {manifest_path}")
    return n_ok, n_fail, n_skip


# ============================================================
# UNZIP PHASE
# ============================================================

def get_zip_top_dir(zip_path):
    """Return the single top-level directory inside a ZIP, or None if no clear single top dir.
    Most NLSP zips have one top-level dir matching the ZIP filename."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
        top_dirs = set(n.split("/")[0] for n in names if "/" in n)
        if len(top_dirs) == 1:
            return list(top_dirs)[0]
    except zipfile.BadZipFile:
        pass
    return None


def unzip_all(force=False):
    """Extract all ZIPs in 03_raw_downloads/.

    For ZIPs with a single top-level directory (typical NLSP case), extract to the
    ZIP's parent dir so the resulting structure is parent/<top>/<files> -- not the
    double-nested parent/<top>/<top>/<files>.

    Skips ZIPs whose target directory already exists and is non-empty,
    unless force=True (then deletes existing and re-extracts).
    """
    log("=" * 60)
    log("UNZIP")
    log("=" * 60)
    n_unzipped = 0
    n_skipped = 0
    n_bad = 0
    n_force = 0
    for zip_path in sorted(DIR_RAW.rglob("*.zip")):
        if not zipfile.is_zipfile(zip_path):
            log(f"  BAD ZIP (not a valid archive): {zip_path.name}")
            n_bad += 1
            continue

        top = get_zip_top_dir(zip_path)
        if top:
            target_dir = zip_path.parent / top
            extract_to = zip_path.parent
        else:
            # ZIP has no single top-level dir; extract under a directory named
            # after the ZIP itself
            target_dir = zip_path.with_suffix("")
            extract_to = target_dir

        if target_dir.exists() and target_dir.is_dir() and any(target_dir.iterdir()):
            if force:
                shutil.rmtree(target_dir)
                n_force += 1
            else:
                n_skipped += 1
                continue

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_to)
            n_unzipped += 1
            log(f"  Unzipped: {zip_path.name} -> {target_dir.name}/")
        except zipfile.BadZipFile:
            n_bad += 1
            log(f"  BAD ZIP during extract: {zip_path.name}")

    log(f"Unzipped: {n_unzipped}  |  Skipped: {n_skipped}  |  Re-extracted (force): {n_force}  |  Bad: {n_bad}")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bulk", action="store_true", help="Actually download (default: inventory only)")
    p.add_argument("--inventory-only", action="store_true", help="Only build inventory, skip downloads")
    p.add_argument("--resume", action="store_true", help="Skip ZIPs that already exist (verifies integrity)")
    p.add_argument("--unzip", action="store_true", help="Unzip after download")
    p.add_argument("--unzip-only", action="store_true",
                   help="Skip catalog/inventory/download phases entirely; just unzip every ZIP in 03_raw_downloads/")
    p.add_argument("--reunzip", action="store_true",
                   help="Used with --unzip or --unzip-only: delete existing extracted dirs and re-extract")
    p.add_argument("--campaign", type=int, choices=[1, 3, 11], help="Restrict to one campaign")
    p.add_argument("--only-exp", type=str, help="Restrict to one experiment ID (e.g. MR080G)")
    args = p.parse_args()

    mkdirs()

    # Fast path: just unzip everything that's already on disk
    if args.unzip_only:
        log(f"--unzip-only: skipping catalog/inventory/download phases")
        unzip_all(force=args.reunzip)
        return

    cookie_header = load_cookie_header()
    log(f"Cookie loaded ({len(cookie_header)} chars)")

    catalog = load_catalog()
    catalog = filter_to_campaign(catalog, args.campaign)
    if args.only_exp:
        catalog = [r for r in catalog if r.get("experiment_id") == args.only_exp]
        log(f"Filtered to experiment {args.only_exp}: {len(catalog)} record(s)")

    if not catalog:
        log("ERROR: No experiments to process after filtering.")
        sys.exit(1)

    session = requests.Session()
    log(f"Priming session: {PRIME_URL}")
    try:
        session.get(PRIME_URL, headers={"Cookie": cookie_header}, timeout=30)
    except requests.exceptions.RequestException as e:
        log(f"  Prime warning (continuing anyway): {e}")

    # === PRE-FLIGHT CHECKS ===
    log("=" * 60)
    log("PRE-FLIGHT CHECKS")
    log("=" * 60)

    # Cookie validation against a sample experiment
    sample_rec = catalog[0]
    cookie_ok, cookie_msg = preflight_cookie(
        session, cookie_header, sample_rec["_key"], sample_rec.get("experiment_id", "?")
    )
    if cookie_ok:
        log(f"  Cookie: {cookie_msg}")
    else:
        log(f"  Cookie: FAIL -- {cookie_msg}")
        log("")
        log("  ABORTING. To fix:")
        log("  1. Open NLSP in your browser, log into a session if needed")
        log("  2. Open DevTools -> Network tab")
        log("  3. Reload an experiment page, find any xhr/fetch request")
        log("  4. Right-click -> Copy -> Copy as cURL, save to ~/Downloads/test_download.sh")
        log("  5. Re-run the cookie extraction one-liner that wrote 02_scrapes/cookie.txt")
        log("  6. Re-run this script with --resume")
        sys.exit(2)

    # Disk space check
    if args.bulk:
        preflight_disk()

    # === INVENTORY PHASE ===
    inventory = build_inventory(session, catalog, cookie_header)

    if not inventory:
        log("ERROR: Empty inventory. Cannot proceed.")
        sys.exit(1)

    if args.inventory_only:
        log("--inventory-only: stopping after inventory build")
        return

    if args.bulk:
        bulk_download(session, inventory, cookie_header, resume=args.resume)
        if args.unzip:
            unzip_all(force=args.reunzip)
    else:
        log("=" * 60)
        log("DRY-RUN ONLY")
        log("=" * 60)
        log(f"Would download {len(inventory)} folder ZIPs across "
            f"{len(set(i['experiment_id'] for i in inventory))} experiments.")
        log(f"Total files: {sum(i['num_files'] for i in inventory)}")
        log("")
        log("To actually download:")
        log("  python3 04_lsda_pull_via_zips3.py --bulk --unzip")
        log("To resume after interrupt:")
        log("  python3 04_lsda_pull_via_zips3.py --bulk --resume --unzip")
        log("To test on one experiment first:")
        log("  python3 04_lsda_pull_via_zips3.py --bulk --only-exp MR080G")
        log("To re-extract existing ZIPs without re-downloading:")
        log("  python3 04_lsda_pull_via_zips3.py --unzip-only --reunzip")


if __name__ == "__main__":
    sys.exit(main() or 0)
