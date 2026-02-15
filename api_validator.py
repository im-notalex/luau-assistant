import json
import os
import threading
import re
from typing import Optional

try:
    import requests
except Exception:
    requests = None

_lock = threading.Lock()
_loaded = False
_property_names = set()
_loaded_path = None


def _candidate_paths(root: str):
    # Common filenames first
    common = []
    # Environment override comes first if set
    try:
        env_path = os.environ.get('API_DUMP_PATH')
        if env_path:
            common.append(env_path)
    except Exception:
        pass
    common += [
        os.path.join(root, 'Docs', 'api_dump.json'),
        os.path.join(root, 'Docs', 'api_dump.latest.json'),
        os.path.join(root, 'Docs', 'api_dump_min.json'),
        os.path.join(root, 'Docs', 'ApiDump.json'),
        os.path.join(root, 'Docs', 'API-Dump.json'),
        # Explicit nested location requested
        os.path.join(root, 'Docs', 'Api', 'ApiDump.json'),
        # Absolute path requested by user (Windows)
        r'D:\\luau_rag\\Docs\\Api\\ApiDump.json',
        r'D:\luau_rag\Docs\Api\ApiDump.json',
    ]
    # Fall back to a loose recursive scan for any file name containing both 'api' and 'dump'
    try:
        for dirpath, _, files in os.walk(os.path.join(root, 'Docs')):
            for f in files:
                low = f.lower()
                if low.endswith('.json') and ('api' in low and 'dump' in low):
                    common.append(os.path.join(dirpath, f))
    except Exception:
        pass
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in common:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_api_dump(root: str) -> bool:
    """Load a Roblox API dump JSON into memory. Returns True if loaded."""
    global _loaded, _property_names, _loaded_path
    if _loaded:
        return True
    with _lock:
        if _loaded:
            return True
        for path in _candidate_paths(root):
            if not os.path.exists(path):
                continue
            # Skip empty files
            try:
                if os.path.isfile(path) and os.path.getsize(path) == 0:
                    continue
            except Exception:
                pass
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    data = json.load(fh)
                props = set()
                # Support common formats: {"Classes":[{"Name":"Humanoid","Members":[{"MemberType":"Property","Name":"MoveDirection"}, ...]}]}
                classes = data.get('Classes') or []
                for c in classes:
                    for m in c.get('Members', []) or []:
                        if (m.get('MemberType') or '').lower() == 'property':
                            name = m.get('Name')
                            if isinstance(name, str) and name:
                                props.add(name)
                _property_names = props
                _loaded = True
                _loaded_path = path
                return True
            except Exception:
                continue
        return False


def validate_luau_code_blocks(text: str, root: str):
    """Return notes about likely API misuse based on the API dump.
    Currently flags use of known Property names as if they were methods (e.g., Foo:MoveDirection()).
    """
    try:
        if not text:
            return []
        if not load_api_dump(root):
            return []
        notes = []
        # Find method calls: .Name( or :Name(
        for m in re.finditer(r"[\.:]\s*([A-Za-z_]\w*)\s*\(", text):
            name = m.group(1)
            if name in _property_names:
                notes.append(f"'{name}' appears to be a Property in the Roblox API; avoid calling it as a function. Use property access or the appropriate method instead.")
        return notes
    except Exception:
        return []


def get_stats():
    """Return a dict with load status and counts for UI/debugging."""
    return {
        "loaded": _loaded,
        "property_count": len(_property_names) if _loaded else 0,
        "path": _loaded_path,
    }


def ensure_api_dump(root: str) -> Optional[str]:
    """Ensure an API dump exists and is parseable. If missing or empty, attempt download.
    Returns the path if available after ensure, else None.
    """
    # If already loaded, return the loaded path
    if load_api_dump(root):
        return _loaded_path

    # Try to download if requests is available
    targets = [
        # Preferred target location
        os.path.join(root, 'Docs', 'Api', 'ApiDump.json')
    ]
    sources = [
        # Known community-maintained API JSON (RBXAPI)
        'https://anaminus.github.io/rbx/json/api/latest.json',
        # Mirrors in popular trackers (names can change over time)
        'https://raw.githubusercontent.com/MaximumADHD/Roblox-Client-Tracker/roblox/API-Dump.json',
        'https://raw.githubusercontent.com/CloneTrooper1019/Roblox-Client-Tracker/roblox/API-Dump.json',
        'https://raw.githubusercontent.com/CloneTrooper1019/Roblox-Client-Tracker/roblox/api-dump.json',
    ]
    if requests is None:
        return None
    # Ensure folder exists
    try:
        os.makedirs(os.path.dirname(targets[0]), exist_ok=True)
    except Exception:
        pass
    # Attempt sources in order
    for url in sources:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            # Basic validation
            if not isinstance(data, dict):
                continue
            # Save to target
            with open(targets[0], 'w', encoding='utf-8') as fh:
                json.dump(data, fh, ensure_ascii=False)
            # Reload
            if load_api_dump(root):
                return _loaded_path
        except Exception:
            continue
    return None


if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.abspath(__file__))
    # Ensure presence; download if missing
    path = ensure_api_dump(repo_root)
    if not path:
        # Try load again to surface any existing non-empty file
        load_api_dump(repo_root)
    s = get_stats()
    if s.get('loaded'):
        print(f"Loaded API dump: {s.get('path')} ({s.get('property_count',0)} properties)")
    else:
        print("API dump not detected (place ApiDump.json under Docs/Api or set API_DUMP_PATH).")
