"""
setup_verify.py
----------------
Fully automated setup verifier for your LuaU RAG environment.
- Auto-elevates to admin if needed
- Ensures required folders exist
- Installs missing tools via winget
- Installs Python libs via pip
- Downloads Roblox DevHub and Luau documentation
- Checks for a local model
"""

import os
import sys
import subprocess
import ctypes
import shutil
from datetime import datetime, timezone

# =========================
# CONFIGURATION
# =========================
ROOT = os.path.dirname(__file__)
REQUIRED_FOLDERS = [
    os.path.join(ROOT, "Docs"),
    os.path.join(ROOT, "Docs", "html"),
    os.path.join(ROOT, "Docs", "markdown"),
    os.path.join(ROOT, "chroma"),
    os.path.join(ROOT, "models")
]
# Only require tools actually used
REQUIRED_TOOLS = []
LLM_BACKEND = "ollama"  # change to "llama.cpp" if using that locally
if LLM_BACKEND == "ollama":
    REQUIRED_TOOLS.append("ollama")
else:
    REQUIRED_TOOLS.append("llama.cpp")

PYTHON_PACKAGES = ["chromadb", "sentence-transformers", "beautifulsoup4", "markdownify", "requests"]

# Documentation sources
LUAU_ORG_DOCS = [
    # Main Luau documentation
    "https://luau.org/",
    "https://luau.org/compatibility",
    "https://luau.org/syntax",
    "https://luau.org/analysis",
    
    # Luau-lang.org docs
    "https://luau-lang.org/",
    "https://luau-lang.org/getting-started",
    "https://luau-lang.org/syntax",
    "https://luau-lang.org/performance",
    "https://luau-lang.org/typecheck",
    "https://luau-lang.org/optimizations",
    "https://luau-lang.org/compatibility",
    "https://luau-lang.org/profile",
    "https://luau-lang.org/analysis",
    
    # Original Lua docs for reference
    "https://www.lua.org/manual/5.1/manual.html",
    "https://www.lua.org/pil/contents.html"
]

ROBLOX_DOCS = [
    # Core Luau Documentation
    "https://create.roblox.com/docs/luau",
    "https://create.roblox.com/docs/luau/strings",
    "https://create.roblox.com/docs/luau/numbers",
    "https://create.roblox.com/docs/luau/booleans",
    "https://create.roblox.com/docs/luau/functions",
    "https://create.roblox.com/docs/luau/tables",
    "https://create.roblox.com/docs/luau/functions",
    "https://create.roblox.com/docs/luau/control-flow",
    "https://create.roblox.com/docs/luau/scope",
    "https://create.roblox.com/docs/luau/metatables",
    
    # Scripting Documentation
    "https://create.roblox.com/docs/scripting",
    "https://create.roblox.com/docs/scripting/scripts",
    "https://create.roblox.com/docs/scripting/luau",
    "https://create.roblox.com/docs/scripting/events",
    "https://create.roblox.com/docs/scripting/services",
    "https://create.roblox.com/docs/scripting/networking",
    "https://create.roblox.com/docs/scripting/debugging",
    "https://create.roblox.com/docs/scripting/security",
    
    # Engine Reference
    "https://create.roblox.com/docs/reference/engine",
    "https://create.roblox.com/docs/reference/engine/classes",
    "https://create.roblox.com/docs/reference/engine/datatypes",
    "https://create.roblox.com/docs/reference/engine/enums",
    
    # Common Classes
    "https://create.roblox.com/docs/reference/engine/classes/DataModel",
    "https://create.roblox.com/docs/reference/engine/classes/Workspace",
    "https://create.roblox.com/docs/reference/engine/classes/Players",
    "https://create.roblox.com/docs/reference/engine/classes/ReplicatedStorage",
    "https://create.roblox.com/docs/reference/engine/classes/Player",
    "https://create.roblox.com/docs/reference/engine/classes/Part",
    "https://create.roblox.com/docs/reference/engine/classes/Model",
    "https://create.roblox.com/docs/reference/engine/classes/BasePart",
    "https://create.roblox.com/docs/reference/engine/classes/Instance"
]

# Combined documentation URLs
DEVHUB_URLS = LUAU_ORG_DOCS + ROBLOX_DOCS

# =========================
# DOCUMENTATION DOWNLOAD
# =========================
def download_devhub_docs():
    """Download and convert Roblox DevHub and Luau documentation to markdown."""
    print("\n📚 Downloading documentation...")
    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
    # Ensure we have necessary packages
    for pkg in ["requests", "beautifulsoup4", "markdownify"]:
        try:
            if pkg == "beautifulsoup4":
                __import__("bs4")
            elif pkg == "markdownify":
                __import__(pkg)
            else:
                __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

    # Now import after ensuring installation
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify
    import time
    import re

    def sanitize_filename(name):
        """Clean up filename for Windows compatibility."""
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    docs_dir = os.path.join(ROOT, "Docs")
    md_dir = os.path.join(docs_dir, "markdown")
    html_dir = os.path.join(docs_dir, "html")
    
    # Ensure directories exist
    for d in [docs_dir, md_dir, html_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache'
    }

    processed_urls = set()
    urls_to_process = DEVHUB_URLS.copy()
    error_count = 0

    while urls_to_process and error_count < 5:  # Allow some errors before giving up
        url = urls_to_process.pop(0)
        if url in processed_urls:
            continue

        try:
            print(f"\n  📥 Fetching: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save raw HTML
            domain = url.split('/')[2]  # e.g., create.roblox.com or luau-lang.org
            page_name = url.split('/')[-1] if url.split('/')[-1] else 'index'
            html_filename = sanitize_filename(f"{domain}_{page_name}.html")
            html_path = os.path.join(html_dir, html_filename)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"  💾 Saved HTML: {html_filename}")

            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')
            content = None

            # Get main content based on the site
            if "luau.org" in url or "luau-lang.org" in url:
                for selector in [
                    'article', 
                    'main', 
                    '.content',
                    '.document',
                    '#content'
                ]:
                    content = soup.select_one(selector)
                    if content:
                        print(f"  ✓ Found Luau.org content using: {selector}")
                        break
            elif "lua.org" in url:
                for selector in [
                    '#content',
                    'main',
                    'body > table > tr > td:nth-child(2)',  # For Lua PIL
                    'body > .content'  # For Lua manual
                ]:
                    content = soup.select_one(selector)
                    if content:
                        print(f"  ✓ Found Lua.org content using: {selector}")
                        break
            else:  # Roblox DevHub
                for selector in [
                    'main[class*="docs-body"]',
                    'div[class*="markdown"]',
                    'div[class*="docContent"]',
                    'div[class*="content"]',
                    'article[class*="docs"]',
                    'div.api-reference',
                    'article',
                    'main'
                ]:
                    content = soup.select_one(selector)
                    if content:
                        print(f"  ✓ Found DevHub content using: {selector}")
                        break
    ## psst, did you know this was made by im.notalex on discord, linked HERE? https://boogerboys.github.io/alex/clickme.html :)
            if not content:
                print(f"  ⚠ No content found using standard selectors in {url}")
                # Try finding content by structure (backup method)
                for elem in soup.find_all(['article', 'main', 'div']):
                    if elem.get_text().strip() and len(elem.get_text().split()) > 100:
                        content = elem
                        print("  ✓ Found content using backup method")
                        break

            if content:
                # Clean up content
                for tag in content.select('script, style, nav, footer, header, aside'):
                    tag.decompose()

                # Get title
                title = soup.find('h1')
                if title:
                    title = title.get_text(strip=True)
                else:
                    title = page_name.replace('-', ' ').title()

                # Convert to markdown
                md_content = markdownify(str(content))

                # Save markdown
                md_filename = sanitize_filename(f"{domain}_{page_name}.md")
                md_path = os.path.join(md_dir, md_filename)
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {title}\n")
                    f.write(f"Source: {url}\n\n")
                    f.write(md_content)
                print(f"  ✅ Saved markdown: {md_filename}")

                # Find more documentation links (only for Roblox DevHub)
                if "create.roblox.com" in url:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('/docs/'):
                            full_url = f"https://create.roblox.com{href}"
                            if (full_url not in processed_urls and 
                                full_url not in urls_to_process and
                                any(x in full_url for x in ['reference', 'luau', 'scripting', 'classes'])):
                                urls_to_process.append(full_url)
                                print(f"  📑 Found new link: {href}")
            else:
                print(f"  ⚠ No content found in {url}")
                print("  Available elements:", [
                    f"{tag.name}:{','.join(tag.get('class', []))}" 
                    for tag in soup.find_all(class_=True)
                ])
                error_count += 1

            processed_urls.add(url)
            time.sleep(2)  # Be nice to the server

        except Exception as e:
            print(f"  ⚠ Error processing {url}: {str(e)}")
            if hasattr(e, 'response'):
                print(f"  Status code: {e.response.status_code}")
                print(f"  Headers: {e.response.headers}")
            error_count += 1
            time.sleep(5)  # Wait longer after an error

    print(f"\n✅ Downloaded {len(processed_urls)} documentation pages")
    if error_count > 0:
        print(f"⚠ Encountered {error_count} errors during download")

# =========================
# ADMIN PRIVILEGE CHECK
# =========================
def ensure_admin():
    """Re-run the script as admin if not already."""
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False

    if not is_admin:
        print("⚠️  Admin privileges required. Requesting elevation...")
        params = " ".join([f'"{arg}"' for arg in sys.argv])
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, params, None, 1
            )
        except Exception as e:
            print(f"❌ Failed to elevate privileges: {e}")
        sys.exit(0)

# =========================
# ENVIRONMENT SETUP
# =========================
def check_or_create_folders():
    print("📁 Checking folders...")
    for folder in REQUIRED_FOLDERS:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"  ✅ Created missing folder: {folder}")
        else:
            print(f"  ✔ Folder exists: {folder}")

def install_tool(tool_name):
    """Try installing a missing tool using winget."""
    install_cmds = {
        "wget": "winget install --id GnuWin32.Wget -e --source winget -h",
        "pandoc": "winget install --id Pandoc.Pandoc -e --source winget -h",
        "ollama": "winget install Ollama.Ollama -h",
        "llama.cpp": "echo 'Please manually install llama.cpp from GitHub.'"
    }

    cmd = install_cmds.get(tool_name)
    if cmd:
        print(f"🛠 Installing {tool_name}...")
        subprocess.run(cmd, shell=True)

def check_tools():
    print("\n🧰 Checking required tools...")
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        if path:
            print(f"  ✔ Found {tool} at: {path}")
        else:
            print(f"  ⚠ Missing: {tool}")
            install_tool(tool)
    print("✅ Tool check complete.")

def check_python_libs():
    print("\n🐍 Checking Python libraries...")
    for pkg in PYTHON_PACKAGES:
        try:
            if pkg == "beautifulsoup4":
                __import__("bs4")
            else:
                __import__(pkg)
            print(f"  ✔ {pkg} installed")
        except ImportError:
            print(f"  ⚠ Missing Python package: {pkg}")
            print(f"   → Installing {pkg} via pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg])
    print("✅ Python package check complete.")

def check_model():
    print("\n🧠 Checking model folder...")
    model_dir = os.path.join(ROOT, "models")
    files = os.listdir(model_dir)
    found_model = any(f.endswith(".gguf") or "coder" in f.lower() for f in files)

    if found_model:
        print("  ✔ Local model found.")
    else:
        print("  ⚠ No local model found in /models.")
        if LLM_BACKEND == "ollama":
            ans = input("   → Would you like to pull DeepSeek-Coder:7b-instruct (≈7GB)? [y/N]: ").strip().lower()
            if ans == "y":
                subprocess.run(["ollama", "pull", "deepseek-coder:7b-instruct"])
        else:
            print("   → Please place your .gguf model file in the /models folder manually.")

# =========================
# MAIN
# =========================
def configure_git_credentials():
    """Configure Git to store credentials using the Windows Credential Manager (if available)."""
    try:
        subprocess.run(["git", "config", "--global", "credential.helper", "manager-core"], check=False)
        subprocess.run(["git", "config", "--global", "credential.useHttpPath", "true"], check=False)
        print("Configured Git credential helper (manager-core). Future GitHub auth will be stored.")
    except Exception as e:
        print(f"Warning: could not configure git credential helper: {e}")

def download_all_docs():
    """Download Luau + Roblox docs and selected repos into Docs/markdown using robust fetching."""
    # Ensure packages
    for pkg in ["requests", "beautifulsoup4", "markdownify"]:
        try:
            if pkg == "beautifulsoup4":
                __import__("bs4")
            elif pkg == "markdownify":
                __import__(pkg)
            else:
                __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    from urllib.parse import urlparse
    import time

    RATE_LIMIT_SECONDS = 1.0

    def make_session():
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"], raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        })
        return s

    SESSION = make_session()

    def sanitize_url_to_name(url: str) -> str:
        p = urlparse(url)
        name = f"{p.netloc}{p.path}".replace("/", "_")
        if not name.endswith(".html"):
            name += ".html"
        return name

    def fetch(url: str):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            r = SESSION.get(url, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            return None

    def html_to_markdown(html: str, url: str):
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return None
        text = md(str(main), heading_style="ATX", code_style="fenced", bullets="-")
        fm = (
            f"---\nsource: {url}\n"
            f"type: documentation\n"
            f"fetched_at: {datetime.now(timezone.utc).isoformat()}\n---\n\n"
        )
        return fm + text

    LUAU_SOURCES = [
        "https://luau-lang.org/syntax",
        "https://luau-lang.org/performance",
        "https://luau-lang.org/compatibility",
        "https://luau-lang.org/typecheck",
        "https://luau-lang.org/profile",
        "https://luau-lang.org/getting-started",
        "https://luau-lang.org/grammar",
    ]

    ROBLOX_DOCS = [
        "https://create.roblox.com/docs/reference/engine",
        "https://create.roblox.com/docs/reference/engine/datatypes",
        "https://create.roblox.com/docs/reference/engine/classes",
        "https://create.roblox.com/docs/reference/engine/enums",
        "https://create.roblox.com/docs/scripting/security",
    ]

    LUAU_GITHUB_DOCS = [
        "https://raw.githubusercontent.com/Roblox/luau/master/docs/grammar.md",
        "https://raw.githubusercontent.com/Roblox/luau/master/docs/typechecking.md",
        "https://raw.githubusercontent.com/Roblox/luau/master/docs/bytecode.md",
        "https://raw.githubusercontent.com/Roblox/luau/master/docs/builtins.md",
    ]

    GITHUB_REPOS = [
        "https://github.com/Sleitnick/Knit",
        "https://github.com/Sleitnick/Component",
        "https://github.com/Sleitnick/Comm",
        "https://github.com/MadStudioRoblox/ProfileService",
        "https://github.com/Roblox/roact",
    ]

    docs_dir = os.path.join(ROOT, "Docs")
    md_dir = os.path.join(docs_dir, "markdown")
    html_dir = os.path.join(docs_dir, "html")
    for d in [docs_dir, md_dir, html_dir]:
        os.makedirs(d, exist_ok=True)

    print("\nDownloading Luau and Roblox Documentation...")
    for url in LUAU_SOURCES + ROBLOX_DOCS:
        html = fetch(url)
        if not html:
            continue
        name = sanitize_url_to_name(url)
        with open(os.path.join(html_dir, name), "w", encoding="utf-8") as f:
            f.write(html)
        md_text = html_to_markdown(html, url)
        if md_text:
            save_path = os.path.join(md_dir, name.replace(".html", ".md"))
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(md_text)

    print("\nDownloading Luau GitHub Docs...")
    for url in LUAU_GITHUB_DOCS:
        text = fetch(url)
        if text:
            fm = (
                f"---\nsource: {url}\n"
                f"type: documentation\n"
                f"fetched_at: {datetime.now(timezone.utc).isoformat()}\n---\n\n"
            )
            out_path = os.path.join(md_dir, os.path.basename(url))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(fm + text)

    print("\nConfiguring Git credentials (to avoid repeated sign-ins)...")
    configure_git_credentials()

    def clone_or_update_repo(url: str) -> str:
        name = url.split("/")[-1]
        path = os.path.join(ROOT, "Docs", "repos", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            print(f"Updating repo {name} ...")
            subprocess.run(["git", "-C", path, "pull"], check=False)
        else:
            print(f"Cloning repo {name} ...")
            subprocess.run(["git", "clone", "--depth=1", url, path], check=False)
        return path

    def convert_repo_to_markdown(repo_url: str, repo_path: str):
        for root, _, files in os.walk(repo_path):
            for f in files:
                if not f.lower().endswith(".lua"):
                    continue
                file_path = os.path.join(root, f)
                rel = os.path.relpath(file_path, repo_path)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
                        code = fp.read()
                except Exception:
                    continue
                md_text = (
                    f"---\nsource: {repo_url}\nfile: {rel}\n"
                    f"type: repo_code\n"
                    f"fetched_at: {datetime.now(timezone.utc).isoformat()}\n---\n\n"
                    f"```lua\n{code}\n```\n"
                )
                safe_rel = rel.replace(os.sep, "_")
                out_path = os.path.join(ROOT, "Docs", "markdown", f"repo_{os.path.basename(repo_path)}_{safe_rel}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(md_text)

    print("\nCloning Roblox-Luau Frameworks...")
    for repo in GITHUB_REPOS:
        local = clone_or_update_repo(repo)
        if local:
            convert_repo_to_markdown(repo, local)

    print("\nAll documentation collected successfully!")
    print(f"Markdown directory: {os.path.join(ROOT, 'Docs', 'markdown')}")

def run_download_docs():
    """Deprecated wrapper: run integrated downloader instead."""
    print("Running integrated document downloader...")
    download_all_docs()

def main():
    ensure_admin()
    print("🔍 Starting full LuaU RAG setup verification...\n")
    try:
        check_or_create_folders()
        check_tools()
        check_python_libs()
        check_model()
        # Download documentation (integrated)
        download_all_docs()
        print("\n✅ Environment setup complete! Now you can run:")
        print("1. python embeddings_index.py  (to index the documentation)")
        print("2. python luau_rag.py         (to start the RAG system)")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nPress Enter to exit...")
        input()

if __name__ == "__main__":
    main()
