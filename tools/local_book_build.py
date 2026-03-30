#!/usr/bin/env python3
"""Local Jupyter Book build runner.

This script mirrors the GitHub Actions workflow in .github/workflows/deploy.yml:

1) Convert Jupytext-authored .py files into .ipynb notebooks
2) Start a Jupyter server (so `jupyter book build --execute` can connect)
3) Build the book to HTML
4) Copy any standalone *.html exports (e.g. PyVista) into `_build/html/pyvista/`

Usage examples:

  python tools/local_book_build.py
  python tools/local_book_build.py --port 8889 --token my-token
  python tools/local_book_build.py --no-execute

Notes:
- Requires: jupytext, jupyter-server, and jupyter-book (MyST) available on PATH.
- By default, converts all *.py under the repo root (like CI). Use --py-glob to narrow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from urllib.parse import urlsplit
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN = "my-jupyter-token"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888

# PyVista (via trame-vtk) exports HTML that references a VTK.js viewer bundle at
# absolute paths like `/vtk-js/assets/app.<hash>.js`. Those files won't exist on
# a typical local Jupyter Book site, causing the iframe to render blank.
#
# We post-process the copied HTML exports to load the viewer assets from the
# official VTK.js GitHub Pages site, which serves ES modules with permissive CORS.
DEFAULT_VTKJS_BASE_URL = "https://kitware.github.io/vtk-js"


JUPYTEXT_EXTRA_KEYS = (
    # Keys that trigger MyST warnings when present in notebook metadata.
    "notebook_metadata_filter",
    "cell_metadata_filter",
    "custom_cell_magics",
)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _fetch_url_text(url: str, *, timeout_s: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "local_book_build.py"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _discover_vtkjs_app_bundle(*, vtkjs_base_url: str) -> str:
    """Return full URL to the current vtk-js app bundle, e.g. .../assets/app.X.js."""

    index_url = vtkjs_base_url.rstrip("/") + "/"
    html = _fetch_url_text(index_url)

    origin = f"{urlsplit(index_url).scheme}://{urlsplit(index_url).netloc}"

    # Typical: <script type="module" src="/vtk-js/assets/app.CMgV8hKz.js"></script>
    m = re.search(
        r'<script\s+type="module"\s+src="(?P<src>/vtk-js/assets/app\.[^"]+\.js)"', html
    )
    if not m:
        raise RuntimeError(f"Could not find vtk-js app bundle in {index_url}")

    src = m.group("src")
    if src.startswith("/"):
        return origin + src
    return src


def _patch_trame_vtk_export_html(
    *,
    html_path: Path,
    vtkjs_base_url: str,
    vtkjs_app_bundle_url: str,
) -> bool:
    """Patch a trame-vtk static export so its viewer assets resolve."""

    try:
        text = html_path.read_text(encoding="utf-8")
    except Exception:
        return False

    # Only touch files that look like trame-vtk vtksz2html output.
    if "OfflineLocalView.load" not in text and "/vtk-js/assets/app." not in text:
        return False

    patched = text

    # 1) Point /vtk-js/... assets at the public vtk-js site.
    # Use a conservative rewrite on href/src attributes.
    patched = re.sub(
        r'(?P<attr>\b(?:href|src)=")/vtk-js/',
        rf"\g<attr>{vtkjs_base_url.rstrip('/')}/",
        patched,
    )
    patched = re.sub(
        r"(?P<attr>\b(?:href|src)=')/vtk-js/",
        rf"\g<attr>{vtkjs_base_url.rstrip('/')}/",
        patched,
    )

    # 2) Update stale hashed app bundle name to the current one.
    vtkjs_base_prefix = re.escape(vtkjs_base_url.rstrip("/"))
    patched = re.sub(
        rf"{vtkjs_base_prefix}/assets/app\.[^\"']+\.js",
        vtkjs_app_bundle_url,
        patched,
    )

    if patched == text:
        return False

    try:
        html_path.write_text(patched, encoding="utf-8")
        return True
    except Exception:
        return False


def _iter_files_by_globs(root: Path, globs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in globs:
        out.extend(sorted(root.glob(pattern)))
    # De-duplicate while keeping order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _strip_jupytext_extra_keys(ipynb_file: Path) -> bool:
    """Remove unsupported keys from metadata.jupytext to silence MyST warnings."""

    try:
        data = json.loads(ipynb_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    md = data.get("metadata")
    if not isinstance(md, dict):
        return False

    jmd = md.get("jupytext")
    if not isinstance(jmd, dict):
        return False

    changed = False
    for key in JUPYTEXT_EXTRA_KEYS:
        if key in jmd:
            jmd.pop(key, None)
            changed = True

    if changed:
        # Keep a stable, readable formatting.
        ipynb_file.write_text(
            json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return changed


def _is_tcp_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_for_server(host: str, port: int, timeout_s: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if _is_tcp_open(host, port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for Jupyter server at http://{host}:{port}/")


def convert_py_to_ipynb(*, root: Path, py_globs: list[str]) -> None:
    py_files = _iter_files_by_globs(root, py_globs)
    if not py_files:
        print("No .py files matched; skipping conversion.")
        return

    # CI uses: jupytext --from py --to notebook <file>
    # We use --update to preserve cell outputs/ids where present.
    for py_file in py_files:
        # Skip obvious virtualenv / build / git metadata.
        parts = set(py_file.parts)
        if {
            ".git",
            "_build",
            "site_exports",
            ".venv",
            "venv",
            "__pycache__",
            "tools",
        } & parts:
            continue
        if not py_file.is_file():
            continue
        _run(
            ["jupytext", "--from", "py", "--to", "ipynb", "--update", str(py_file)],
            cwd=root,
        )

        ipynb_file = py_file.with_suffix(".ipynb")
        if ipynb_file.exists():
            _strip_jupytext_extra_keys(ipynb_file)


def _find_free_port(host: str, starting_port: int, max_tries: int = 50) -> int:
    port = starting_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"Could not find a free port starting at {starting_port}")


def start_jupyter_server(
    *,
    root: Path,
    host: str,
    port: int,
    token: str,
    env: dict[str, str] | None = None,
) -> tuple[int, subprocess.Popen[str]]:
    cmd = [
        "jupyter",
        "server",
        f"--IdentityProvider.token={token}",
        f"--ServerApp.port={port}",
        "--ServerApp.port_retries=0",
        "--no-browser",
        f"--ServerApp.ip={host}",
        "--allow-root",
    ]
    print(f"\n$ {' '.join(cmd)} (background)", flush=True)
    # Inherit stdio so users can see logs; most environments run fine like this.
    proc = subprocess.Popen(cmd, cwd=str(root), text=True, env=env)
    return port, proc


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    try:
        proc.terminate()
        proc.wait(timeout=10)
        return
    except Exception:
        pass

    try:
        proc.kill()
    except Exception:
        pass


def build_book(*, root: Path, host: str, port: int, token: str, execute: bool) -> None:
    env = os.environ.copy()
    # MyST/jupyter-book starts a local web server and then fetches pages.
    # On some systems, `localhost` resolves to IPv6 (::1) first, which can
    # cause fetch failures if the server only listens on IPv4.
    env["HOST"] = host
    # Ensure node prefers IPv4 when resolving `localhost`.
    env.setdefault("NODE_OPTIONS", "--dns-result-order=ipv4first")
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    env.setdefault("PYVISTA_JUPYTER_BACKEND", "html")
    env.setdefault("JUPYTER_EXTENSION_ENABLED", "true")
    env["JUPYTER_BASE_URL"] = f"http://{host}:{port}/"
    env["JUPYTER_TOKEN"] = token

    cmd = ["jupyter", "book", "build", "--html", "--ci", "--keep-host"]
    if execute:
        cmd.append("--execute")

    _run(cmd, cwd=root, env=env)


def clear_myst_execute_cache(*, root: Path) -> None:
    """Remove the MyST execution cache.

    MyST can reuse cached notebook outputs even when `--execute` is set.
    That skips code execution and therefore skips side-effects like
    PyVista's `export_html(...)` writing files to disk.
    """

    # MyST v1 stores executed notebook artifacts under `_build/execute/`.
    # Some older setups may also use `_build/cache/execute/`.
    cache_dirs = [
        root / "_build" / "execute",
        root / "_build" / "cache" / "execute",
    ]
    cleared_any = False
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared MyST execute cache: {cache_dir}")
            cleared_any = True
    if not cleared_any:
        print("No MyST execute cache directories found to clear.")


def copy_pyvista_html_exports(*, root: Path) -> None:
    build_html = root / "_build" / "html"
    dest = build_html / "pyvista"
    dest.mkdir(parents=True, exist_ok=True)

    excluded_dirnames = {
        "_build",
        ".git",
        "site_exports",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
    }

    copied = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in excluded_dirnames]

        for name in filenames:
            if not name.endswith(".html"):
                continue
            html = Path(dirpath) / name
            target = dest / name
            try:
                shutil.copy2(html, target)
                copied += 1
            except Exception:
                # Keep going like CI ("|| true")
                pass

    print(f"Copied {copied} HTML export(s) into {dest}")


def patch_copied_pyvista_exports(*, root: Path, vtkjs_base_url: str) -> None:
    """Fix copied PyVista exports so their vtk-js assets load locally."""

    dest = root / "_build" / "html" / "pyvista"
    if not dest.exists():
        return

    try:
        vtkjs_app_bundle_url = _discover_vtkjs_app_bundle(vtkjs_base_url=vtkjs_base_url)
    except Exception as e:
        print(
            f"WARNING: Could not discover vtk-js app bundle ({e}). PyVista HTML exports may render blank."
        )
        return

    patched_count = 0
    for html in sorted(dest.glob("*.html")):
        if _patch_trame_vtk_export_html(
            html_path=html,
            vtkjs_base_url=vtkjs_base_url,
            vtkjs_app_bundle_url=vtkjs_app_bundle_url,
        ):
            patched_count += 1

    if patched_count:
        print(
            f"Patched {patched_count} PyVista HTML export(s) to load vtk-js assets from {vtkjs_base_url}"
        )


def serve_built_site(*, root: Path, host: str, port: int) -> None:
    build_html = root / "_build" / "html"
    if not build_html.exists():
        raise FileNotFoundError(f"Build output not found: {build_html}")

    port = _find_free_port(host, port)

    url = f"http://{host}:{port}/"
    print(f"\nServing built site from {build_html}")
    print(f"Open: {url}")
    print("Press Ctrl-C to stop.\n")
    subprocess.run(
        ["python", "-m", "http.server", str(port), "--bind", host],
        cwd=str(build_html),
        check=True,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: repo root)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument(
        "--py-glob",
        action="append",
        default=["jb2_playground/**/*.py"],
        help="Glob (relative to --root) of .py files to convert; can be provided multiple times",
    )
    parser.add_argument(
        "--no-execute", action="store_true", help="Build without executing notebooks"
    )
    parser.add_argument(
        "--server-timeout", type=float, default=30.0, help="Seconds to wait for server"
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep MyST execution cache (by default we clear it to ensure side-effects like PyVista export_html run)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="After building, serve the static site from _build/html using python -m http.server",
    )
    parser.add_argument(
        "--serve-port", type=int, default=8000, help="Port for --serve (default: 8000)"
    )
    parser.add_argument(
        "--vtkjs-base-url",
        default=DEFAULT_VTKJS_BASE_URL,
        help="Base URL used to load vtk-js assets for PyVista HTML exports",
    )
    parser.add_argument(
        "--no-patch-pyvista-exports",
        action="store_true",
        help="Do not rewrite copied PyVista HTML exports to load vtk-js assets",
    )

    args = parser.parse_args(argv)
    root: Path = args.root.resolve()

    env = os.environ.copy()
    env["HOST"] = args.host
    env.setdefault("NODE_OPTIONS", "--dns-result-order=ipv4first")
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    env.setdefault("PYVISTA_JUPYTER_BACKEND", "html")
    env.setdefault("JUPYTER_EXTENSION_ENABLED", "true")

    server_proc: subprocess.Popen[str] | None = None
    try:
        convert_py_to_ipynb(root=root, py_globs=args.py_glob)

        if not args.no_execute and not args.keep_cache:
            clear_myst_execute_cache(root=root)

        port = _find_free_port(args.host, args.port)
        port, server_proc = start_jupyter_server(
            root=root, host=args.host, port=port, token=args.token, env=env
        )
        _wait_for_server(args.host, port, args.server_timeout)

        build_book(
            root=root,
            host=args.host,
            port=port,
            token=args.token,
            execute=not args.no_execute,
        )
        copy_pyvista_html_exports(root=root)
        if not args.no_patch_pyvista_exports:
            patch_copied_pyvista_exports(root=root, vtkjs_base_url=args.vtkjs_base_url)

        print("\nBuild finished.")
        print(
            "Note: the transient MyST build server port (e.g. :3004/:3005) stops when the build ends."
        )
        print(f"Static site output: {root / '_build' / 'html'}")
        if args.serve:
            serve_built_site(root=root, host=args.host, port=args.serve_port)

        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        if server_proc is not None:
            stop_process(server_proc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
