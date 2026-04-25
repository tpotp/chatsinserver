#!/usr/bin/env python3
"""Launch the public Vercel site in host-simulation mode for the Chile swarm demo."""

from __future__ import annotations

import argparse
import datetime as dt
import random
import shutil
import string
import subprocess
import sys
import urllib.parse
import webbrowser
from pathlib import Path


PUBLIC_BASE_URL = "https://chatsinserver.vercel.app/index.html"
HOST_BASE_URL = "https://chatsinserver.vercel.app/host.html"
DEFAULT_SIMULATED_PHONES = 12
DEFAULT_BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
WEBGPU_FLAGS = [
    "--new-window",
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
    "--enable-features=Vulkan",
]


def build_room_name() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    return f"pudu-chile-{stamp}-{suffix}"


def build_urls(public_base_url: str, host_base_url: str, simulated_phones: int) -> tuple[str, str, str]:
    host_query = urllib.parse.urlencode({"simulate": simulated_phones})
    assist_query = urllib.parse.urlencode({"assistOnly": 1})
    return (
        f"{host_base_url}?{host_query}",
        public_base_url,
        f"{public_base_url}?{assist_query}",
    )


def find_browser(preferred: str | None) -> str | None:
    candidates = [preferred] if preferred else []
    candidates.extend(DEFAULT_BROWSER_CANDIDATES)
    for candidate in candidates:
      if candidate and Path(candidate).exists():
        return candidate
    return None


def copy_to_clipboard(value: str) -> bool:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return False
    try:
        subprocess.run(
            [powershell, "-NoProfile", "-Command", "Set-Clipboard", "-Value", value],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def open_host(browser_path: str | None, host_url: str) -> None:
    if browser_path:
        subprocess.Popen([browser_path, *WEBGPU_FLAGS, host_url])
        return
    webbrowser.open(host_url, new=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the public Pudu swarm host with Chilean phone simulation ready for a real mobile to join.",
    )
    parser.add_argument("--simulate", type=int, default=DEFAULT_SIMULATED_PHONES, help="Celulares simulados por sala (3-15).")
    parser.add_argument("--room", type=str, default="", help="Opcional. Ya no hace falta en el modo multi-sala actual; se mantiene solo por compatibilidad.")
    parser.add_argument("--browser", type=str, default="", help="Optional full path to Chrome or Edge.")
    parser.add_argument("--public-url", type=str, default=PUBLIC_BASE_URL, help="Public chat URL for users.")
    parser.add_argument("--host-url", type=str, default=HOST_BASE_URL, help="Host dashboard URL.")
    parser.add_argument("--no-open", action="store_true", help="Print the URLs without opening the host browser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulated_phones = min(15, max(3, int(args.simulate)))
    room = args.room.strip() or build_room_name()
    host_url, phone_url, assist_url = build_urls(args.public_url, args.host_url, simulated_phones)
    browser_path = find_browser(args.browser.strip() or None)
    copied = copy_to_clipboard(phone_url)

    print()
    print("PUDU COLMENA CHILE")
    print("==================")
    print(f"Sala               : automatica (Sala 1, Sala 2, ...)")
    print(f"Celulares simulados: {simulated_phones}")
    print(f"Host dashboard     : {host_url}")
    print(f"Celular            : {phone_url}")
    print(f"Cliente liviano    : {assist_url}")
    print(f"Clipboard          : {'URL del celular copiada' if copied else 'No se pudo copiar automaticamente'}")
    if browser_path:
        print(f"Browser            : {browser_path}")
    else:
        print("Browser            : fallback del sistema")
    print()
    print("Pasos de manana:")
    print("1. Deja abierto el host dashboard para que levante la colmena de la primera sala.")
    print("2. Abre la URL 'Celular' desde tu Redmi 10C en Entel 4G o 5G.")
    print("3. Invoca al pudu con @pudu o @pudubot seguido de tu mensaje.")
    print("4. Si pruebas desde otro navegador del PC, usa 'Cliente liviano' para enrutar sin shards locales.")
    print("5. Para probar salas llenas, usa entre --simulate 10 y --simulate 15.")
    print()

    if not args.no_open:
        open_host(browser_path, host_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
