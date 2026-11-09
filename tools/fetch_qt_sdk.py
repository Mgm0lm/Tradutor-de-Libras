"""Prepare the Qt Base SDK for the native Windows build inside the workspace."""
import hashlib
from pathlib import Path
import subprocess
import urllib.request

DOWNLOADS = Path(__file__).resolve().parent / "downloads"
VERSION = "6.11.1"
ARCHIVE = "6.11.1-0-202605090529qtbase-Windows-Windows_11_24H2-MSVC2022-Windows-Windows_11_24H2-X86_64.7z"
URL = "https://download.qt.io/online/qtsdkrepository/windows_x86/desktop/qt6_6111/qt6_6111_msvc2022_64/qt.qt6.6111.win64_msvc2022_64/" + ARCHIVE
SHA256 = "7f97edc3937fec7383eb865e010ed5128155bf9c80a563abca450860f3e9bef5"


def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    sdk = DOWNLOADS / f"qt-{VERSION}"
    archive = DOWNLOADS / ARCHIVE
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(f"Checksum inválido: {archive}")
    sdk.mkdir(exist_ok=True)
    if not (sdk / "lib/Qt6Widgets.lib").is_file():
        subprocess.run(["tar", "-xf", str(archive), "-C", str(sdk)], check=True)
    if not (sdk / "lib/Qt6Widgets.lib").is_file():
        raise SystemExit(f"Qt Widgets não encontrado em {sdk}")
    licenses = sdk / "licenses"
    licenses.mkdir(exist_ok=True)
    for name in ("LGPL-3.0-only.txt", "GPL-3.0-only.txt"):
        path = licenses / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/qt/qtbase/v{VERSION}/LICENSES/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
    print(sdk)


if __name__ == "__main__":
    main()
