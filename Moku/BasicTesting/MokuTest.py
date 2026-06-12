"""MokuTest.py -- Moku environment and connectivity diagnostic.

Verifies everything needed to control a Moku from Python:
  1. Python interpreter and version
  2. moku Python package        (pip install moku)
  3. mokucli command-line tool  (used for firmware/data downloads)
  4. Moku instrument data files (firmware bitstreams)
  5. Network connection to the Moku device itself

Run with:  python MokuTest.py
"""

import os
import re
import socket
import subprocess
import sys

MOKU_IP = "192.168.73.1"  # Same address used by ReadMoku.py
MOKU_PORT = 80            # Moku REST API port
TIMEOUT_S = 3

MOKU_LINK = "https://apis.liquidinstruments.com/starting-python.html"
CLI_LINK = "https://liquidinstruments.com/software/utilities/"
TARGET_LINK = ("https://apis.liquidinstruments.com/cli/moku-cli.html"
               "#finding-the-target-path-python")


def extract_version(text):
    """Pull a dotted version number (e.g. 3.2.1) out of command output."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    match = re.search(r"\d+(\.\d+)+", text)
    return match.group(0) if match else "unknown"


def import_moku():
    """Import the moku package, returning (module, error_message)."""
    try:
        import moku
        return moku, None
    except ImportError as e:
        if "pkg_resources" in str(e):
            return None, ("pkg_resources missing -- run `pip install "
                          "setuptools`")
        return None, ("moku package not installed -- run `pip install moku` "
                      "(%s)" % MOKU_LINK)


def check_python():
    return True, "%s (%s)" % (extract_version(sys.version), sys.executable)


def check_moku_package():
    moku, err = import_moku()
    if moku is None:
        return False, err
    version = getattr(moku, "__version__", None)
    if version:
        return True, extract_version(version)
    # Older packages don't expose __version__; ask pip instead
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", "moku"], timeout=30)
        return True, extract_version(out)
    except Exception as e:
        return False, "installed, but version unknown (%s)" % e


def check_moku_cli():
    moku, err = import_moku()
    if moku is None:
        return False, err
    cli_path = getattr(moku, "MOKU_CLI_PATH", "mokucli")
    try:
        out = subprocess.check_output([cli_path, "--version"], timeout=30)
        return True, "%s (%s)" % (extract_version(out), cli_path)
    except Exception:
        return False, "mokucli not found -- install it from %s" % CLI_LINK


def check_moku_data():
    moku, err = import_moku()
    if moku is None:
        return False, err
    data_path = getattr(moku, "MOKU_DATA_PATH", None)
    if data_path is None:
        return False, "moku package does not define MOKU_DATA_PATH"
    if not os.path.isdir(data_path):
        return False, ("data folder %s doesn't exist -- run `mokucli "
                       "download <FW_VER> --target=%s` (%s)"
                       % (data_path, data_path, TARGET_LINK))
    files = os.listdir(data_path)
    if not files:
        return False, ("no data files in %s -- run `mokucli download "
                       "<FW_VER> --target=%s` (%s)"
                       % (data_path, data_path, TARGET_LINK))
    return True, "%d file(s) in %s" % (len(files), data_path)


def check_device():
    try:
        with socket.create_connection((MOKU_IP, MOKU_PORT), TIMEOUT_S):
            return True, "Moku reachable at %s" % MOKU_IP
    except OSError as e:
        return False, ("no response from %s (%s) -- check power, USB/network "
                       "cable, and IP address" % (MOKU_IP, e))


def main():
    checks = [
        ("Python",            check_python),
        ("moku package",      check_moku_package),
        ("mokucli",           check_moku_cli),
        ("instrument data",   check_moku_data),
        ("device connection", check_device),
    ]

    print("Moku environment check")
    print("-" * 70)
    all_passed = True
    for name, check in checks:
        passed, detail = check()
        all_passed &= passed
        print("[%s] %-17s %s" % ("PASS" if passed else "FAIL", name, detail))
    print("-" * 70)
    print("RESULT: %s" % ("all checks passed"
                          if all_passed else "one or more checks FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
