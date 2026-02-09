"""Scan all project .py files and extract third-party imports."""

import os
import re

STDLIB = set(
    """
__future__ abc argparse array ast asyncio atexit base64 binascii bisect builtins
bz2 calendar cgi cmath cmd code codecs collections colorsys compileall concurrent
configparser contextlib contextvars copy copyreg cProfile crypt csv ctypes curses
dataclasses datetime dbm decimal difflib dis distutils doctest email encodings enum
errno faulthandler filecmp fileinput fnmatch fractions ftplib functools gc getopt
getpass gettext glob gzip hashlib heapq hmac html http idlelib imaplib imghdr
importlib inspect io ipaddress itertools json keyword linecache locale logging lzma
math mimetypes mmap multiprocessing netrc numbers operator optparse os pathlib pdb
pickle pipes pkgutil platform plistlib pprint profile pstats queue quopri random re
readline reprlib resource runpy sched secrets select selectors shelve shlex shutil
signal site smtplib socket socketserver sqlite3 ssl stat statistics string struct
subprocess sys sysconfig tarfile tempfile textwrap threading time timeit tkinter
token tokenize tomllib trace traceback tracemalloc types typing unicodedata unittest
urllib uuid venv warnings wave weakref webbrowser winreg winsound wsgiref xml xmlrpc
zipfile zipimport zlib _thread typing_extensions ntpath posixpath
""".split()
)

LOCAL = set(
    """
api ai ocr Biometric biometric ledger infrastructure core utils app blockchain
scripts models routers services dashboard
""".split()
)

# Only scan these project source directories
SOURCE_DIRS = [
    "api",
    "ai",
    "ocr",
    "Biometric",
    "infrastructure",
    "ledger",
    "core",
    "scripts",
    "utils",
    "blockchain",
]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
imports = set()

for src_dir in SOURCE_DIRS:
    full_dir = os.path.join(project_root, src_dir)
    if not os.path.isdir(full_dir):
        continue
    for root, dirs, files in os.walk(full_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(root, f)
            try:
                with open(fp, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        m = re.match(r"^from\s+(\w+)", line)
                        if m:
                            imports.add(m.group(1))
                        m = re.match(r"^import\s+(\w+)", line)
                        if m:
                            imports.add(m.group(1))
            except Exception:
                pass

# Also scan root-level .py files
for f in os.listdir(project_root):
    if f.endswith(".py"):
        fp = os.path.join(project_root, f)
        try:
            with open(fp, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    m = re.match(r"^from\s+(\w+)", line)
                    if m:
                        imports.add(m.group(1))
                    m = re.match(r"^import\s+(\w+)", line)
                    if m:
                        imports.add(m.group(1))
        except Exception:
            pass

third_party = sorted(imports - STDLIB - LOCAL - {""})
for p in third_party:
    print(p)
