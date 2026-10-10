"""Use direct HTTPS for one charged original-file materialization.

An explicit empty ProxyHandler disables the configured system/env proxy.
The frozen materializer retains its TLS validation, redirect allowlist,
complete-binary checks, deadline and independent reservation accounting.
"""
import urllib.request

from tools.diagnose_geobc_drive_materialization import main as diagnosed_main


def main():
    original = urllib.request.build_opener

    def direct_opener(*handlers):
        # The explicit handler suppresses urllib's automatic default proxy.
        # Remove caller ProxyHandlers too, so none can re-enable a proxy.
        retained = tuple(handler for handler in handlers
                         if not isinstance(handler, urllib.request.ProxyHandler)
                         and handler is not urllib.request.ProxyHandler)
        return original(urllib.request.ProxyHandler({}), *retained)

    urllib.request.build_opener = direct_opener
    try:
        return diagnosed_main()
    finally:
        urllib.request.build_opener = original


if __name__ == "__main__":
    raise SystemExit(main())
