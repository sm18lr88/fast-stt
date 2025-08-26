"""
Deprecated shim. Use the packaged entrypoint instead:
    python -m fast_stt.cli_download
or the console script:
    fast-stt-download-model
"""

from fast_stt.cli_download import main

if __name__ == "__main__":
    main()
