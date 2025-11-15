# run_all.ps1 — sets up venv, installs deps, runs analysis, and opens presentation
python3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\hello.py
# try to open the generated pptx (Windows)
if (Test-Path presentation.pptx) { Start-Process presentation.pptx }
