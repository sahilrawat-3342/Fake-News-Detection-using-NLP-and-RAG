@echo off
cd /d "%~dp0"
python evaluate_performance.py --visualize --export-json performance_metrics.json --export-csv performance_metrics.csv
pause