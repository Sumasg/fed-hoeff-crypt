@echo off
echo Running FedHoeffCrypt Experiments with Python 3.10...
echo.

if "%1"=="" (
    echo No mode specified, running quick test...
    py -3.10 -X utf8 run_experiments.py --mode quick
) else (
    py -3.10 -X utf8 run_experiments.py --mode %1
)

pause