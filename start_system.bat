@echo off
echo Starting Full Face Detection System...
echo.
echo Launching Caretaker Dashboard...
start "Caretaker Dashboard" cmd /k "py -m streamlit run caretaker_dashboard.py"
echo.
echo Launching Camera Node...
start "Camera Sensor" cmd /k "py camera_node.py"
echo.
echo System Started. Minimize this window but do not close it.
pause
