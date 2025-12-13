@echo off
echo Starting Full Face Detection System...
echo.
echo Launching Caretaker Dashboard...
start "Caretaker Dashboard" cmd /k "streamlit run caretaker_dashboard.py"
echo.
echo Launching Camera Node...
start "Camera Sensor" cmd /k "python camera_node.py"
echo.
echo System Started. Minimize this window but do not close it.
pause
