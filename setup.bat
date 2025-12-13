@echo off
echo Installing Dependencies...
py -m pip install -r requirements.txt
echo.
echo Creating Initial Dummy Model (for demo purposes)...
py create_dummy_model.py
echo.
echo ==============================================
echo Setup Complete!
echo You can now run "start_system.bat" to launch the project.
echo ==============================================
pause
