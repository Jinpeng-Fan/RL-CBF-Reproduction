@echo off
setlocal
cd /d "%~dp0"
title RL-CBF - pendulum
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_demo.ps1" -Mode history -Task pendulum %*
if errorlevel 1 pause
