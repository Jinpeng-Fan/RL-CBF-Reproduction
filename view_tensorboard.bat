@echo off
setlocal
cd /d "%~dp0"
title RL-CBF - Saved Experiments
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_demo.ps1" -Mode history %*
if errorlevel 1 pause
