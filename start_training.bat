@echo off
setlocal
cd /d "%~dp0"
title RL-CBF - Live Training
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch_demo.ps1" -Mode train %*
if errorlevel 1 pause
