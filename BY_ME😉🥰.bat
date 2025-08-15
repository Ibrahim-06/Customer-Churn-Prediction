@echo off
title By: Ibrahim Mohamed
mode con cols=40 lines=10

setlocal enabledelayedexpansion
set colors=0A 0B 0C 0D 0E 0F 1A 1B 1C 1D 1E 1F 2A 2B 2C 2D 2E 2F

:loop
for %%c in (%colors%) do (
    color %%c
    cls
    echo.
    echo         =====================
    echo         Project By: Ibrahim Mohamed
    echo         =====================
    timeout /t 0.2 >nul
)
goto loop
