@echo off
rem build.bat — build ARIA from src/ with MSVC 2022 BuildTools.
rem Usage: build.bat [output-name]   (default: aria_next.exe)
rem Matches the session-verified flags: /utf-8 /Zi /O2 /EHsc /std:c++17.
setlocal
set OUT=%~1
if "%OUT%"=="" set OUT=aria_next.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /utf-8 /Zi /O2 /EHsc /std:c++17 src\main.cpp src\evolution.cpp src\graph.cpp src\node.cpp src\logger.cpp src\serialize.cpp src\subgraph_library.cpp /Fe:%OUT% /Fd:%OUT:.exe=.pdb%
endlocal
