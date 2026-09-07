@echo off
rem build.bat — build ARIA from src/ with MSVC 2022 BuildTools.
rem Usage: build.bat [output-name]   (default: aria_next.exe)
rem Intermediates (.obj/.pdb/.ilk) land in build/ (gitignored); the exe
rem lands at the repo root. Flags: BUILD_REPRO.md documents the pin.
setlocal
set OUT=%~1
if "%OUT%"=="" set OUT=aria_next.exe
where cl >nul 2>&1
if errorlevel 1 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found — install VS 2022 BuildTools (C++ workload^)
    exit /b 1
)
if not exist build mkdir build
cd build
cl /nologo /utf-8 /Zi /O2 /EHsc /std:c++17 /Fo..\build\ /Fd..\build\ ^
   ..\src\main.cpp ..\src\evolution.cpp ..\src\graph.cpp ..\src\node.cpp ^
   ..\src\logger.cpp ..\src\serialize.cpp ..\src\subgraph_library.cpp ^
   /Fe:"..\%OUT%" /Fd:"..\build\vc.pdb"
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD OK: %OUT%
endlocal
