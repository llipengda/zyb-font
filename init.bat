@echo off
chcp 65001 > nul
setlocal

set PYTHON=python

rem 获取 Python 版本信息
for /f "tokens=2 delims= " %%a in ('%PYTHON% -V 2^>^&1') do set python_version=%%a

rem 检测 Python 版本是否在指定范围内
if "%python_version%" geq "3.11.3" if "%python_version%" lss "3.12" (
    goto :next
) else (
    goto :end
)

:end
echo Python 版本不满足 ^>=^ 3.11.3 且 ^<^ 3.12
echo 当前 Python 版本为 %python_version%
echo 考虑修改 init.bat 中的 PYTHON 变量
exit /b 1

:next
echo 当前 Python 版本为 %python_version%
echo.
echo 创建虚拟环境
if exist venv (
    echo SKIP
) else (
    %PYTHON% -m venv venv
    if %errorlevel% neq 0 (
        echo 错误：创建虚拟环境失败
        exit /b 1
    )
    echo OK
)
echo.
echo 激活虚拟环境
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo 错误：激活虚拟环境失败
    exit /b 1
)
echo OK
echo.
echo 升级 pip
%PYTHON% -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo 错误：升级 pip 失败
    exit /b 1
)
echo OK
echo.
echo 安装依赖
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误：安装依赖失败
    exit /b 1
)
echo OK

endlocal
