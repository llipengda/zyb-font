$PYTHON = "python"

# 获取 Python 版本信息
$python_version = & $PYTHON -V 2>&1 | ForEach-Object { $_.Split(' ')[1] }

function Next {
    Write-Host "当前 Python 版本为 $python_version"
    Write-Host ""
    Write-Host "创建虚拟环境"
    if (Test-Path .\venv) {
        Write-Host "SKIP"
    } else {
        & $PYTHON -m venv venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "错误：创建虚拟环境失败"
            exit 1
        }
        Write-Host "OK"
    }
    Write-Host ""
    Write-Host "激活虚拟环境"
    . .\venv\Scripts\Activate
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误：激活虚拟环境失败"
        exit 1
    }
    Write-Host "OK"
    Write-Host ""
    Write-Host "升级 pip"
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误：升级 pip 失败"
        exit 1
    }
    Write-Host "OK"
    Write-Host ""
    Write-Host "安装依赖"
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误：安装依赖失败"
        exit 1
    }
    Write-Host "OK"
}

function End {
    Write-Host "Python 版本不满足 >= 3.11.3 且 < 3.12"
    Write-Host "当前 Python 版本为 $python_version"
    Write-Host "考虑修改 init.ps1 中的 PYTHON 变量"
    exit 1  
}

if ($python_version -ge "3.11.3" -and $python_version -lt "3.12") {
    Next
}
else {
    End
}
