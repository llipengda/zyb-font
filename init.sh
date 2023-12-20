#!/bin/bash

PYTHON="python3"

# 获取 Python 版本信息
python_version=$("$PYTHON" -V 2>&1 | awk '{print $2}')

next() {
    echo "当前 Python 版本为 $python_version"
    echo ""
    echo "创建虚拟环境"
    if [[ -d "venv" ]]; then
        echo "SKIP"
    else
        if ! "$PYTHON" -m venv venv; then
            echo "错误：创建虚拟环境失败"
            exit 1
        fi
        echo "OK"
    fi
    echo ""
    echo "激活虚拟环境"
    # shellcheck disable=SC1091
    if ! source venv/bin/activate; then
        echo "错误：激活虚拟环境失败"
        exit 1
    fi
    echo "OK"
    echo ""
    echo "升级 pip"
    if ! pip install --upgrade pip; then
        echo "错误：升级 pip 失败"
        exit 1
    fi
    echo "OK"
    echo ""
    echo "安装依赖"
    if ! pip install -r requirements.txt; then
        echo "错误：安装依赖失败"
        exit 1
    fi
    echo "OK"
}

end() {
    echo "Python 版本不满足 >= 3.11.3 且 < 3.12"
    echo "当前 Python 版本为 $python_version"
    echo "考虑修改 init.sh 中的 PYTHON 变量"
    exit 1
}

# 检测 Python 版本是否在指定范围内
# shellcheck disable=SC2072
if [[ "$python_version" == "3.11."* && "$python_version" < "3.12" ]]; then
    next
else
    end
fi
