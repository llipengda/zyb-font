# zyb-font -- 生成你的手写字体

## 前提

- `Python` `>= 3.11.4 && < 3.12`
- `CUDA` 12.1

## 部署

你可以使用脚本自动部署，也可以手动部署。

### 使用脚本

- Linux

```bash
./init.sh
```

- Windows(CMD)

```bash
.\init.bat
```

- Windows(PowerShell)

```bash
.\init.ps1
```

### 手动部署

创建 `Python` 虚拟环境

- Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

激活虚拟环境

- Linux

```bash
source venv/bin/activate
```

- Windows(PowerShell)

```bash
.\venv\Scripts\activate.ps1
```

- Windows(CMD)

```bash
.\venv\Scripts\activate.bat
```

安装依赖

```bash
pip install -r requirements.txt
```
