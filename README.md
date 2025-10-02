# track1_nottingDuck

## Team members

We are a team called *NottingDuck* with following three members:

- [Shuli Wang](https://github.com/NingShuZhu)
- [Yujie Yao](https://github.com/JackyYao1021)
- [Ningbo Wei](https://github.com/NingboWEI)

## Project description

### 📌 Project Overview

**Samantha** is an intelligent terminal assistant that bridges the gap between human intent and system commands in the **openEuler OS** (a Linux distribution developed by Huawei, based on the Linux kernel). 

Instead of memorizing complex CLI syntax, users can express their goals in natural language, and Samantha interprets, executes, and confirms actions automatically. Our goal is to make terminal interaction more intuitive, efficient, and human-centric.

### 2. 🏗️ System Architecture

In our design, Samantha follows a multi-agent, loop-until-success workflow. A user speaks in natural language; the system clarifies (if needed), plans the task, generates shell commands, asks for confirmation, executes, and—on failure—self-corrects and retries.



## Step-by-step running instructions



运行指引, 先进入容器，然后创建虚拟环境
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

启动容器 `docker start oe`
进入容器 `docker exec -it oe bash`
离开容器 `exit`
关闭容器 `docker stop oe`
激活虚拟环境 `source venv/bin/activate`
退出虚拟环境 `deactivate`

可运行代码都在work文件夹里面，能直接在docker中运行调用
