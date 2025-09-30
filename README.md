# track1_nottingDuck

运行指引, 先进入容器，然后创建虚拟环境
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

进入容器 `docker exec -it oe bash`
离开容器 `exit`
激活虚拟环境 `source venv/bin/activate`
退出虚拟环境 `deactivate`

可运行代码都在work文件夹里面，能直接在docker中运行调用
