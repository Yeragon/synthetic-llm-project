FROM python:3.10-slim

# 设置容器中的工作目录
WORKDIR /app

# 拷贝并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 默认执行指令（可 later 替换成 jupyter notebook）
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


