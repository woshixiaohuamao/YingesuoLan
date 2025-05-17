from typing import Optional
from fastapi import FastAPI
import uvicorn
#创建FastAPI实例
app = FastAPI()

#创建访问路径(将 HTTP GET 请求映射到 read_root() 函数)
@app.get("/")
def read_root():#定义根目录方法
    return {"message": "Hello World"}#返回响应信息

#定义方法，处理请求
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

#运行
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)