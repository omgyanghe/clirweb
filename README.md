# ckirweb

中哈跨语言信息检索系统

## 项目目录结构

```
ckirweb/
├── .gitignore                    # Git忽略文件配置
├── LICENSE                       # 项目许可证
├── README.md                     # 项目说明文档
├── ckirweb.excalidraw           # 项目架构图
├── app/                          # 主应用程序目录
│   ├── __init__.py              # Python包初始化文件
│   ├── main.py                  # 应用程序入口文件
│   ├── schemas.py               # 数据模式定义
│   ├── utils.py                 # 工具函数
│   ├── api/                     # API接口模块
│   │   ├── __init__.py
│   │   ├── main.py              # API主文件
│   │   └── routes/              # API路由目录
│   ├── core/                    # 核心配置模块
│   │   ├── __init__.py
│   │   └── config.py            # 配置文件
│   ├── models/                  # 模型定义目录
│   │   ├── __init__.py
│   │   ├── cross_encoder.py     # 交叉编码器模型
│   │   └── sentence_encoder.py  # 句子编码器模型
│   └── services/                # 业务服务层
│       ├── __init__.py
│       ├── document_service.py  # 文档服务
│       ├── rerank_service.py    # 重排序服务
│       └── retrieval_service.py # 检索服务
├── static/                      # 静态资源目录
│   ├── index.html              # 主页面
│   ├── css/                    # 样式文件目录
│   │   └── style.css           # 主样式文件
│   └── js/                     # JavaScript文件目录
│       └── app.js              # 主应用脚本
├── test/                       # 测试文件目录
│   ├── faiss_test.py          # FAISS向量检索测试
│   └── flagembedding_test.py   # FlagEmbedding模型测试
└── logs/                       # 日志文件目录
```

### 主要模块说明

- **app/**: 核心应用程序代码
  - **api/**: RESTful API接口实现
  - **core/**: 核心配置和设置
  - **models/**: 机器学习模型定义
  - **services/**: 业务逻辑服务层
- **static/**: 前端静态资源文件
- **test/**: 单元测试和集成测试
- **logs/**: 应用程序运行日志