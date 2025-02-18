# UNet-App: 乳腺癌医学图像分割工具

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**UNet-App** 是一个基于深度学习的乳腺癌医学图像分割工具。该项目使用 UNet 模型对医学图像进行分割，并提供了一个用户友好的前端界面，方便用户上传图像并查看分割结果。

## 功能特点

- **图像分割**：支持上传医学图像并生成预测掩码和叠加图像。
- **黑白主题切换**：支持浅色模式和深色模式，提升用户体验。
- **响应式设计**：适配不同屏幕尺寸，确保在桌面端和移动端都能正常使用。
- **科研成果展示**：展示乳腺癌相关的研究成果，包括损失函数曲线、交并比 (IoU) 和示例图像。
- **前后端分离架构**：
  - 前端：基于 React 和 Chakra UI 构建。
  - 后端：基于 Flask 和 PyTorch 实现图像分割模型推理。

---

## 技术栈

### 前端
- **React**：用于构建用户界面。
- **Chakra UI**：提供现代化的 UI 组件和主题支持。
- **Axios**：处理 HTTP 请求。
- **React Router**：实现页面路由。

### 后端
- **Flask**：轻量级 Python Web 框架。
- **PyTorch**：深度学习框架，用于加载和运行 UNet 模型。
- **OpenCV**：处理图像数据。
- **Albumentations**：用于图像预处理和增强。

---

## 安装与运行

### 前置条件

1. **Python 环境**：
   - 安装 Python 3.8 或更高版本。
   - 安装依赖项：
     ```bash
     pip install -r requirements.txt
     ```

2. **Node.js 环境**：
   - 安装 Node.js 和 npm。
   - 安装前端依赖项：
     ```bash
     npm install
     ```

3. **模型文件**：
   - 将训练好的 UNet 模型文件（如 `model.pth`）放置在 `models/data_NestedUNet_woDS/` 目录下。
   - 确保配置文件（如 `config.yml`）也位于同一目录。

---

### 后端启动

1. 进入后端目录：
   ```bash
   cd unet-backend
   ```

2. 启动 Flask 应用：
   ```bash
   python main.py
   ```

3. 默认情况下，后端服务会运行在 `http://localhost:5000`。

---

### 前端启动

1. 进入前端目录：
   ```bash
   cd frontend
   ```

2. 启动开发服务器：
   ```bash
   npm i
   npx vite
   ```

3. 默认情况下，前端应用会运行在 `http://localhost:5173`。

---

## 使用说明

1. **访问首页**：
   - 打开浏览器并访问 `http://localhost:5173`。
   - 首页展示了乳腺癌相关的科研成果，包括损失函数曲线、交并比 (IoU) 和示例图像。

2. **图像分割**：
   - 导航到“图像分割”页面。
   - 点击或拖拽上传一张医学图像。
   - 系统会自动调用后端模型进行分割，并显示原图、预测掩码和叠加图像。

3. **主题切换**：
   - 在侧边栏中点击“切换主题”按钮，可以在浅色模式和深色模式之间切换。

---

## 目录结构

```
unet-app/
├── backend/               # 后端代码
│   ├── main.py             # Flask 主程序
│   ├── models/            # 存放模型文件和配置
│   └── requirements.txt   # Python 依赖项
├── frontend/              # 前端代码
│   ├── public/            # 静态资源
│   ├── src/               # React 源码
│   │   ├── components/    # 可复用组件
│   │   ├── pages/         # 页面组件
│   │   └── App.tsx        # 主入口文件
│   ├── package.json       # Node.js 依赖项
│   └── vite.config.ts     # Vite 配置文件
├── README.md              # 项目说明文档
└── LICENSE                # 开源许可证
```

---

## 贡献指南

欢迎为本项目贡献代码！请遵循以下步骤：

1. **Fork 仓库**：
   - 点击右上角的 "Fork" 按钮，将项目复制到你的 GitHub 账户。

2. **克隆仓库**：
   ```bash
   git clone https://github.com/lijin6/unet-app
   cd unet-app
   ```

3. **创建分支**：
   ```bash
   git checkout -b github.com/lijin6/unet-app
   ```

4. **提交更改**：
   - 完成修改后，提交代码：
     ```bash
     git add .
     git commit -m "Add your commit message"
     git push origin feature/your-feature-name
     ```

5. **创建 Pull Request**：
   - 在 GitHub 上打开你的分支，点击 "Compare & pull request" 按钮，描述你的更改。

---

## 许可证

本项目采用 [MIT License](LICENSE)，允许自由使用、修改和分发。

---

## 致谢

- **Chakra UI**：提供了现代化的 UI 组件库。
- **Flask**：简化了后端开发流程。
- **PyTorch**：强大的深度学习框架，支持 UNet 模型推理。

---

## 联系方式

如有任何问题或建议，请通过以下方式联系我：

- **GitHub Issues**：在项目仓库中提交问题。
- **邮箱**：[lijinliu.ac@gmail.com]

---
