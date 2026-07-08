from setuptools import setup, find_packages

setup(
    name="disk_robot",
    version="0.1.0",
    packages=find_packages(),  # 自动识别disk_robot包
    python_requires=">=3.11",
    # 可选：自动把scripts下脚本注册成命令行工具
    entry_points={
        "console_scripts": [
            "mjx_train_walk = scripts.mjx_train_walk:main",
        ]
    },
)