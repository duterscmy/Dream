#!/bin/bash
# download.sh - 使用scp从服务器下载文件到本地logs文件夹

# 检查参数是否提供
if [ -z "$1" ]; then
    echo "错误：请提供服务器上的文件路径"
    echo "用法: ./download.sh <服务器文件路径>"
    echo "示例: ./download.sh /home/mc03002/experiment/log.txt"
    exit 1
fi

# 服务器配置
SERVER_USER="mc03002"
SERVER_HOST="aisurrey-submit01.surrey.ac.uk"

# 服务器文件路径
SERVER_PATH="$1"

# 本地logs目录
LOCAL_DIR="./logs"

# 创建logs目录（如果不存在）
mkdir -p "$LOCAL_DIR"

echo "正在从服务器下载文件..."
echo "  源路径: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
echo "  目标目录: ${LOCAL_DIR}/"

# 执行scp下载
scp "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}" "$LOCAL_DIR/"

# 检查scp是否成功
if [ $? -eq 0 ]; then
    echo "✓ 下载成功！文件已保存到: ${LOCAL_DIR}/$(basename "$SERVER_PATH")"
else
    echo "✗ 下载失败，请检查："
    echo "  1. 服务器路径是否正确"
    echo "  2. 网络连接是否正常"
    echo "  3. 是否有权限访问该文件"
    exit 1
fi