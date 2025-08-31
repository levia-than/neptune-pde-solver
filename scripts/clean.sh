#!/bin/bash
set -e

# 1. 定义目录
PROJECT_ROOT=$(cd $(dirname $0)/..; pwd)
BUILD_DIR=${PROJECT_ROOT}/build
THIRD_PARTY_BUILD=${PROJECT_ROOT}/third_party/build
THIRD_PARTY_INSTALL=${PROJECT_ROOT}/third_party/install

# 2. 清理主项目构建目录
if [ -d "${BUILD_DIR}" ]; then
  echo "Cleaning main project build directory..."
  rm -rf ${BUILD_DIR}
fi

# 3. 清理third_party构建/安装目录
if [ -d "${THIRD_PARTY_BUILD}" ]; then
  echo "Cleaning third_party build directory..."
  rm -rf ${THIRD_PARTY_BUILD}
fi
if [ -d "${THIRD_PARTY_INSTALL}" ]; then
  echo "Cleaning third_party install directory..."
  rm -rf ${THIRD_PARTY_INSTALL}
fi

echo "Clean completed successfully!"
