#!/usr/bin/env bash
set -euo pipefail

# build.sh -- 将所有构建集中到 repo_root/build 下（支持 LLVM21.x、ccache、lld 检测）
# Usage:
#   ./scripts/build.sh              # 正常构建
#   ./scripts/build.sh --clean      # 清理 build_root 下的所有构建产物
#
# Env overrides:
#   BUILD_ROOT            - 默认: <repo_root>/build
#   NINJA_JOBS            - 并行构建数，默认: cpu cores
#   CCACHE_DIR            - ccache 路径，默认: ~/.ccache_mymlir
#   CCACHE_MAXSIZE        - ccache 大小，默认: 40G
#   LLVM_TAG              - 要 clone 的 llvm tag，默认: llvmorg-21.0.0
#   LLVM_ENABLE_PROJECTS  - 要构建的 llvm projects，默认: mlir
#   LLVM_TARGETS_TO_BUILD - targets，默认: Host

# ---------------- defaults ----------------
: "${NINJA_JOBS:=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
: "${CCACHE_DIR:=${HOME}/.ccache_mymlir}"
: "${CCACHE_MAXSIZE:=40G}"
: "${LLVM_TAG:=llvmorg-21.0.0}"
: "${LLVM_ENABLE_PROJECTS:=mlir}"
: "${LLVM_TARGETS_TO_BUILD:=host}"
: "${BUILD_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build}"

# ---------------- parse args ----------------
CLEAN_ONLY=0
if [ "${1:-}" = "--clean" ] || [ "${1:-}" = "-c" ]; then
  CLEAN_ONLY=1
fi

# ---------------- paths ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
LLVM_PROJ_DIR="${THIRD_PARTY_DIR}/llvm-project"
LLVM_SRC_DIR="${LLVM_PROJ_DIR}/llvm"

# centralized build root
mkdir -p "${BUILD_ROOT}"
LLVM_BUILD_DIR="${BUILD_ROOT}/llvm-build"
LLVM_INSTALL_DIR="${BUILD_ROOT}/llvm-install"
TOP_BUILD_DIR="${BUILD_ROOT}/project-build"

echo "=== MyMLIRProject build helper ==="
echo "Repo root: ${REPO_ROOT}"
echo "Build root (all artifacts here): ${BUILD_ROOT}"
echo "LLVM tag: ${LLVM_TAG}"
echo "LLVM projects: ${LLVM_ENABLE_PROJECTS}"
echo "LLVM targets: ${LLVM_TARGETS_TO_BUILD}"
echo "Ninja jobs: ${NINJA_JOBS}"
echo "CCACHE_DIR: ${CCACHE_DIR} (max ${CCACHE_MAXSIZE})"
echo

# -- clean mode: remove the centralized build root
if [ "${CLEAN_ONLY}" -eq 1 ]; then
  echo "Cleaning build root: ${BUILD_ROOT}"
  rm -rf "${BUILD_ROOT}"
  echo "Done."
  exit 0
fi

# ---------------- prerequisites check ----------------
for cmd in git cmake ninja ccache; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}"
    echo "Please install it (e.g. apt/brew/pacman) and re-run."
    exit 1
  fi
done

# choose compilers (prefer clang)
CC_CANDIDATE="clang"
CXX_CANDIDATE="clang++"
if [ "$(uname -s)" = "Darwin" ]; then
  # prefer Homebrew llvm if present
  if [ -x "/opt/homebrew/opt/llvm/bin/clang" ]; then
    CC_CANDIDATE="/opt/homebrew/opt/llvm/bin/clang"
    CXX_CANDIDATE="/opt/homebrew/opt/llvm/bin/clang++"
  elif [ -x "/usr/local/opt/llvm/bin/clang" ]; then
    CC_CANDIDATE="/usr/local/opt/llvm/bin/clang"
    CXX_CANDIDATE="/usr/local/opt/llvm/bin/clang++"
  fi
fi
if ! command -v "${CC_CANDIDATE}" >/dev/null 2>&1; then
  if command -v gcc >/dev/null 2>&1; then
    CC_CANDIDATE="gcc"
    CXX_CANDIDATE="g++"
  else
    echo "ERROR: no suitable C/C++ compiler found (clang or gcc)."
    exit 1
  fi
fi
echo "Using C compiler: ${CC_CANDIDATE}"
echo "Using C++ compiler: ${CXX_CANDIDATE}"
echo

# ---------------- ensure llvm-project exists ----------------
if [ ! -d "${LLVM_PROJ_DIR}" ]; then
  echo "llvm-project not found under ${LLVM_PROJ_DIR}. Cloning ${LLVM_TAG} (shallow)..."
  mkdir -p "${THIRD_PARTY_DIR}"
  git -C "${THIRD_PARTY_DIR}" clone --depth 1 --branch "${LLVM_TAG}" https://github.com/llvm/llvm-project.git llvm-project
fi
if [ ! -d "${LLVM_SRC_DIR}" ]; then
  echo "ERROR: llvm source not found at ${LLVM_SRC_DIR}"
  exit 1
fi

# ---------------- ccache setup ----------------
mkdir -p "${CCACHE_DIR}"
export CCACHE_DIR="${CCACHE_DIR}"
ccache --max-size="${CCACHE_MAXSIZE}" >/dev/null 2>&1 || true
echo "ccache stats (before):"
ccache -s || true
echo

# ---------------- lld detection ----------------
ENABLE_LLD=OFF
if command -v ld.lld >/dev/null 2>&1 || command -v lld >/dev/null 2>&1; then
  ENABLE_LLD=ON
  echo "lld detected -> will enable LLVM_ENABLE_LLD=ON"
else
  ENABLE_LLD=OFF
  echo "lld NOT detected -> LLVM_ENABLE_LLD will be OFF (avoids -fuse-ld=lld test failure)"
  echo "Install lld to enable it (Debian/Ubuntu: sudo apt install lld; macOS: brew install llvm)."
fi
echo

# ---------------- 1) Configure & build LLVM (out-of-source) ----------------
mkdir -p "${LLVM_BUILD_DIR}"
echo "Configuring LLVM build (source: ${LLVM_SRC_DIR}, build: ${LLVM_BUILD_DIR})..."
cmake -G Ninja \
  -S "${LLVM_SRC_DIR}" \
  -B "${LLVM_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}" \
  -DLLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS}" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD}" \
  -DLLVM_CCACHE_BUILD=true \
  -DLLVM_ENABLE_LLD=${ENABLE_LLD} \
  -DLLVM_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER="${CC_CANDIDATE}" \
  -DCMAKE_CXX_COMPILER="${CXX_CANDIDATE}"

echo
echo "Building + installing LLVM (this may take a while)..."
cmake --build "${LLVM_BUILD_DIR}" --target install -- -j"${NINJA_JOBS}"
echo "LLVM installed to: ${LLVM_INSTALL_DIR}"
echo

# ---------------- 2) Configure & build top-level project (use installed LLVM) ----------------
mkdir -p "${TOP_BUILD_DIR}"
echo "Configuring top-level project (build dir: ${TOP_BUILD_DIR})..."
cmake -G Ninja \
  -S "${REPO_ROOT}" \
  -B "${TOP_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="${LLVM_INSTALL_DIR}/lib/cmake/llvm" \
  -DMLIR_DIR="${LLVM_INSTALL_DIR}/lib/cmake/mlir" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

echo
echo "Building top-level project..."
cmake --build "${TOP_BUILD_DIR}" -- -j"${NINJA_JOBS}"

# ---------------- 3) run tests if present ----------------
if [ -d "${REPO_ROOT}/test" ]; then
  echo
  echo "Running tests with ctest..."
  ctest --test-dir "${TOP_BUILD_DIR}" --output-on-failure -j"${NINJA_JOBS}" || {
    echo "Some tests failed (see above)."
  }
fi

echo
echo "Build finished. ccache stats (after):"
ccache -s || true
echo "All artifacts are under: ${BUILD_ROOT}"
