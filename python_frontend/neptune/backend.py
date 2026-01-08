import os
import hashlib
import ctypes
import subprocess
import tempfile
import atexit
import shutil
from pathlib import Path
import time

class AOTCompiler:
    def __init__(self):
        # 1. 确定缓存目录
        # 优先用环境变量，否则用 ~/.neptune/cache
        env_cache = os.environ.get("NEPTUNE_CACHE_DIR")
        if env_cache:
            self.cache_dir = Path(env_cache)
        else:
            self.cache_dir = Path.home() / ".neptune" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 尝试清理旧缓存 (可选，比如清理超过 7 天的)
        self._cleanup_old_cache()

    def compile_and_load(self, compiler_instance):
        # 1. 获取 IR 字符串用于计算 Hash
        # 注意：这里假设 compiler_instance 还没被破坏性降级
        # 如果 compile_to_object_file 会修改 module，你需要先 dump
        ir_str = compiler_instance.dump()
        module_hash = hashlib.sha256(ir_str.encode("utf-8")).hexdigest()[:16]
        
        # 2. 构造文件名
        base_name = self.cache_dir / f"neptune_kernel_{module_hash}"
        so_file = base_name.with_suffix(".so")
        obj_file = base_name.with_suffix(".o")

        # 3. 检查缓存命中
        if so_file.exists():
            # TODO: 可以在这里加个日志 "Cache hit!"
            try:
                return ctypes.CDLL(str(so_file))
            except OSError:
                # 如果加载失败（比如文件损坏），则重新编译
                pass

        # 4. 编译 (.o)
        # 调用 C++ 接口生成对象文件
        # 注意：这个接口需要接受 str 路径
        compiler_instance.compile_to_object_file(str(obj_file))

        # 5. 链接 (.so)
        # 这里需要获取 runtime 库的路径
        # 假设 runtime 就在 nptdsl/backend 目录下
        runtime_lib_dir = Path(__file__).parent / "backend"
        runtime_lib_name = "neptune_runtime" # libneptune_runtime.so
        
        # 链接命令
        cmd = [
            "clang++", 
            "-shared", 
            "-fPIC",
            "-o", str(so_file), 
            str(obj_file),
            f"-L{runtime_lib_dir}", 
            f"-l{runtime_lib_name}",
            # 如果 runtime 已经链接了 petsc，这里可能不需要显式链接 petsc，
            # 但为了保险通常还是加上，或者依靠 rpath
            "-Wl,-rpath," + str(runtime_lib_dir) 
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)

        # 6. 加载
        return ctypes.CDLL(str(so_file))

    def _cleanup_old_cache(self):
        # 简单清理策略：如果文件超过 7 天没用，删掉
        # 这里为了性能，可以用 try-except 包裹，避免阻塞主线程
        try:
            now = time.time()
            cutoff = 7 * 24 * 3600
            for p in self.cache_dir.glob("neptune_kernel_*"):
                if now - p.stat().st_atime > cutoff:
                    p.unlink()
        except Exception:
            pass

# 全局单例
_compiler = AOTCompiler()

def jit_compile(compiler_instance):
    return _compiler.compile_and_load(compiler_instance)
