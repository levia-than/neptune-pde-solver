import functools
import hashlib
from .core import get_compiler
from .expr import Expr
from .backend import jit_compile
import ctypes

class JITClassWrapper:
    def __init__(self, cls, *args, **kwargs):
        self._cls = cls
        self._init_args = args
        self._init_kwargs = kwargs
        
        self._compiled_lib = None
        self._compiler = get_compiler()
        
        # 这一步非常关键：我们需要一个真实的 Python 实例来跑一遍逻辑
        # 但这个实例里的成员变量（self.H）实际上是 MLIR Value (Handle)
        # 我们需要捕获这些 Handle，并决定哪些需要持久化到 Context 中
        self._instance = None 

    def _compile(self, method_name, sample_args):
        print(f"[Neptune JIT] Tracing {self._cls.__name__}...")

        # -------------------------------------------------------------
        # Phase 1: Compile __init__ (创建上下文)
        # -------------------------------------------------------------
        self._compiler.start_function(f"{self._cls.__name__}_init", [])
        
        # 实例化类，这会执行 __init__，并在 IR 里插入 assemble_matrix
        self._instance = self._cls(*self._init_args, **self._init_kwargs)
        
        # 扫描 _instance 的成员，找到所有 Expr 类型的变量
        # 这些就是需要保存到 Context 里的“状态”
        state_vars = [] # [(name, handle), ...]
        for name, val in self._instance.__dict__.items():
            if isinstance(val, Expr):
                state_vars.append((name, val))
                print(f"  > Found state variable: self.{name}")

        # TODO: 这里应该生成一个 struct 并 alloc/store
        # MVP 简化：假设只有一个状态 (matrix)，直接返回它
        # 也就是说 init 函数返回 void* (矩阵指针)
        if state_vars:
            # 返回第一个状态
            self._compiler.create_return(state_vars[0][1]._handle)
        else:
            # 无状态，返回 0
            zero = self._compiler.create_constant(0.0)
            self._compiler.create_return(zero._handle)
            
        self._compiler.end_function()

        # -------------------------------------------------------------
        # Phase 2: Compile Method (step)
        # -------------------------------------------------------------
        # 我们需要把 Python 的 self.H 替换成从函数参数里 Load 出来的 Value
        # 但因为 _instance 已经持有 Handle 了，如果是 MVP 简化版（Context=Matrix），
        # 我们可以直接用 _instance.H 里的 handle，但这是不对的！
        # 因为那个 handle 是在 @init 函数里的 SSA Value，在 @step 里不可见！
        
        # 【关键修正】：跨函数传值
        # 在 @step 函数开头，我们需要声明一个 Argument 作为 Context
        arg_hints = []
        # Arg 0: Context (来自 init 的返回值)
        # 我们需要告诉 C++ 这是一个 Opaque Pointer 或者是 Matrix Type
        # 暂时用 create_wrap(None) 产生的 Dummy 来充当 Type Hint
        dummy_ctx = self._compiler.create_wrap(None, "ctx") 
        arg_hints.append(dummy_ctx._handle)
        
        # Args 1...N: 用户参数
        for arg in sample_args:
            if isinstance(arg, Expr):
                arg_hints.append(arg._handle)
        
        self._compiler.start_function(f"{self._cls.__name__}_{method_name}", arg_hints)
        
        # 更新 self._instance 里的成员变量！
        # 让 self.H 指向 @step 函数的第 0 个参数 (Context)
        # (假设 Context 就是 Matrix 本身)
        ctx_val_handle = self._compiler.get_function_arg(0)
        
        # 这是一个极其 Dirty 的 Hack，但对于 MVP 是最直观的：
        # 我们假设 self.H 就是那个 context。
        if state_vars:
            name, _ = state_vars[0]
            # 替换 instance 里的 handle
            setattr(self._instance, name, Expr(ctx_val_handle))
            
        # 准备方法参数 (从 Arg 1 开始)
        method_args = []
        for i in range(len(sample_args)):
            # +1 是因为第 0 个是 context
            val = self._compiler.get_function_arg(i + 1)
            method_args.append(Expr(val))
            
        # 调用方法
        method = getattr(self._instance, method_name)
        result = method(*method_args)
        
        if isinstance(result, Expr):
            self._compiler.create_return(result._handle)
            
        self._compiler.end_function()

        # -------------------------------------------------------------
        # Phase 3: JIT Compile
        # -------------------------------------------------------------
        self._compiled_lib = jit_compile(self._compiler)
        print(f"[Neptune JIT] Library ready: {self._compiled_lib}")
        
        # -------------------------------------------------------------
        # Phase 4: Runtime Initialization
        # -------------------------------------------------------------
        # 既然编译好了，我们立刻调用 init 函数来创建 Context
        init_func_name = f"{self._cls.__name__}_init"
        if hasattr(self._compiled_lib, init_func_name):
            init_func = getattr(self._compiled_lib, init_func_name)
            # 设置返回类型为 void* (Context 指针)
            init_func.restype = ctypes.c_void_p
            
            print(f"[Neptune Runtime] Initializing Context...")
            self._runtime_context = init_func()
            print(f"  > Context Handle: {self._runtime_context}")
        else:
            self._runtime_context = None

    def __getattr__(self, name):
        def method_proxy(*args, **kwargs):
            if not self._compiled_lib:
                self._compile(name, args)
            
            # 调用 C 函数
            func_name = f"{self._cls.__name__}_{name}"
            c_func = getattr(self._compiled_lib, func_name)
            
            # 组装参数：Context + User Args
            call_args = []
            if self._runtime_context is not None:
                call_args.append(self._runtime_context)
            
            # TODO: Handle NumPy -> C Pointers
            # 这里暂时假设 args 已经是 c_void_p 或者能被 ctypes 识别
            call_args.extend(args)
            
            print(f"[Neptune Runtime] Running {func_name}...")
            return c_func(*call_args)

        return method_proxy

def jit_class(cls):
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        return JITClassWrapper(cls, *args, **kwargs)
    return wrapper
