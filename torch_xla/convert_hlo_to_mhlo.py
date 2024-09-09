
"""
pip install jax
"""
from jax._src.lib import xla_extension as xe
from jax._src.lib import xla_client as xc


# 读取HLO文件
hlo_file_path = './mnist_ddp.hlo'
with open(hlo_file_path, 'r') as hlo_file:
    hlo_text = hlo_file.read()

"""
解析HLO文本, 构建hlomodule
https://github.com/openxla/xla/blob/39a982c1757bbb7136431b2df48d067f122c5190/xla/python/xla_extension/__init__.pyi#L221 
    def hlo_module_from_text(hlo_module_text: str) -> HloModule: ...

https://github.com/openxla/xla/blob/39a982c1757bbb7136431b2df48d067f122c5190/xla/python/xla_compiler.cc#L833
    m.def("hlo_module_from_text",
    ps: 里面还有很多有用函数和基础类，比如 HloModule
"""
# 
# 
hlo_module = xe.hlo_module_from_text(hlo_text)

# 转为XlaComputation
# https://github.com/alpa-projects/alpa/blob/b8078a9f75cb4c90cabb4550ee48c99ef394e209/alpa/wrapped_hlo.py#L39
xla_computation = xe.XlaComputation(hlo_module.as_serialized_hlo_module_proto())


"""
step2. 转换为MHLO
https://github.com/openxla/xla/blob/39a982c1757bbb7136431b2df48d067f122c5190/xla/python/xla_extension/mlir.pyi#L19
    def xla_computation_to_mlir_module(
    computation: XlaComputation, emit_stable_hlo: bool = ...
    ) -> str: ...
    ps: 里面还有很多有用函数和基础类，比如 mhlo_to_stablehlo, stablehlo_to_mhlo
    
https://github.com/openxla/xla/blob/39a982c1757bbb7136431b2df48d067f122c5190/xla/python/mlir.cc#L241 
    mlir_module.def("xla_computation_to_mlir_module",
    
"""

mhlo_module = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation, False)

import pdb; pdb.set_trace()
# 打印MHLO
with open(hlo_file_path.replace('.hlo', '.mhlo'), 'w') as mhlo_file:
    mhlo_file.write(mhlo_module)