import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client, xla_bridge
from jax.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
ops = xla_client.ops

"""
Use jax to build xla computation

1. Test all-gather

# Note:
jax version      0.4.30
jaxlib version   0.4.30
pip install --upgrade jax==0.4.30 jaxlib==0.4.30+cpu -f https://storage.googleapis.com/jax-releases/jax_releases.html
或者直接装最新的   pip install -U jax
Ref: 
https://github.com/alpa-projects/alpa/blob/b8078a9f75cb4c90cabb4550ee48c99ef394e209/alpa/mesh_profiling.py#L228
https://github.com/alpa-projects/alpa/blob/b8078a9f75cb4c90cabb4550ee48c99ef394e209/playground/xla_builder/test_multi_host.py#L19
https://github.com/alpa-projects/alpa/blob/b8078a9f75cb4c90cabb4550ee48c99ef394e209/playground/xla_builder/test_xla_builder.py#L104
"""

def parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(np.dtype(dtype), shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)

def _op_parameter(builder, num, shape, dtype):
    shape = xc.Shape.array_shape(dtype, shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)


def _create_channel_id(backend):
    channel_id = backend.create_channel_handle()
    channel_id.type = xe.ChannelHandle_ChannelType.DEVICE_TO_DEVICE
    channel_id.handle = 1
    return channel_id

def _op_all_gather(operand, all_gather_dim, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    if channel_id is None:
        ret = ops.AllGather(operand, all_gather_dim, len(replica_groups[0]),
                            replica_groups_protos, channel_id, None, False)
    else:
        ret = ops.AllGather(operand, all_gather_dim, len(replica_groups[0]),
                            replica_groups_protos, channel_id, None, True)
    return ret

def _op_all_reduce(operand, reduce_op, replica_groups):
    replica_groups_protos = xla_client.make_replica_groups(replica_groups)
    if reduce_op == 'add':
        rc = xla_client.XlaBuilder("reduce_" + reduce_op)
        x = parameter(rc, 0, (), np.float32)
        y = parameter(rc, 1, (), np.float32)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    return ops.AllReduce(operand, rc, replica_groups_protos,
            None, None)


def _op_all_to_all(operand, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    ret = ops.AllToAll(operand, 0, 0, len(replica_groups[0]),
                       replica_groups_protos, channel_id, None, True)
    return ret


def _op_reduce_scatter(operand, dtype, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    if reduce_op == "add":
        rc = xc.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(rc, 0, (), dtype)
        y = _op_parameter(rc, 1, (), dtype)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.ReduceScatter(operand, rc, 0, len(replica_groups[0]),
                            replica_groups_protos, channel_id, None, True)
    return ret


def parameter(builder, num, shape, dtype):
    shape = xla_client.Shape.array_shape(np.dtype(dtype), shape)
    return ops.Parameter(builder, num, shape.with_major_to_minor_layout_if_absent(), "", [])


def test_all_gather(backend, num_replicas):
    full_shape = (8, 2)
    all_gather_dim = 0
    
    # 构建计算图
    operand_shape = (full_shape[0]//num_replicas, 2)
    c = xla_client.XlaBuilder("all_gather_test")
    x = parameter(c, 0, operand_shape, np.float32)
    # channel_id = _create_channel_id(backend)
    channel_id = None
    replica_groups=[list(range(num_replicas))]
    # replica_groups=[[2,1,3,0]]
    result = _op_all_gather(
        x,
        all_gather_dim=all_gather_dim,
        replica_groups=replica_groups,
        channel_id=channel_id
    )
    xla_computation = c.build(result)
    
    
    # 打印 HLO
    print(xla_computation.as_hlo_text())
    
    # 新版本的compile要输入一个mlir module或者string，不支持xla_computation jax_lib=0.4.30
    # 旧版本是直接输入xla_computation jax_lib=0.3.25
    mhlo_module = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation, False)
    print("MHLO Module: \n", mhlo_module)
    
    # 编译选项
    device_assignment = np.array([[i] for i in range(num_replicas)], dtype=np.int32)
    device_assignment = xla_client.DeviceAssignment.create(device_assignment)
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.device_assignment = device_assignment
    
    # 编译计算图
    executable = backend.compile(mhlo_module, compile_options)
    
    # 准备输入数据
    input_data = np.ones(operand_shape, dtype=np.float32)
    device_inputs = []
    device_inputs.append([
        backend.buffer_from_pyval(input_data * (i + 1), backend.devices()[i])
        for i in range(num_replicas)
    ])
    
    print("Device inputs:")
    for i, inp in enumerate(device_inputs[0]):
        print(f"Device {i}:\n", inp)
    
    # 执行计算
    outputs = executable.execute_sharded_on_local_devices(device_inputs)
    
    # 打印结果
    print("Output shapes:", [out.shape for out in outputs[0]])
    print("Output values:")
    for i, out in enumerate(outputs[0]):
        # 尝试多种可能的转换方法
        print(f"Device {i}:\n", out)

def test_reduce_scatter(backend, num_replicas):
    full_shape = (8, 2)
    reduce_op = "add"
    
    # 构建计算图
    c = xla_client.XlaBuilder("reduce_scatter_test")
    x = parameter(c, 0, full_shape, np.float32)
    channel_id = None
    replica_groups = [list(range(num_replicas))]
    result = _op_reduce_scatter(
        x,
        dtype=np.float32,
        reduce_op=reduce_op,
        replica_groups=replica_groups,
        channel_id=channel_id
    )
    xla_computation = c.build(result)
    
    # 打印 HLO
    print(xla_computation.as_hlo_text())
    
    # 新版本的compile要输入一个mlir module或者string，不支持xla_computation jax_lib=0.4.30
    # 旧版本是直接输入xla_computation jax_lib=0.3.25
    mhlo_module = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation, False)
    print("MHLO Module: \n", mhlo_module)
    
    # 编译选项
    device_assignment = np.array([[i] for i in range(num_replicas)], dtype=np.int32)
    device_assignment = xla_client.DeviceAssignment.create(device_assignment)
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    build_options.num_replicas = num_replicas
    build_options.device_assignment = device_assignment
    
    # 编译计算图
    executable = backend.compile(mhlo_module, compile_options)
    
    # 准备输入数据
    input_data = np.ones(full_shape, dtype=np.float32)
    device_inputs = [
        backend.buffer_from_pyval(input_data, backend.devices()[i])
        for i in range(num_replicas)
    ]
    
    print("Device inputs:")
    for i, inp in enumerate(device_inputs):
        print(f"Device {i}:\n", inp)
    
    # 执行计算
    outputs = executable.execute_sharded_on_local_devices(device_inputs)
    
    # 打印结果
    print("Output shapes:", [out.shape for out in outputs])
    print("Output values:")
    for i, out in enumerate(outputs):
        print(f"Device {i}:\n", out)
        
def test_sin_cos():
    def f(x):
        return jax.numpy.sin(jax.numpy.cos(x.T))

    c = jax.xla_computation(f)(np.ones((10,8)))

    gpu_backend = xla_bridge.get_backend("gpu")
    compiled_computation = gpu_backend.compile(c)

    print(c.as_hlo_text())
    print(compiled_computation.hlo_modules()[0].to_string())

    host_input = np.ones((10,8), dtype=np.float32)
    device_input = gpu_backend.buffer_from_pyval(host_input)
    device_out = compiled_computation.execute([device_input,])

if __name__ == "__main__":
    num_replicas = 4
    # 获取backend 和设备数量
    platform = 'CPU'  # or 'CPU'
    if platform == 'GPU':
        backend = xla_bridge.get_backend("gpu")
    else:
        # 设置环境变量
        # 需要设置后再import 库，否则无效
        # 也可以在外面设置shell环境变量 export XLA_FLAGS="--xla_force_host_platform_device_count=4"
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_replicas}"
        from jax.lib import xla_client, xla_bridge
        backend = xla_bridge.get_backend("cpu")
    
    print("Backend:", backend.platform)
    print("Device count:", backend.device_count())
    # test_all_gather(backend, num_replicas)
    test_reduce_scatter(backend, num_replicas)
    # test_sin_cos()