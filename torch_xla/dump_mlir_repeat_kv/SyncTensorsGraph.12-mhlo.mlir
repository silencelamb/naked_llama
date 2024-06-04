module @"SyncTensorsGraph.12-mhlo" attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = true, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x2x4x3xf32>) -> (tensor<1x2x4x3xf32>, tensor<1x4x4x3xf32>) {
    %0 = mhlo.reshape %arg0 : (tensor<1x2x4x3xf32>) -> tensor<2x4x3xf32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 3, 4]> : tensor<3xi64>} : (tensor<2x4x3xf32>) -> tensor<1x2x2x4x3xf32>
    %2 = mhlo.reshape %1 : (tensor<1x2x2x4x3xf32>) -> tensor<1x4x4x3xf32>
    %3 = mhlo.tuple %arg0, %2 {xla_shape = "(f32[1,2,4,3]{3,2,1,0}, f32[1,4,4,3]{3,2,1,0})"} : tuple<tensor<1x2x4x3xf32>, tensor<1x4x4x3xf32>>
    return %arg0, %2 : tensor<1x2x4x3xf32>, tensor<1x4x4x3xf32>
  }
}