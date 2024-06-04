module @"IrToHlo.12-mhlo" attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = true, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x2x4x3xf32>) -> (tensor<1x2x4x3xf32>, tensor<1x4x4x3xf32>) {
    %0 = mhlo.copy %arg0 : tensor<1x2x4x3xf32>
    %1 = mhlo.reshape %arg0 : (tensor<1x2x4x3xf32>) -> tensor<2x4x3xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[1, 3, 4]> : tensor<3xi64>} : (tensor<2x4x3xf32>) -> tensor<1x2x2x4x3xf32>
    %3 = mhlo.reshape %2 : (tensor<1x2x2x4x3xf32>) -> tensor<1x4x4x3xf32>
    %4 = mhlo.tuple %0, %3 {xla_shape = "(f32[1,2,4,3]{3,2,1,0}, f32[1,4,4,3]{3,2,1,0})"} : tuple<tensor<1x2x4x3xf32>, tensor<1x4x4x3xf32>>
    return %0, %3 : tensor<1x2x4x3xf32>, tensor<1x4x4x3xf32>
  }
}