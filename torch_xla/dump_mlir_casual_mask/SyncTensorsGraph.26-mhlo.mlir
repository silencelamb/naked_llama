module @"SyncTensorsGraph.26-mhlo" attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = true, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>) -> (tensor<2x2xf32>, tensor<5x8xf32>) {
    %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5x5xi32>
    %1 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<5x5xi32>
    %2 = mhlo.compare  GE, %0, %1 : (tensor<5x5xi32>, tensor<5x5xi32>) -> tensor<5x5xi1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x5xf32>
    %5 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x5xf32>
    %6 = mhlo.select %2, %4, %5 : tensor<5x5xi1>, tensor<5x5xf32>
    %7 = "mhlo.pad"(%6, %3) {edge_padding_high = dense<0> : tensor<2xi64>, edge_padding_low = dense<[0, 3]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<5x5xf32>, tensor<f32>) -> tensor<5x8xf32>
    %8 = mhlo.tuple %arg0, %7 {xla_shape = "(f32[2,2]{1,0}, f32[5,8]{1,0})"} : tuple<tensor<2x2xf32>, tensor<5x8xf32>>
    return %arg0, %7 : tensor<2x2xf32>, tensor<5x8xf32>
  }
}