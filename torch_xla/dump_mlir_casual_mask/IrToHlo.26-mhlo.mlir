module @"IrToHlo.26-mhlo" attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = true, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>) -> (tensor<2x2xf32>, tensor<5x8xf32>) {
    %0 = mhlo.copy %arg0 : tensor<2x2xf32>
    %1 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5x5xi32>
    %2 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<5x5xi32>
    %3 = mhlo.compare  GE, %1, %2 : (tensor<5x5xi32>, tensor<5x5xi32>) -> tensor<5x5xi1>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x5xf32>
    %6 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x5xf32>
    %7 = mhlo.select %3, %5, %6 : tensor<5x5xi1>, tensor<5x5xf32>
    %8 = "mhlo.pad"(%7, %4) {edge_padding_high = dense<0> : tensor<2xi64>, edge_padding_low = dense<[0, 3]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<5x5xf32>, tensor<f32>) -> tensor<5x8xf32>
    %9 = mhlo.tuple %0, %8 {xla_shape = "(f32[2,2]{1,0}, f32[5,8]{1,0})"} : tuple<tensor<2x2xf32>, tensor<5x8xf32>>
    return %0, %8 : tensor<2x2xf32>, tensor<5x8xf32>
  }
}