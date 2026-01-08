// RUN: neptune-opt %s --canonicalize

#loc = #neptune_ir.location<"cell">
#b   = #neptune_ir.bounds<lb = [0], ub = [32]>

!temp  = !neptune_ir.temp<element = f64, bounds = #b, location = #loc>
!field = !neptune_ir.field<element = f64, bounds = #b, location = #loc>

module {
  // A(v) = v - dt*(a v_xx + b v_x + c v)
  neptune_ir.linear_opdef @bs_A : (!temp) -> !temp {
  ^bb0(%v: !temp):
    %y = neptune_ir.apply(%v) attributes {bounds = #neptune_ir.bounds<lb = [1], ub = [31]>}
      : (!temp) -> !temp {
      // (index) + (v)
      ^bb0(%i: index, %v_in: !temp):
        %vm1 = neptune_ir.access %v_in[-1] : !temp -> f64
        %v0  = neptune_ir.access %v_in[0]  : !temp -> f64
        %vp1 = neptune_ir.access %v_in[1]  : !temp -> f64

        %dxinv2 = arith.constant 100.0 : f64  // 1/(0.1^2)
        %inv2dx = arith.constant 5.0   : f64  // 1/(2*0.1)

        // sigma=0.2, r=0.05 => a=0.02, b=0.03, c=-0.05
        %a  = arith.constant 2.0e-2  : f64
        %b  = arith.constant 3.0e-2  : f64
        %c  = arith.constant -5.0e-2 : f64
        %dt = arith.constant 1.0e-2  : f64

        // v_xx
        %two = arith.constant 2.0 : f64
        %t0  = arith.mulf %two, %v0 : f64
        %t1  = arith.subf %vm1, %t0 : f64
        %t2  = arith.addf %t1, %vp1 : f64
        %vxx = arith.mulf %dxinv2, %t2 : f64

        // v_x
        %t3  = arith.subf %vp1, %vm1 : f64
        %vx  = arith.mulf %inv2dx, %t3 : f64

        // L = a*vxx + b*vx + c*v0
        %t4  = arith.mulf %a, %vxx : f64
        %t5  = arith.mulf %b, %vx  : f64
        %t6  = arith.addf %t4, %t5 : f64
        %t7  = arith.mulf %c, %v0  : f64
        %L   = arith.addf %t6, %t7 : f64

        %dtL = arith.mulf %dt, %L : f64
        %out = arith.subf %v0, %dtL : f64
        neptune_ir.yield %out : f64
    }
    neptune_ir.return %y : !temp
  }

  func.func @entry(%out: memref<?xf64>, %in: memref<?xf64>) -> memref<?xf64> {
    %fout = neptune_ir.wrap %out : memref<?xf64> -> !field
    %fin  = neptune_ir.wrap %in  : memref<?xf64> -> !field
    %v0   = neptune_ir.load %fin : !field -> !temp

    %dt = arith.constant 1.0e-2 : f64
    %v1 = neptune_ir.time_advance %v0, %dt {
            method = 2 : i32,
            system = @bs_A,
            solver = "gmres",
            tol = 1.0e-10,
            max_iters = 500
          } : !temp, f64 -> !temp

    neptune_ir.store %v1 to %fout : !temp to !field
    %res = neptune_ir.unwrap %fout : !field -> memref<?xf64>
    func.return %res : memref<?xf64>
  }
}
