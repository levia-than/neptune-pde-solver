// RUN: neptune-opt %s --canonicalize

#loc = #neptune_ir.location<"cell">
#b   = #neptune_ir.bounds<lb = [0], ub = [16]>

!temp  = !neptune_ir.temp<element = f64, bounds = #b, location = #loc>
!field = !neptune_ir.field<element = f64, bounds = #b, location = #loc>

module {
  // Lap(u)[i] = (u[i-1] - 2u[i] + u[i+1]) / dx^2
  neptune_ir.linear_opdef @ac_lap : (!temp) -> !temp {
  ^bb0(%u: !temp):
    %lap = neptune_ir.apply(%u) attributes {bounds = #neptune_ir.bounds<lb = [1], ub = [15]>}
      : (!temp) -> !temp {
      // apply block args: (index) + (inputs...)
      ^bb0(%i: index, %u_in: !temp):
        %um1 = neptune_ir.access %u_in[-1] : !temp -> f64
        %u0  = neptune_ir.access %u_in[0]  : !temp -> f64
        %up1 = neptune_ir.access %u_in[1]  : !temp -> f64

        %two    = arith.constant 2.0 : f64
        %dxinv2 = arith.constant 100.0 : f64  // 1/(0.1^2), test

        %t0 = arith.mulf %two, %u0 : f64
        %t1 = arith.subf %um1, %t0 : f64
        %t2 = arith.addf %t1, %up1 : f64
        %lap_i = arith.mulf %dxinv2, %t2 : f64
        neptune_ir.yield %lap_i : f64
      }
    neptune_ir.return %lap : !temp
  }

  // A(x) = x - alpha * Lap(x), alpha = dt*eps2 (这里用常量 alpha 以满足 linear 区域限制)
  neptune_ir.linear_opdef @ac_A : (!temp) -> !temp {
  ^bb0(%x: !temp):
    %lapx = neptune_ir.apply_linear @ac_lap(%x) : (!temp) -> !temp

    %y = neptune_ir.apply(%x, %lapx) attributes {bounds = #neptune_ir.bounds<lb = [1], ub = [15]>}
      : (!temp, !temp) -> !temp {
      // (index) + (x, lapx)
      ^bb0(%i: index, %x_in: !temp, %lap_in: !temp):
        %x0   = neptune_ir.access %x_in[0]   : !temp -> f64
        %lap0 = neptune_ir.access %lap_in[0] : !temp -> f64

        %alpha = arith.constant 1.0e-4 : f64 // dt*eps2 = 1e-2 * 1e-2
        %t  = arith.mulf %alpha, %lap0 : f64
        %out = arith.subf %x0, %t : f64
        neptune_ir.yield %out : f64
      }
    neptune_ir.return %y : !temp
  }

  func.func @entry(%out: memref<?xf64>, %in: memref<?xf64>) -> memref<?xf64> {
    %fout = neptune_ir.wrap %out : memref<?xf64> -> !field
    %fin  = neptune_ir.wrap %in  : memref<?xf64> -> !field
    %u0   = neptune_ir.load %fin : !field -> !temp

    // u* = u0 + dt*(u0 - u0^3)  (显式反应)
    %ustar = neptune_ir.apply(%u0) attributes {bounds = #neptune_ir.bounds<lb = [1], ub = [15]>}
      : (!temp) -> !temp {
      ^bb0(%i: index, %u_in: !temp):
        %u  = neptune_ir.access %u_in[0] : !temp -> f64
        %dt = arith.constant 1.0e-2 : f64
        %u2 = arith.mulf %u, %u : f64
        %u3 = arith.mulf %u2, %u : f64
        %react = arith.subf %u, %u3 : f64
        %dt_react = arith.mulf %dt, %react : f64
        %u_out = arith.addf %u, %dt_react : f64
        neptune_ir.yield %u_out : f64
      }

    // (I - alpha*Lap) u_{n+1} = u*
    %dt = arith.constant 1.0e-2 : f64
    %u1 = neptune_ir.time_advance %ustar, %dt {
            method = 2 : i32,
            system = @ac_A,
            solver = "gmres",
            tol = 1.0e-8,
            max_iters = 200
          } : !temp, f64 -> !temp

    neptune_ir.store %u1 to %fout : !temp to !field
    %res = neptune_ir.unwrap %fout : !field -> memref<?xf64>
    func.return %res : memref<?xf64>
  }
}

