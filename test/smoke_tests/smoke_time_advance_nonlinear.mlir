// RUN: neptune-opt %s --neptuneir-to-llvm

#loc = #neptune_ir.location<"cell">
#b   = #neptune_ir.bounds<lb = [0], ub = [16]>

!temp  = !neptune_ir.temp<element = f64, bounds = #b, location = #loc>
!field = !neptune_ir.field<element = f64, bounds = #b, location = #loc>

module {
  // ------------------------------------------------------------
  // residual opdef: F(u_next; u_prev) -> temp
  // signature: (!temp, !temp) -> !temp
  //    arg0 = u_next (unknown / current iterate)
  //    arg1 = u_prev (capture)
  // ------------------------------------------------------------
  neptune_ir.nonlinear_opdef @ac_residual
    : (!temp, !temp) -> !temp {
  ^bb0(%u_next: !temp, %u_prev: !temp):

    // compute residual over full domain [0,16)
    %F = neptune_ir.apply(%u_next, %u_prev) attributes
          {bounds = #neptune_ir.bounds<lb = [0], ub = [16]>}
        : (!temp, !temp) -> !temp {
    // apply region args: (i, u_next, u_prev)
    ^bb0(%i: index, %un: !temp, %up: !temp):
      %c0  = arith.constant 0 : index
      %c15 = arith.constant 15 : index

      %isL = arith.cmpi eq, %i, %c0  : index
      %isR = arith.cmpi eq, %i, %c15 : index
      %isB = arith.ori %isL, %isR : i1

      // boundary: F = u_next - u_prev  (keep boundary fixed)
      %Fb = scf.if %isB -> (f64) {
        %un0 = neptune_ir.access %un[0] : !temp -> f64
        %up0 = neptune_ir.access %up[0] : !temp -> f64
        %r   = arith.subf %un0, %up0 : f64
        scf.yield %r : f64
      } else {
        // interior: fully-implicit Euler residual
        %um1 = neptune_ir.access %un[-1] : !temp -> f64
        %u0  = neptune_ir.access %un[0]  : !temp -> f64
        %up1 = neptune_ir.access %un[1]  : !temp -> f64
        %uold= neptune_ir.access %up[0]  : !temp -> f64

        // constants (smoke 固定参数)
        %two    = arith.constant 2.0 : f64
        %dxinv2 = arith.constant 100.0 : f64   // 1/dx^2, dx=0.1
        %dt     = arith.constant 1.0e-2 : f64
        %eps2   = arith.constant 1.0e-2 : f64

        // lap = (u[i-1] - 2u[i] + u[i+1]) / dx^2
        %t0  = arith.mulf %two, %u0 : f64
        %t1  = arith.subf %um1, %t0 : f64
        %t2  = arith.addf %t1, %up1 : f64
        %lap = arith.mulf %dxinv2, %t2 : f64

        // react = u - u^3
        %u2 = arith.mulf %u0, %u0 : f64
        %u3 = arith.mulf %u2, %u0 : f64
        %react = arith.subf %u0, %u3 : f64

        // rhs = eps2*lap + react
        %diff = arith.mulf %eps2, %lap : f64
        %rhs  = arith.addf %diff, %react : f64

        // F = u_next - u_prev - dt*rhs
        %dt_rhs = arith.mulf %dt, %rhs : f64
        %t3     = arith.subf %u0, %uold : f64
        %r      = arith.subf %t3, %dt_rhs : f64
        scf.yield %r : f64
      }

      neptune_ir.yield %Fb : f64
    }

    neptune_ir.return %F : !temp
  }

  func.func @entry(%out: memref<?xf64>, %in: memref<?xf64>) -> memref<?xf64> {
    %fout = neptune_ir.wrap %out : memref<?xf64> -> !field
    %fin  = neptune_ir.wrap %in  : memref<?xf64> -> !field
    %u0   = neptune_ir.load %fin : !field -> !temp

    %dt = arith.constant 1.0e-2 : f64

    // 你要测的点：implicit_nonlinear + residual=@ac_residual
    %u1 = neptune_ir.time_advance %u0, %dt {
            method   = 1 : i32,            // kImplicitNonlinear
            residual = @ac_residual,
            solver   = "newton",           // 你 runtime 自己解释
            tol      = 1.0e-10,
            max_iters= 20
          } : !temp, f64 -> !temp

    neptune_ir.store %u1 to %fout : !temp to !field
    %res = neptune_ir.unwrap %fout : !field -> memref<?xf64>
    func.return %res : memref<?xf64>
  }
}
