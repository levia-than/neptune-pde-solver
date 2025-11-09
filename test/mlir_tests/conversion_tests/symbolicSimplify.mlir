// RUN: neptune-opt %s -verify-diagnostics 2>&1
// XFAILS: *

module {
	func.func @heat_sim_single_step_kernel_broder_1(%u_old: memref<16xf64>, %u_new:memref<16xf64>) {
		%c0 = arith.constant 0 : index
		%expr = neptune_ir.field.ref %u_old[%c0]: memref<16xf64> -> !neptune_ir.field_elem<etype=f64>
		neptune_ir.evaluate %u_new, %expr: memref<16xf64>, !neptune_ir.field_elem<etype=f64>
		return
	}

	func.func @heat_sim_single_step_kernel_broder_2(%u_old: memref<16xf64>, %u_new:memref<16xf64>) {
		%c15 = arith.constant 15 : index
		%expr = neptune_ir.field.ref %u_old[%c15]: memref<16xf64> -> !neptune_ir.field_elem<etype=f64>
		neptune_ir.evaluate %u_new, %expr: memref<16xf64>, !neptune_ir.field_elem<etype=f64>
		return
	}

	func.func @heat_sim_single_step_kernel(%u_old: memref<16xf64>, %u_new:memref<16xf64>){
		%c0 = arith.constant 0 : index
		%c1 = arith.constant 1 : index
		%c15 = arith.constant 15 : index
		%start = arith.addi %c0, %c1 : index
		%end = arith.subi %c15, %c1 : index
		
		scf.for %i = %start to %end step %c1 : index {
			%ni_1 = arith.addi %i, %c1 : index
			%ni_2 = arith.subi %i, %c1 : index
			%fe_i = neptune_ir.field.ref %u_old[%i] : memref<16xf64> -> !neptune_ir.field_elem<etype=f64> 
			%fe_ip1 = neptune_ir.field.ref %u_old[%ni_1] : memref<16xf64> -> !neptune_ir.field_elem<etype=f64>
			%fe_im1 = neptune_ir.field.ref %u_old[%ni_2] : memref<16xf64> -> !neptune_ir.field_elem<etype=f64>
			%t1 = neptune_ir.field.sub %fe_ip1, %fe_i : !neptune_ir.field_elem<etype=f64>, !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>
			%t2 = neptune_ir.field.sub %fe_i, %fe_im1 : !neptune_ir.field_elem<etype=f64>, !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>
			%lap = neptune_ir.field.sub %t1, %t2 : !neptune_ir.field_elem<etype=f64>, !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>
			%s1 = neptune_ir.field.scale %lap { scalar = 0.5 } : !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>
			%s2 = neptune_ir.field.scale %s1 { scalar = 0.25 } : !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>
			%expr = neptune_ir.field.add %fe_i, %s2 : !neptune_ir.field_elem<etype=f64>, !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>
			neptune_ir.evaluate %u_new, %expr : memref<16xf64>, !neptune_ir.field_elem<etype=f64>
			scf.yield
		}
		return
	}

	func.func @heat_sim(%input: memref<16xf64>, %total_time: index) -> memref<16xf64> {
		%c0 = arith.constant 0 : index
		%c1 = arith.constant 1 : index
		%c2 = arith.constant 2 : index
		%ping_buf = memref.alloc() : memref<16xf64>
		%pong_buf = memref.alloc() : memref<16xf64>

		memref.copy %input, %ping_buf : memref<16xf64> to memref<16xf64>

		scf.for %time_step = %c0 to %total_time step %c1 : index {
			%is_even_step_idx = arith.remui %time_step, %c2 : index
			%is_even_step = arith.cmpi eq, %is_even_step_idx, %c0 : index

			%u_old = scf.if %is_even_step -> (memref<16xf64>) {
				scf.yield %ping_buf : memref<16xf64>
			} else {
				scf.yield %pong_buf : memref<16xf64>
			}

			%u_new = scf.if %is_even_step -> (memref<16xf64>) {
				scf.yield %pong_buf : memref<16xf64>
			} else {
				scf.yield %ping_buf : memref<16xf64>
			}

			func.call @heat_sim_single_step_kernel_broder_1(%u_old, %u_new) : (memref<16xf64>, memref<16xf64>) -> ()
			func.call @heat_sim_single_step_kernel_broder_2(%u_old, %u_new) : (memref<16xf64>, memref<16xf64>) -> ()
			func.call @heat_sim_single_step_kernel(%u_old, %u_new) : (memref<16xf64>, memref<16xf64>) -> ()
		}

		%is_total_time_odd_idx = arith.remui %total_time, %c2 : index
		%is_total_time_odd = arith.cmpi eq, %is_total_time_odd_idx, %c0 : index

		%final_result = scf.if %is_total_time_odd -> (memref<16xf64>) {
			scf.yield %pong_buf : memref<16xf64>
		} else {
			scf.yield %ping_buf : memref<16xf64>
		}

		scf.if %is_total_time_odd {
			memref.dealloc %ping_buf : memref<16xf64>
		} else {
			memref.dealloc %pong_buf : memref<16xf64>
		}

		return %final_result : memref<16xf64>
	}
}