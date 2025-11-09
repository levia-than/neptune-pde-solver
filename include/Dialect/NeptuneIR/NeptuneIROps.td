#ifndef NEPTUNEIR_NEPTUNEIROPS_TD
#define NEPTUNEIR_NEPTUNEIROPS_TD

// === NeptuneIR core ops/types (symbols + field-expr primitives) ===
// Append this file to your existing NeptuneIR .td set. It expects the
// NeptuneIR_Dialect and NeptuneIR_Type helper you already defined.
//
// Important includes for Symbol traits & builtin Attr/Types:
include "mlir/IR/Interfaces.td"          // SymbolOpInterface etc.
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"   // FlatSymbolRefAttr, SymbolNameAttr, FloatAttr
include "mlir/Interfaces/SideEffectInterfaces.td"
include "NeptuneIRDialect.td"


// ------------------------------------------------------------------
// Opaque runtime descriptor type: allows passing a descriptor Value if desired.
// Example: host/device pointer, strides, shape, device id, etc.
def DescriptorType : NeptuneIR_Type<"Descriptor", "desc"> {
  let summary = "Opaque runtime descriptor type for storage (device pointer/metadata).";
  let parameters = (ins "IntegerAttr":$device, "ArrayAttr":$shape);
  let assemblyFormat = "`<` `device` `=` $device `,` `shape` `=` $shape `>`";
  let description = [{
    Example textual forms:
      !neptune_ir.desc
      !neptune_ir.desc<device=0>
      !neptune_ir.desc<device=0,shape=[128,256]>

    The parameters are optional at the textual level; the generated parser will
    accept the angle-bracket form. If you need stronger invariants or custom
    uniquing of the parameters, implement a TypeStorage in C++ (the ODS will
    still generate the parse/print glue).
  }];
}

// Field element reference type (symbolic element-level value).
// This represents values in the high-level symbolic layer before lowering to scalars.
def FieldElementRefType : NeptuneIR_Type<"FieldElementRef", "field_elem"> {
  let summary = "High-level per-element reference into a Field (symbolic).";
  let parameters = (ins "Type":$elemType);
    // Examples:
  //   !neptune_ir.field_elem
  //   !neptune_ir.field_elem<etype=f32>
  let assemblyFormat = "`<` `etype` `=` $elemType `>`";
  let description = [{
    The `etype` parameter is an MLIR Type, e.g. f32, i32. If omitted the type
    represents a generic field element (no stored scalar type).
  }];
}

// FieldRef: per-element symbolic reference into a provided storage Value.
// The first operand should be a memref Value or an opaque Descriptor value.
// Subsequent operands are index Values (typically Index type) describing the logical indices.
def NeptuneIR_FieldRefOp : NeptuneIR_Op<"field.ref", [NoMemoryEffect]> {
  let summary = "Per-element reference into a provided storage value (memref/descriptor).";
  let arguments = (ins AnyType:$storageValue, Variadic<Index>:$indices);
  let results = (outs FieldElementRefType:$elem);
  let assemblyFormat = "$storageValue `[` $indices `]` `:` type($storageValue) `->` type($elem) attr-dict ";
  let description = [{
    %e = neptune_ir.field.ref %storage, %i, %j : returns a symbolic element ref.
    The storage operand is a Value (memref or descriptor) rather than a symbol.
    Lowering maps this op to memref.load with appropriate index math and halo offsets.
  }];
}

// def NeptuneIR_TimeLoopOp : NeptuneIR_Op<"time.loop"> {
//   let summary = "Time-stepping loop with constant dt attribute.";
//   let arguments = (ins Index:$t_start, Index:$t_end, F64Attr:$dt);
//   let regions = (region AnyRegion:$body);
//   let assemblyFormat = "$t_start `=` $t_end `,` $t_step `:` type($t_start) `,` type($t_end) `,` type($t_step) attr-dict $body";
// }

// def NeptuneIR_MeshLoopOp : NeptuneIR_Op<"mesh.loop"> {
//   let summary = "Mesh loop.";
//   let arguments = (ins Index:$i_start, Index:$i_end, Index:$t_step);
//   let regions = (region AnyRegion:$body);
//   let assemblyFormat = "$i_start `,` $i_end `,` $i_step `:` type($i_start) `,` type($i_end) `,` type($i_step) attr-dict $body";
// }

// Field-level binary ops (symbolic): these accept and produce FieldElementRefType.
// They represent per-element algebraic operations in the symbolic layer.
def NeptuneIR_FieldAddOp : NeptuneIR_Op<"field.add", [NoMemoryEffect]> {
  let summary = "Symbolic per-element add: res = lhs + rhs";
  let arguments = (ins FieldElementRefType:$lhs, FieldElementRefType:$rhs);
  let results = (outs FieldElementRefType:$res);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($res) attr-dict ";
}

def NeptuneIR_FieldSubOp : NeptuneIR_Op<"field.sub", [NoMemoryEffect]> {
  let summary = "Symbolic per-element subtract: res = lhs - rhs";
  let arguments = (ins FieldElementRefType:$lhs, FieldElementRefType:$rhs);
  let results = (outs FieldElementRefType:$res);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($res) attr-dict ";
}

def NeptuneIR_FieldMulOp : NeptuneIR_Op<"field.mul", [NoMemoryEffect]> {
  let summary = "Symbolic per-element multiply: res = lhs * rhs";
  let arguments = (ins FieldElementRefType:$lhs, FieldElementRefType:$rhs);
  let results = (outs FieldElementRefType:$res);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($res) attr-dict ";
}

def NeptuneIR_FieldDivOp : NeptuneIR_Op<"field.div", [NoMemoryEffect]> {
  let summary = "Symbolic per-element divide: res = lhs / rhs";
  let arguments = (ins FieldElementRefType:$lhs, FieldElementRefType:$rhs);
  let results = (outs FieldElementRefType:$res);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($res) attr-dict ";
}

// Scalar scaling by a FloatAttr constant (symbolic)
def NeptuneIR_FieldScaleOp : NeptuneIR_Op<"field.scale", [NoMemoryEffect]> {
  let summary = "Symbolic per-element scaling: res = lhs * scalarAttr";
  let arguments = (ins FieldElementRefType:$lhs, F64Attr:$scalar);
  let results = (outs FieldElementRefType:$res);
  let description = [{
    The scalar is a FloatAttr on the op, representing a compile-time constant.
    If you need runtime scalars, consider exposing a scalar Value operand instead.
  }];
  let assemblyFormat = "$lhs attr-dict `:` type($lhs) `->` type($res)";
}

// Evaluate: materialize a symbolic expression into a provided destination Value.
// Destination is a memref or descriptor Value (first operand). The second operand
// is the symbolic FieldElementRefType expression to materialize.
def NeptuneIR_EvaluateOp : NeptuneIR_Op<"evaluate"> {
  let summary = "Materialize a FieldExpr into a provided destination memref/descriptor.";
  let arguments = (ins AnyType:$dstValue, FieldElementRefType:$expr);
  let description = [{
    Lowering of evaluate generates nested loops that perform memref.load/arith/memref.store,
    writing the element-level result into the provided destination storage Value.
  }];
  let assemblyFormat = "$dstValue `,` $expr `:` type($dstValue) `,` type($expr) attr-dict ";
}

// Swap: semantic swap of two storage values (memref/descriptor).
// Lowering may perform pointer swap or element-wise copy depending on backend.
def NeptuneIR_SwapOp : NeptuneIR_Op<"swap"> {
  let summary = "Swap two storage Values (memref/descriptors).";
  let arguments = (ins AnyType:$aValue, AnyType:$bValue);
  let description = [{
    Swap semantics: lowering decides whether to perform a descriptor/pointer swap
    or a copy (e.g., for incompatible layouts). Optional debugging/print features
    can be implemented as runtime calls inserted by lowering.
  }];
  let assemblyFormat = "$aValue `,` $bValue `:` type($aValue) `,` type($bValue) attr-dict ";
}

// Optional: utility ops for reading/writing descriptor metadata (device id, stride, etc).
// These are left generic and can be specialized if needed.
def NeptuneIR_GetDescFieldOp : NeptuneIR_Op<"desc.get"> {
  let summary = "Get a field from an opaque descriptor (example utility op).";
  let arguments = (ins DescriptorType:$desc, StrAttr:$key);
  let results = (outs AnyType:$value);
  let description = [{
    This is a utility op that a runtime/descriptor lowering may expand into field loads.
  }];
}

def NeptuneIR_SetDescFieldOp : NeptuneIR_Op<"desc.set"> {
  let summary = "Set a field on an opaque descriptor (example utility op).";
  let arguments = (ins DescriptorType:$desc, StrAttr:$key, AnyType:$value);
}

#endif
