#ifndef NPTIMPL_NPTIMPL_OPS_TD
#define NPTIMPL_NPTIMPL_OPS_TD

include "mlir/IR/OpBase.td"
include "NptImplDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NptImpl_PolyMeshOp : NptImpl_Op<"poly_mesh", []> {
    let summary = "A poly mesh op";
    let description = [{
        A poly mesh op
    }];

    let arguments = (ins 
        // variadic SSA values for all unknown fields stored at this cell
        Variadic<AnyType>:$values,

        // a defaulted integer attribute used for scheduling/advanced solving strategies
        // DefaultValuedAttr<IntegerAttr, 0> keeps a default of 0 when the attr is omitted.
        DefaultValuedAttr<I64Attr, "0">:$timestamp,

        // flattened cell index (i64 attribute)
        I64Attr:$index,

        // optional list/array of neighbor indices (use DenseI64ArrayAttr or ArrayAttr of I64Attrs as you prefer)
        OptionalAttr<ArrayAttr>:$neighbors,

        // whether this cell is a boundary cell
        OptionalAttr<BoolAttr>:$isBound,

        // optional enum Attr defined elsewhere (replace with the real enum attr type name)
        OptionalAttr<I64Attr>:$texture
    );

    let results = (outs);

    let assemblyFormat = [{
        `(` ($values^ `:` type($values))? `)` attr-dict
    }];

    let hasVerifier = 1;
}

#endif
