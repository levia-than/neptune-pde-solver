#ifndef NEPTUNEIR_NEPTUNEIRATTRS_TD
#define NEPTUNEIR_NEPTUNEIRATTRS_TD

include "mlir/IR/BuiltinAttributes.td"
include "NeptuneIRDialect.td"

// 迭代域 [lb[i], ub[i])，用于 interior / boundary / 子域等。
def NeptuneIR_BoundsAttr
    : NeptuneIR_Attr<"Bounds", "bounds"> {
  let summary = "iteration bounds [lb, ub) for stencil domains";
  let description = [{
    Per-dimension half-open interval [lb[i], ub[i)) describing a logical
    domain. Typical uses:
      - distinguish interior vs boundary regions
      - represent subdomains for splitting / scheduling
  }];

  let parameters =
    (ins "::mlir::DenseI64ArrayAttr":$lb,
         "::mlir::DenseI64ArrayAttr":$ub);

  let assemblyFormat = [{
    `<` `lb` `=` $lb `,` `ub` `=` $ub `>`
  }];
}

// 场在网格上的“位置”，给 FDM / FVM 用。
def NeptuneIR_LocationAttr
    : NeptuneIR_Attr<"Location", "location"> {
  let summary = "location of a field on the grid (cell/vertex/face)";
  let description = [{
    Encodes where the field lives with respect to the mesh:
      - "cell"   : cell-centered quantity
      - "vertex" : vertex-based quantity
      - "face_x" : face-normal to x
      - "face_y" : face-normal to y
      - "face_z" : face-normal to z

    This is useful for finite-volume schemes to distinguish states and
    fluxes.
  }];

  let parameters = (ins StringRefParameter<"location kind">:$kind);

  let assemblyFormat = [{
    `<` $kind `>`
  }];
}

// （可选）Stencil 形状描述：一组 offset 向量
// 例如 [ [-1,0,0], [0,0,0], [1,0,0] ] 表示 1D 三点 stencil。
def NeptuneIR_StencilShapeAttr
    : NeptuneIR_Attr<"StencilShape", "stencil_shape"> {
  let summary = "set of neighbor offsets for a stencil pattern";
  let description = [{
    A list of integer offset vectors. Each entry is a per-dimension offset
    from the current logical point, e.g.:

      #neptune_ir.stencil_shape<[
        [-1, 0, 0],
        [ 0, 0, 0],
        [ 1, 0, 0]
      ]>

    This is not mandatory for expressing kernels, but is convenient for
    passes that classify or optimize stencils based on their pattern.
  }];

  // 用 ArrayAttr 包 DenseI64ArrayAttr，简单好 parse/print。
  let parameters = (ins "::mlir::ArrayAttr":$offsets);

  let assemblyFormat = [{
    `<` $offsets `>`
  }];
}

#endif // NEPTUNEIR_NEPTUNEIRATTRS_TD
