#ifndef NEPTUNEIR_NEPTUNEIRDIALECT_TD
#define NEPTUNEIR_NEPTUNEIRDIALECT_TD

include "mlir/IR/OpBase.td"

def NeptuneIR_Dialect : Dialect {
    let name = "neptune_ir";
    let cppNamespace = "::mlir::Neptune::NeptuneIR";
    let summary = "neptune ir";
    let useDefaultTypePrinterParser = 1;
}

class NeptuneIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<NeptuneIR_Dialect, mnemonic, traits>;

class NeptuneIR_Type<string name, string typeMnemonic, list<Trait> traits = []> 
  : TypeDef<NeptuneIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic; 
}

#endif
