#ifndef NPTIMPL_NPTIMPLDIALECT_TD
#define NPTIMPL_NPTIMPLDIALECT_TD

include "mlir/IR/OpBase.td"

def NptImpl_Dialect : Dialect {
    let name = "npt_impl";
    let cppNamespace = "::mlir::Neptune::NptImpl";
}

class NptImpl_Op<string mnemonic, list<Trait> traits = []> :
    Op<NptImpl_Dialect, mnemonic, traits>;

#endif // NPTIMPL_NPTIMPLDIALECT_TD
