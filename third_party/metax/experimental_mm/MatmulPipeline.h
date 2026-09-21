#pragma once

#include "llvm/IR/Module.h"

unsigned optimizeMatmulPipeline(llvm::Module &module, unsigned rows = 0,
                                unsigned columns = 0, unsigned reduction = 0,
                                unsigned groupRows = 1, unsigned splits = 1);
