
#include "../include/variable.hpp"
#include "../include/activations.hpp"


Variable relu(const Variable& var)
{
    Variable newVar(var.value, var.require_grad_);

    for (size_t i = 0; i < newVar.h(); ++i) {
        for (size_t j = 0; j < newVar.w(); ++j) {
            // ReLU(x) = max(0, x)
            newVar.set(i, j, std::max(0.0f, newVar.get(i, j)));
        }
    }

    newVar.parent.push_back(&var);
    newVar.op_ = VariableOp::RELU;

    return newVar;
}

