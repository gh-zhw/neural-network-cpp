#pragma once

enum class VariableOp {
    NONE = 0,

    _MATH_OP,
    ADD_MM,  // Matrix + Matrix
    ADD_MS,  // Matrix + Scalar
    ADD_SM,  // Scalar + Matrix
    SUB_MM,
    SUB_MS,
    MUL_MM,
    MUL_MS,
    MUL_SM,

    _FUNC_OP,
    RELU,
    SIGMOID,
    CROSS_ENTROPY_LOSS,
    MSE_LOSS
};
