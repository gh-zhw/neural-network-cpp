#include <stdexcept>

#include "../include/variable.hpp"
#include "../include/operations.hpp"


void Variable::set(size_t i, size_t j, float val)
{
    if (i >= value.row() || j >= value.col()) {
        throw std::out_of_range("Variable::set: index out of range");
    }
    value.set(i, j, val);
}
void Variable::assign(float val)
{
    value.assign(val);
}

Variable Variable::operator+(const Variable& var) const {
    Variable newVar(value + var.value, require_grad_ || var.require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(&var);
    newVar.op_ = VariableOp::ADD_MM;
    return newVar;
}
Variable Variable::operator-(const Variable& var) const {
    Variable newVar(value - var.value, require_grad_ || var.require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(&var);
    newVar.op_ = VariableOp::SUB_MM;
    return newVar;
}
Variable Variable::operator*(const Variable& var) const {
    Variable newVar(value * var.value, require_grad_ || var.require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(&var);
    newVar.op_ = VariableOp::MUL_MM;
    return newVar;
}

Variable Variable::operator+(float scalar) const {
    Variable newVar(value + scalar, require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(scalar);
    newVar.op_ = VariableOp::ADD_MS;
    return newVar;
}
Variable Variable::operator-(float scalar) const {
    Variable newVar(value - scalar, require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(scalar);
    newVar.op_ = VariableOp::SUB_MS;
    return newVar;
}
Variable Variable::operator*(float scalar) const {
    Variable newVar(value * scalar, require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(scalar);
    newVar.op_ = VariableOp::MUL_MS;
    return newVar;
}
Variable Variable::operator/(float scalar) const {
    Variable newVar(value / scalar, require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(1.0f / scalar);
    // val / s == val * (1. / s)
    newVar.op_ = VariableOp::MUL_MS;
    return newVar;
}

Variable operator+(float scalar, const Variable& var) {
    Variable newVar(var.value + scalar, var.require_grad_);
    newVar.parent.push_back(&var);
    newVar.parent.push_back(scalar);
    newVar.op_ = VariableOp::ADD_SM;
    return newVar;
}
Variable operator*(float scalar, const Variable& var) {
    Variable newVar(var.value * scalar, var.require_grad_);
    newVar.parent.push_back(&var);
    newVar.parent.push_back(scalar);
    newVar.op_ = VariableOp::MUL_SM;
    return newVar;
}