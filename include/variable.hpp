#pragma once

#include <string>
#include <variant>

#include "matrix.hpp"
#include "operations.hpp"


class Variable
{
public:
    Variable(float val, bool require_grad=true)
        : value(1, 1, val), require_grad_(require_grad) {}
    Variable(const Matrix& matrix, bool require_grad=true)
        : value(matrix), require_grad_(require_grad) {}
    Variable(size_t h, size_t w, bool require_grad=true)
        : value(h, w), require_grad_(require_grad) {}
    Variable(const Variable&) = default;
    Variable& operator=(const Variable&) = default;
    Variable(Variable&& var) noexcept
        :value(std::move(var.value)),
         grad(std::move(var.grad)),
         require_grad_(var.require_grad_),
         parent(std::move(var.parent)),
         op_(var.op_) {}
    Variable& operator=(Variable&& var) noexcept {
        if (this != &var) {
            value = std::move(var.value);
            grad = std::move(var.grad);
            require_grad_= var.require_grad_;
            parent = std::move(var.parent);
            op_ = var.op_;
        }
        return *this;
    }

    float get(size_t i, size_t j) const { return value.get(i, j); }
    void set(size_t, size_t, float);
    void assign(float);
    size_t h() const { return value.row(); }
    size_t w() const { return value.col(); }

    // variable math operations
    Variable operator+(const Variable&) const;
    Variable operator-(const Variable&) const;
    Variable operator*(const Variable&) const;

    Variable operator+(float) const;
    Variable operator-(float) const;
    Variable operator*(float) const;
    Variable operator/(float) const;

    friend Variable operator+(float, const Variable&);
    friend Variable operator*(float, const Variable&);

    // activation functions
    friend Variable relu(const Variable&);
    friend Variable sigmoid(const Variable&);

    // loss functions
    friend Variable cross_entropy_loss(const Variable&, const Variable&);
    friend Variable mse_loss(const Variable&, const Variable&);
    

private:
    bool require_grad_;
    Matrix value;
    Matrix grad;
    std::vector<std::variant<const Variable*, float>> parent;
    VariableOp op_ = VariableOp::NONE;
};
