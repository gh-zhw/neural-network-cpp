#include "../include/optimizer.hpp"


void SGD::update(Variable& var)
{
    if (!var.require_grad_) { return; }

    for (size_t i = 0; i < var.h(); ++i) {
        for (size_t j = 0; j < var.w(); ++j) {
            var.set(i, j, 
                var.get(i, j) - learning_rate_ * var.grad.get(i, j));
        }
    }
}

void MomentumSGD::update(Variable& var)
{
    if (!var.require_grad_) return;
    Matrix& velocity = get_velocity(var);
    for (size_t i = 0; i < var.h(); ++i) {
        for (size_t j = 0; j < var.w(); ++j) {
            // v = momentum * v - lr * grad
            float new_v = momentum_ * velocity.get(i, j) - learning_rate_ * var.grad.get(i, j);
            velocity.set(i, j, new_v);
            // param += v
            var.set(i, j, var.get(i, j) + new_v);
        }
    }
}

Matrix& MomentumSGD::get_velocity(Variable& var) {
    auto it = velocities_.find(&var);
    if (it == velocities_.end()) {
        Matrix vel(var.h(), var.w(), 0.0f);
        auto result = velocities_.emplace(&var, std::move(vel));
        return result.first->second;
    }
    return it->second;
}
