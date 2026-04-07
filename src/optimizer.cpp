#include "../include/optimizer.hpp"


void SGD::update()
{
    for (Variable* param : params) {
        if (!param->require_grad_) { continue; }

        for (size_t i = 0; i < param->h(); ++i) {
            for (size_t j = 0; j < param->w(); ++j) {
                param->set(i, j, 
                    param->get(i, j) - learning_rate_ * param->grad.get(i, j));
            }
        }
    }
}

void SGD::zero_grad()
{
    for (Variable* param : params) {
        if (!param->require_grad_) { continue; }
        param->zero_grad();
    }
}

void MomentumSGD::update()
{
    for (Variable* param : params) {
        if (!param->require_grad_) { continue; }
        Matrix& velocity = velocities_[param];
        for (size_t i = 0; i < param->h(); ++i) {
            for (size_t j = 0; j < param->w(); ++j) {
                // v = momentum * v - lr * grad
                float new_v = momentum_ * velocity.get(i, j) - learning_rate_ * param->grad.get(i, j);
                velocity.set(i, j, new_v);
                // param += v
                param->set(i, j, param->get(i, j) + new_v);
            }
        }
    }
    
}

void MomentumSGD::zero_grad()
{
    for (Variable* param : params) {
        if (!param->require_grad_) { continue; }
        param->zero_grad();
    }
}

void MomentumSGD::init_velocity() {
    for (Variable* param : params) {
        Matrix vel(param->h(), param->w(), 0.0f);
        velocities_.emplace(param, std::move(vel));
    }
}
