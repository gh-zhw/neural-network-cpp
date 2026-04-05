#include <stdexcept>
#include <cmath>

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

// variable math operations
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
    newVar.op_ = VariableOp::MATMUL;
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
    newVar.op_ = VariableOp::ELEMENT_MUL_MS;
    return newVar;
}
Variable Variable::operator/(float scalar) const {
    Variable newVar(value / scalar, require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(1.0f / scalar);
    // val / s == val * (1. / s)
    newVar.op_ = VariableOp::ELEMENT_MUL_MS;
    return newVar;
}
Variable Variable::elementMul(const Variable& var) const
{
    Variable newVar(value.elementMul(value), require_grad_ || var.require_grad_);
    newVar.parent.push_back(this);
    newVar.parent.push_back(&var);
    newVar.op_ = VariableOp::ELEMENT_MUL_MM;
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
    newVar.op_ = VariableOp::ELEMENT_MUL_SM;
    return newVar;
}

// backward
void Variable::backward()
{
    grad = Matrix(value.row(), value.col(), 1.0f);
    _backward();
}

void Variable::_backward()
{
    if (!require_grad_ || op_ == VariableOp::NONE) { return; }

    switch (op_) {
        case VariableOp::ADD_MM: {
            // C = A + B
            const Variable* A = std::get<const Variable*>(parent[0]);
            const Variable* B = std::get<const Variable*>(parent[1]);
            if (A->require_grad_) {
                A->grad += grad;
            }
            if (B->require_grad_) {
                B->grad += grad;
            }
        } break;

        case VariableOp::ADD_MS: {
            // B = A + s
            const Variable* A = std::get<const Variable*>(parent[0]);
            if (A->require_grad_) {
                A->grad += grad;
            }
        } break;

        case VariableOp::ADD_SM: {
            // B = s + A
            const Variable* A = std::get<const Variable*>(parent[1]);
            if (A->require_grad_) {
                A->grad += grad;
            }
        } break;

        case VariableOp::SUB_MM: {
            // C = A - B
            const Variable* A = std::get<const Variable*>(parent[0]);
            const Variable* B = std::get<const Variable*>(parent[1]);
            if (A->require_grad_) {
                A->grad += grad;
            }
            if (B->require_grad_) {
                B->grad -= grad;
            }
        } break;

        case VariableOp::SUB_MS: {
            // B = A - s
            const Variable* A = std::get<const Variable*>(parent[0]);
            if (A->require_grad_) {
                A->grad += grad;
            }
        } break;

        case VariableOp::MATMUL: {
            // C = A * B
            const Variable* A = std::get<const Variable*>(parent[0]);
            const Variable* B = std::get<const Variable*>(parent[1]);
            if (A->require_grad_) {
                // dA = grad * B^T
                Matrix Bt = B->value.T();
                Matrix dA = grad.matMul(Bt);
                A->grad += dA;
            }
            if (B->require_grad_) {
                // dB = A^T * grad
                Matrix At = A->value.T();
                Matrix dB = At.matMul(grad);
                B->grad += dB;
            }
        } break;

        case VariableOp::ELEMENT_MUL_MM: {
            // C = A ⊙ B
            const Variable* A = std::get<const Variable*>(parent[0]);
            const Variable* B = std::get<const Variable*>(parent[1]);
            if (A->require_grad_) {
                // dA = grad ⊙ B
                A->grad += grad.elementMul(B->value);
            }
            if (B->require_grad_) {
                // dB = grad ⊙ A
                B->grad += grad.elementMul(A->value);
            }
        } break;

        case VariableOp::ELEMENT_MUL_MS: {
            // B = A * s
            const Variable* A = std::get<const Variable*>(parent[0]);
            float s = std::get<float>(parent[1]);
            if (A->require_grad_) {
                // dA = grad * s
                Matrix dA = grad.elementMul(s);
                A->grad += dA;
            }
        } break;

        case VariableOp::ELEMENT_MUL_SM: {
            // B = s * A
            float s = std::get<float>(parent[0]);
            const Variable* A = std::get<const Variable*>(parent[1]);
            if (A->require_grad_) {
                Matrix dA = grad.elementMul(s);
                A->grad += dA;
            }
        } break;

        case VariableOp::RELU: {
            // y = max(0, x)
            const Variable* A = std::get<const Variable*>(parent[0]);
            if (A->require_grad_) {
                size_t h = A->h(), w = A->w();
                Matrix dA(h, w, 0.0f);
                for (size_t i = 0; i < h; ++i) {
                    for (size_t j = 0; j < w; ++j) {
                        // dy/dx = 1 if x>0 else 0
                        if (A->get(i, j) > 0) {
                            dA.set(i, j, grad.get(i, j));
                        }
                    }
                }
                A->grad += dA;
            }
        } break;

        case VariableOp::SIGMOID: {
            // y = sigmoid(x)
            const Variable* x = std::get<const Variable*>(parent[0]);
            if (x->require_grad_) {
                size_t h = x->h(), w = x->w();
                Matrix dA(h, w, 0.0f);
                for (size_t i = 0; i < h; ++i) {
                    for (size_t j = 0; j < w; ++j) {
                        float y = this->value.get(i, j);
                        // dy/dx = y * (1 - y)
                        dA.set(i, j, grad.get(i, j) * y * (1 - y));
                    }
                }
                x->grad += dA;
            }
        } break;

        case VariableOp::CROSS_ENTROPY_LOSS: {
            // Loss = -1/N * sum_i (y_i * log(p_i)), where p = softmax(logits)
            // dL/dlogits = (p - y) / N
            const Variable* logits = std::get<const Variable*>(parent[0]);
            const Variable* labels = std::get<const Variable*>(parent[1]);
            if (logits->require_grad_) {
                size_t N = logits->h();          // batch size
                size_t C = logits->w();          // number of classes
                // Compute softmax probabilities p
                Matrix p(N, C, 0.0f);
                for (size_t i = 0; i < N; ++i) {
                    // Find max logit for numerical stability
                    float max_logit = logits->get(i, 0);
                    for (size_t j = 1; j < C; ++j) {
                        max_logit = std::max(max_logit, logits->get(i, j));
                    }
                    // Compute exp(logit - max_logit) and sum
                    float sum_exp = 0.0f;
                    for (size_t j = 0; j < C; ++j) {
                        float exp_val = std::exp(logits->get(i, j) - max_logit);
                        p.set(i, j, exp_val);
                        sum_exp += exp_val;
                    }
                    // Normalize to get probabilities
                    for (size_t j = 0; j < C; ++j) {
                        p.set(i, j, p.get(i, j) / sum_exp);
                    }
                }
                // Gradient scale: upstream grad (from loss node) / N
                float scale = grad.get(0, 0) / N;  // upstream gradient (scalar)
                // Compute gradient matrix dLogits = (p - y) * scale
                Matrix dLogits(N, C, 0.0f);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < C; ++j) {
                        float diff = p.get(i, j) - labels->get(i, j);
                        dLogits.set(i, j, diff * scale);
                    }
                }
                logits->grad += dLogits;
            }
        } break;

        case VariableOp::MSE_LOSS: {
            // Loss = 1/(N*D) * sum_i,j (pred_ij - target_ij)^2
            // dLoss/dpred = 2*(pred - target) / (N*D)
            // dLoss/dtarget = -2*(pred - target) / (N*D)
            const Variable* pred = std::get<const Variable*>(parent[0]);
            const Variable* target = std::get<const Variable*>(parent[1]);
            size_t N = pred->h();
            size_t D = pred->w();
            float scale = 2.0f / (N * D);
            float upstream = grad.get(0, 0);  // upstream gradient (scalar)
            if (pred->require_grad_) {
                Matrix dPred(N, D, 0.0f);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < D; ++j) {
                        float diff = pred->get(i, j) - target->get(i, j);
                        dPred.set(i, j, upstream * scale * diff);
                    }
                }
                pred->grad += dPred;
            }
            if (target->require_grad_) {
                Matrix dTarget(N, D, 0.0f);
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < D; ++j) {
                        float diff = pred->get(i, j) - target->get(i, j);
                        dTarget.set(i, j, -upstream * scale * diff);
                    }
                }
                target->grad += dTarget;
            }
        } break;

        default: {
            throw std::runtime_error("Unknown op in backward");
        } break;
    }
}
