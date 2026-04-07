
#include <cmath>
#include <stdexcept>
#include <numeric>

#include "../include/variable.hpp"
#include "../include/functions.hpp"


// activation functions
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

Variable sigmoid(const Variable& var)
{
    Variable newVar(var.value, var.require_grad_);

    // sigmoid(x) = 1 / (1 + exp(-x))
    for (size_t i = 0; i < newVar.h(); ++i) {
        for (size_t j = 0; j < newVar.w(); ++j) {
            float x = newVar.get(i, j);
            float sig = 1.0f / (1.0f + std::exp(-x));
            newVar.set(i, j, sig);
        }
    }

    newVar.parent.push_back(&var);
    newVar.op_ = VariableOp::SIGMOID;

    return newVar;
}


// loss functions
Variable cross_entropy_loss(const Variable& logits, const Variable& labels)
{
    if (labels.h() != logits.h() || labels.w() != logits.w()) {
        throw std::invalid_argument("cross_entropy_loss: dimensions mismatch");;
    }

    size_t batch_size = logits.h();
    size_t num_elements = logits.w();

    std::vector<float> losses(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        float sum_exp_logit = 0.0f;
        float max_logit = logits.get(i, 0);
        float target_logit = 0.0f;

        for (size_t j = 0; j < num_elements; ++j) {
            if (logits.get(i, j) > max_logit) {
                max_logit = logits.get(i, j);
            }
        }

        for (size_t j = 0; j < num_elements; ++j) {
            float logit = logits.get(i, j);
            if (labels.get(i, j) == 1.0f) {
                target_logit = logit;
            }
            sum_exp_logit += std::exp(logit - max_logit);
        }

        float loss = std::log(sum_exp_logit) + max_logit - target_logit;
        losses[i] = loss;
    }

    float total_loss = std::accumulate(losses.begin(), losses.end(), 0.0f);
    float avg_loss = total_loss / batch_size;
    Variable loss(avg_loss, logits.require_grad_ || labels.require_grad_);
    loss.parent.push_back(&logits);
    loss.parent.push_back(&labels);
    loss.op_ = VariableOp::CROSS_ENTROPY_LOSS;
    loss.name_ = "cross_entropy_loss";

    return loss;
}

Variable mse_loss(const Variable& predictions, const Variable& targets) {
    if (predictions.h() != targets.h() || predictions.w() != targets.w()) {
        throw std::invalid_argument("mse_loss: dimensions mismatch");
    }

    size_t batch_size = predictions.h();
    size_t num_elements = predictions.w();

    std::vector<float> losses(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < num_elements; ++j) {
            float diff = predictions.get(i, j) - targets.get(i, j);
            sum_sq += diff * diff;
        }
        losses[i] = sum_sq / num_elements;
    }

    float total_loss = std::accumulate(losses.begin(), losses.end(), 0.0f);
    float avg_loss = total_loss / batch_size;

    Variable loss(avg_loss, predictions.require_grad_ || targets.require_grad_);
    loss.parent.push_back(&predictions);
    loss.parent.push_back(&targets);
    loss.op_ = VariableOp::MSE_LOSS;
    loss.name_ = "mse_loss";

    return loss;
}
