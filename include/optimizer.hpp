#pragma once

#include <vector>
#include <unordered_map>

#include <variable.hpp>

// virtual base class
class Optimizer
{
public:
    Optimizer() = default;
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    virtual ~Optimizer() = default;
    virtual void update() = 0;
    virtual void zero_grad() = 0;
};


class SGD : public Optimizer
{
public:
    SGD(const std::vector<Variable*>& model_params, float lr = 0.1f) 
        : params(model_params), learning_rate_(lr) {}

    void update() override;
    void zero_grad() override;

private:
    float learning_rate_;
    std::vector<Variable*> params;
};


class MomentumSGD : public Optimizer {
public:
    MomentumSGD(const std::vector<Variable*> model_params, 
                float lr = 0.1f, float momentum = 0.9f) 
                : params(model_params), learning_rate_(lr), momentum_(momentum) {
                    init_velocity();
                }

    void update() override;
    void zero_grad() override;

private:
    float learning_rate_;
    float momentum_;
    std::unordered_map<Variable*, Matrix> velocities_;
    std::vector<Variable*> params;

    void init_velocity();
};
