#pragma once

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
    virtual void update(Variable&) = 0;
};


class SGD : public Optimizer
{
public:
    SGD(float lr = 0.1f) : learning_rate_(lr) {}

    void update(Variable& var) override;

private:
    float learning_rate_;
};


class MomentumSGD : public Optimizer {
public:
    MomentumSGD(float lr = 0.1f, float momentum = 0.9f) 
        : learning_rate_(lr), momentum_(momentum) {}

    void update(Variable& var) override;

private:
    float learning_rate_;
    float momentum_;
    std::unordered_map<Variable*, Matrix> velocities_;

    Matrix& get_velocity(Variable&);
};
