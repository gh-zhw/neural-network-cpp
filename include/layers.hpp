#pragma once

#include "variable.hpp"


class Layer
{
public:
    Layer() = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer() = default;
    virtual Variable forward(const Variable&) = 0;
};


class Linear : public Layer
{
public:
    Linear(size_t inDim, size_t outDim, bool bias = true, 
           const std::string& name = "")
        : name_(name), inDim_(inDim), outDim_(outDim), bias_(bias)
        {
            init_params();
        }

    ~Linear() override;

    void init_params();
    std::vector<Variable*> paramters();
    Variable forward(const Variable&) override;
    void clear_cache();

private:
    std::string name_;
    size_t inDim_, outDim_;
    bool bias_;
    Variable W, b;
    std::vector<Variable*> cache_;
};

