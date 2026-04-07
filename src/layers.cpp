#include <cstdlib>
#include <cmath>
#include <stdexcept>

#include "../include/layers.hpp"


Linear::~Linear()
{
    if (cache_.size()) {
        for (Variable* vp : cache_) {
            delete vp;
        }
        cache_.clear();
    }
}

void Linear::init_params() {
    std::string W_name = "Linear.W";
    if (!name_.empty()) { W_name = name_ + ".W"; }
    W = Variable(inDim_, outDim_, true, W_name);
    // Xavier
    float scale = std::sqrt(1.0f / inDim_);
    for (size_t i = 0; i < inDim_; ++i) {
        for (size_t j = 0; j < outDim_; ++j) {
            W.set(i, j, 
                (2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f) * scale);
        }
    }

    if (bias_) {
        std::string b_name = "Linear.b";
        if (!name_.empty()) { b_name = name_ + ".b"; }
        b = Variable(1, outDim_, true, b_name);
        for (size_t j = 0; j < outDim_; ++j) {
            b.set(0, j, 0.0f);
        }
    } else {
        b = Variable(0, 0, false);
    }
}

std::vector<Variable*> Linear::paramters()
{
    if (bias_) return {&W, &b};
    return {&W};
}

Variable Linear::forward(const Variable& x)
{
    // x: (batch_size, inDim_)
    // W: (inDim_, outDim_)
    // y = x * W + b

    if (x.w() != inDim_) {
        throw std::invalid_argument("Linear::forward: input dimension mismatch");
    }

    Variable* z = new Variable(x * W);
    z->setName(name_ + ".z");
    if (bias_) {
        size_t batch_size = x.h();
        Variable* ones = new Variable(batch_size, 1, false);
        for (size_t i = 0; i < batch_size; ++i) {
            ones->set(i, 0, 1.0f);
        }
        Variable* b_expanded = new Variable((*ones) * b);  // (batch_size, outDim_)

        ones->setName(name_ + ".ones");
        b_expanded->setName(name_ + ".b_expanded");

        cache_.push_back(z);
        cache_.push_back(ones);
        cache_.push_back(b_expanded);

        return (*z) + (*b_expanded);
    }
    return *z;
}

void Linear::clear_cache()
{
    if (cache_.size()) {
        for (Variable* vp : cache_) {
            delete vp;
        }
        cache_.clear();
    }
}

