#pragma once

#include "variable.hpp"

// activation functions
Variable relu(const Variable&);
Variable sigmoid(const Variable&);


// loss functions
Variable cross_entropy_loss(const Variable&, const Variable&);
Variable mse_loss(const Variable&, const Variable&);
