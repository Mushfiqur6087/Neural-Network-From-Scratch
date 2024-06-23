#pragma once
#include <bits/stdc++.h>
#include "Layer_Dense.cpp"
using namespace std;
class Optimizer_SGD
{
public:
    double learning_rate;
    double current_learning_rate;
    double decay;
    double iterations;
    double momentum;
    Optimizer_SGD(double learning_rate = 1.0, double decay = 0.0, double momentum = 0.0)
    {
        this->current_learning_rate = learning_rate;
        this->learning_rate = learning_rate;
        this->decay = decay;
        iterations = 0;
        this->momentum = momentum;
    }
    // Method to update the parameters of a layer
    void update_params(Layer_Dense &layer)
    {
        if (momentum)
        {
            for (size_t i = 0; i < layer.weights.size(); i++)
            {
                for (size_t j = 0; j < layer.weights[0].size(); j++)
                {
                    layer.weights[i][j] += momentum * layer.weight_momentums[i][j] - current_learning_rate * layer.dweights[i][j];
                    layer.weight_momentums[i][j] = momentum * layer.weight_momentums[i][j] - current_learning_rate * layer.dweights[i][j];
                }
            }
            for (size_t i = 0; i < layer.biases.size(); i++)
            {
                layer.biases[i] += momentum * layer.bias_momentums[i] - current_learning_rate * layer.dbiases[i];
                layer.bias_momentums[i] = momentum * layer.bias_momentums[i] - current_learning_rate * layer.dbiases[i];
            }
        }
        else
        {
            for (size_t i = 0; i < layer.weights.size(); ++i)
            {
                for (size_t j = 0; j < layer.weights[0].size(); ++j)
                {
                    layer.weights[i][j] += -current_learning_rate * layer.dweights[i][j];
                }
            }
            for (size_t i = 0; i < layer.biases.size(); ++i)
            {
                layer.biases[i] += -current_learning_rate * layer.dbiases[i];
            }
        }
    }

    void pre_update_params()
    {
        if (decay)
        {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }
    void post_update_params()
    {
        iterations++;
    }
};