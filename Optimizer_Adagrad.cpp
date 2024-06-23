#pragma once
#include <bits/stdc++.h>
#include "Layer_Dense.cpp"
using namespace std;
class Optimizer_Adagrad
{
public:
    double learning_rate;
    double current_learning_rate;
    double decay;
    double iterations;
    double epsilon;
    Optimizer_Adagrad(double learning_rate = 1.0, double decay = 0.0, double epsilon = 1e-7)
    {
        this->current_learning_rate = learning_rate;
        this->learning_rate = learning_rate;
        this->decay = decay;
        iterations = 0;
        this->epsilon = epsilon;
    }
    //! Method to update the parameters of a layer
    void update_params(Layer_Dense &layer)
    {
        //! Update cache with squared current gradients
        for (size_t i = 0; i < layer.weights.size(); i++)
        {
            for (size_t j = 0; j < layer.weights[0].size(); j++)
            {
                layer.weight_cache[i][j] += layer.dweights[i][j] * layer.dweights[i][j];
            }
        }
        for (size_t i = 0; i < layer.biases.size(); i++)
        {
            layer.bias_cache[i] += layer.dbiases[i] * layer.dbiases[i];
        }
        //! Vanilla SGD parameter update + normalization
        //! with square rooted cache
        for (size_t i = 0; i < layer.weights.size(); ++i)
        {
            for (size_t j = 0; j < layer.weights[0].size(); ++j)
            {
                layer.weights[i][j] += -current_learning_rate * layer.dweights[i][j] / (sqrt(layer.weight_cache[i][j]) + epsilon);
            }
        }
        for (size_t i = 0; i < layer.biases.size(); ++i)
        {
            layer.biases[i] += -current_learning_rate * layer.dbiases[i] / (sqrt(layer.bias_cache[i]) + epsilon);
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