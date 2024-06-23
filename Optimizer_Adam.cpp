#pragma once
#include <bits/stdc++.h>
#include "Layer_Dense.cpp"
using namespace std;
class Optimizer_Adam
{
public:
    double learning_rate;
    double current_learning_rate;
    double decay;
    double iterations;
    double epsilon;
    double beta_1;
    double beta_2;
    Optimizer_Adam(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7, double beta_1 = .9, double beta_2 = .999)
    {
        this->current_learning_rate = learning_rate;
        this->learning_rate = learning_rate;
        this->decay = decay;
        this->iterations = 0;
        this->epsilon = epsilon;
        this->beta_1 = beta_1;
        this->beta_2 = beta_2;
    }
    //! Method to update the parameters of a layer
    void update_params(Layer_Dense &layer)
    {
        vector<vector<double>> weight_momentum_corrected;
        vector<double> bias_momentum_corrected;
        vector<vector<double>> weight_cache_corrected;
        vector<double> bias_cache_corrected;
        weight_momentum_corrected.resize(layer.weights.size(), vector<double>(layer.weights[0].size()));
        bias_momentum_corrected.resize(layer.biases.size());
        weight_cache_corrected.resize(layer.weights.size(), vector<double>(layer.weights[0].size()));
        bias_cache_corrected.resize(layer.biases.size());
        //! Update cache with squared current gradients
        for (size_t i = 0; i < layer.weights.size(); i++)
        {
            for (size_t j = 0; j < layer.weights[0].size(); j++)
            {
                layer.weight_momentums[i][j] = beta_1 * layer.weight_momentums[i][j] + (1 - beta_1) * layer.dweights[i][j];
                // Get corrected momentum
                // iteration is 0 at first pass
                // and we need to start with 1 here
                weight_momentum_corrected[i][j] = layer.weight_momentums[i][j] / (1 - pow(beta_1, iterations + 1));
                layer.weight_cache[i][j] = beta_2 * layer.weight_cache[i][j] + (1 - beta_2) * layer.dweights[i][j] * layer.dweights[i][j];
                weight_cache_corrected[i][j] = layer.weight_cache[i][j] / (1 - pow(beta_2, iterations + 1));
                
            }
        }
        for (size_t i = 0; i < layer.biases.size(); i++)
        {
            layer.bias_momentums[i] = beta_1 * layer.bias_momentums[i] + (1 - beta_1) * layer.dbiases[i];
            bias_momentum_corrected[i] = layer.bias_momentums[i] / (1 - pow(beta_1, iterations + 1));
            layer.bias_cache[i] = beta_2 * layer.bias_cache[i] + (1 - beta_2) * layer.dbiases[i] * layer.dbiases[i];
            bias_cache_corrected[i] = layer.bias_cache[i] / (1 - pow(beta_2, iterations + 1));
        }
        //! Vanilla SGD parameter update + normalization
        //! with square rooted cache
        for (size_t i = 0; i < layer.weights.size(); ++i)
        {
            for (size_t j = 0; j < layer.weights[0].size(); ++j)
            {
                layer.weights[i][j] += -current_learning_rate * weight_momentum_corrected[i][j] / (sqrt(weight_cache_corrected[i][j]) + epsilon);
            }
        }
        for (size_t i = 0; i < layer.biases.size(); ++i)
        {
            layer.biases[i] += -current_learning_rate * bias_momentum_corrected[i] / (sqrt(bias_cache_corrected[i]) + epsilon);
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