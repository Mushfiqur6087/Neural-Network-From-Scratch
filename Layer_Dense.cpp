#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace std;

double rand_normal()
{
    static bool initialized = false;
    if (!initialized)
    {
        srand(42);
        initialized = true;
    }
    static const double pi = 3.14159265358979323846;
    static const double epsilon = 1e-10;
    static const double two_pi = 2.0 * pi;

    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);

    double z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    return z0;
}

class Layer_Dense
{
public:
    vector<vector<double>> inputs;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> output;
    vector<vector<double>> dweights;
    vector<double> dbiases;
    vector<vector<double>> dinputs;
    vector<vector<double>> weight_momentums;
    vector<double> bias_momentums;
    vector<vector<double>> weight_cache;
    vector<double> bias_cache;
    double weight_regularizer_l1;
    double weight_regularizer_l2;
    double bias_regularizer_l1;
    double bias_regularizer_l2;
    vector<vector<double>> dweight_regularizer_l1;
    vector<double> dbias_regularizer_l1;

    // Constructor
    Layer_Dense(int n_inputs, int n_neurons, double weight_regularizer_l1 = 0,
                double weight_regularizer_l2 = 0, double bias_regularizer_l1 = 0, double bias_regularizer_l2 = 0)
    {
        // Initialize weights
        weights.resize(n_inputs, vector<double>(n_neurons));
        weight_momentums.resize(n_inputs, vector<double>(n_neurons));
        weight_cache.resize(n_inputs, vector<double>(n_neurons));
        dweight_regularizer_l1.resize(n_inputs, vector<double>(n_neurons));
        dbias_regularizer_l1.resize(n_neurons);
        this->weight_regularizer_l1 = weight_regularizer_l1;
        this->weight_regularizer_l2 = weight_regularizer_l2;
        this->bias_regularizer_l1 = bias_regularizer_l1;
        this->bias_regularizer_l2 = bias_regularizer_l2;
        for (int i = 0; i < n_inputs; ++i)
        {
            for (int j = 0; j < n_neurons; ++j)
            {
                weights[i][j] = 0.01 * rand_normal();
                weight_momentums[i][j] = 0;
                weight_cache[i][j] = 0;
                dweight_regularizer_l1[i][j] = 0;
            }
        }
        bias_momentums.resize(n_neurons, 0.0);
        biases.resize(n_neurons, 0.0);
        bias_cache.resize(n_neurons, 0.0);
        for (int i = 0; i < n_neurons; ++i)
        {
            biases[i] = 0;
            bias_momentums[i] = 0;
            bias_cache[i] = 0;
            dbias_regularizer_l1[i] = 0;
        }
    }

    vector<vector<double>> forward(const vector<vector<double>> &inputs)
    {
        int n_samples = inputs.size();
        int n_neurons = biases.size();
        this->inputs = inputs;
        output.resize(n_samples, vector<double>(n_neurons, 0.0));
        for (int i = 0; i < output.size(); i++)
        {
            for (int j = 0; j < output[i].size(); j++)
            {
                output[i][j] = 0;
            }
        }
        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < n_neurons; ++j)
            {
                for (int k = 0; k < inputs[i].size(); ++k)
                {
                    output[i][j] += inputs[i][k] * weights[k][j];
                }
                output[i][j] += biases[j];
            }
        }
        return output;
    }

    void backward(const std::vector<std::vector<double>> &dvalues)
    {
        dweights.resize(weights.size(), std::vector<double>(weights[0].size()));
        for (int i = 0; i < dweights.size(); i++)
        {
            for (int j = 0; j < dweights[i].size(); j++)
            {
                dweights[i][j] = 0;
            }
        }
        dbiases.resize(biases.size());
        for (int i = 0; i < dbiases.size(); i++)
        {
            dbiases[i] = 0;
        }
        dinputs.resize(inputs.size(), std::vector<double>(weights.size()));
        for (int i = 0; i < dinputs.size(); i++)
        {
            for (int j = 0; j < dinputs[i].size(); j++)
            {
                dinputs[i][j] = 0;
            }
        }

        // Gradients on parameters
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            for (size_t j = 0; j < weights[0].size(); ++j)
            {
                dbiases[j] += dvalues[i][j];
                for (size_t k = 0; k < weights.size(); ++k)
                {
                    dweights[k][j] += inputs[i][k] * dvalues[i][j];
                }
            }
        }
        //!* dweight calculation done here
        //!* Regularization er derivative niye kaj korte hobe
        /*  
            # Gradients on regularization
            # L1 on weights
            if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
            # L2 on weights
            if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2
            self.weights
        */
        if (weight_regularizer_l1 > 0)
        {
            for (size_t i = 0; i < weights.size(); ++i)
            {
                for (size_t j = 0; j < weights[0].size(); ++j)
                {
                    if (weights[i][j] >= 0)
                    {
                        dweight_regularizer_l1[i][j] = 1;
                    }
                    else
                    {
                        dweight_regularizer_l1[i][j] = -1;
                    }
                }
            }
            for (int i = 0; i < dweights.size(); i++)
            {
                for (int j = 0; j < dweights[0].size(); j++)
                {
                    dweights[i][j] += weight_regularizer_l1 * dweight_regularizer_l1[i][j];
                }
            }
        }

        if (bias_regularizer_l1 > 0)
        {
            for (int i = 0; i < biases.size(); i++)
            {
                if (biases[i] >= 0)
                {
                    dbias_regularizer_l1[i] = 1;
                }
                else
                {
                    dbias_regularizer_l1[i] = -1;
                }
            }
            for (int i = 0; i < dbiases.size(); i++)
            {
                dbiases[i] += bias_regularizer_l1 * dbias_regularizer_l1[i];
            }
        }

        if (weight_regularizer_l2 > 0)
        {
            for (int i = 0; i < weights.size(); i++)
            {
                for (int j = 0; j < weights[0].size(); j++)
                {
                    dweights[i][j] += 2 * weight_regularizer_l2 * weights[i][j];
                }
            }
        }
        if (bias_regularizer_l2 > 0)
        {
            for (int i = 0; i < biases.size(); i++)
            {
                dbiases[i] += 2 * bias_regularizer_l2 * biases[i];
            }
        }

        //! Gradient on values
        for (size_t i = 0; i < dvalues.size(); ++i)
        {
            for (size_t j = 0; j < weights.size(); ++j)
            {
                dinputs[i][j] = 0;
                for (size_t k = 0; k < weights[0].size(); ++k)
                {
                    dinputs[i][j] += dvalues[i][k] * weights[j][k];
                }
            }
        }
    }
};
