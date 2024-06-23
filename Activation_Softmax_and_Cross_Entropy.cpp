#pragma once
#include <iostream>
#include "Activation_Softmax.cpp"
#include "Cross_Entropy_Loss.cpp"
class Activation_Softmax_Loss_CategoricalCross_Entropy
{
    public:
    Activation_Softmax activation;
    Cross_Entropy_Loss loss;
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    Activation_Softmax_Loss_CategoricalCross_Entropy()
    {
        activation = Activation_Softmax();
        loss = Cross_Entropy_Loss();
    }
    double regularizationLoss(Layer_Dense layer)
    {
        return loss.regularizationLoss(layer);
    }

    double forward(const vector<vector<double>> &inputs, const vector<int> &y_true)
    {
        this->output = activation.forward(inputs);
        double mean_loss = loss.calculateLossCategorical(output, y_true);
        return mean_loss;
    }
    vector<vector<double>> backward(const std::vector<std::vector<double>> &dvalues, const std::vector<int> &y_true)
    {
        int samples = dvalues.size();
        int labels = dvalues[0].size();
        dinputs = dvalues;
           //!!! Calculate gradient
        for (int i = 0; i < samples; ++i)
        {
            dinputs[i][y_true[i]] -= 1;
        }
           //!!! Normalize gradient
        for (int i = 0; i < samples; ++i)
        {
            for (int j = 0; j < labels; ++j)
            {
                dinputs[i][j] /= samples;
            }
        }
        return dinputs;
    }
};