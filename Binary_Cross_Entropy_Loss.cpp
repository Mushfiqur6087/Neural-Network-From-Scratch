#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer_Dense.cpp"
using namespace std;
//!* output neuron sudhu 1 ta hobe n*1 matrix payoa jabe..setake flat kore then loss calculate korbo
class Binary_Cross_Entropy_Loss
{
public:
    vector<double> dinputs;
    //calculate regularized loss
    double regularizationLoss(Layer_Dense layer)
    {
        double regularizationLoss=0;
        if(layer.weight_regularizer_l1>0){
            for(int i=0;i<layer.weights.size();i++){
                for(int j=0;j<layer.weights[0].size();j++){
                    regularizationLoss+=layer.weight_regularizer_l1*abs(layer.weights[i][j]);
                }
            }
        }
        if(layer.bias_regularizer_l1>0){
            for(int i=0;i<layer.biases.size();i++){
                regularizationLoss+=layer.bias_regularizer_l1*abs(layer.biases[i]);
            }
        }

        if(layer.weight_regularizer_l2>0){
            for(int i=0;i<layer.weights.size();i++){
                for(int j=0;j<layer.weights[0].size();j++){
                    regularizationLoss+=layer.weight_regularizer_l2*layer.weights[i][j]*layer.weights[i][j];
                }
            }
        }

        if(layer.bias_regularizer_l2>0){
            for(int i=0;i<layer.biases.size();i++){
                regularizationLoss+=layer.bias_regularizer_l2*layer.biases[i]*layer.biases[i];
            }
        }
        return regularizationLoss;

    }
    // Calculates the data loss
    double calculateLoss(const vector<double> &output, const vector<int> &y)
    {
        vector<double> sample_losses;
        sample_losses = forward(output, y);
        double data_loss = mean(sample_losses);
        return data_loss;
    }
    // Forward pass for categorical labels
    vector<double> forward(const vector<double> &y_pred, const vector<int> &y_true)
    {
        int samples = y_pred.size();
        vector<double> y_pred_clipped = clip(y_pred, 1e-6, 1.0 - 1e-6);
        vector<double> losses(samples);
        for (int i = 0; i < samples; ++i)
        {
            losses[i] = -(y_true[i]*log(y_pred_clipped[i]))+(1-y_true[i])*log(1-y_pred_clipped[i]);
        }
        return losses;

    }
    // Backward pass
    vector<double> backward(const vector<double> &dvalues,const vector<double> y_true)
    {
        double samples = dvalues.size();
        dinputs.resize(samples);
        vector<double> clipped_value= clip(dvalues, 1e-6, 1.0 - 1e-6);
        for (int i = 0; i < samples; ++i)
        {
            dinputs[i] = -(y_true[i] / clipped_value[i] - (1 - y_true[i]) / (1 - clipped_value[i]));
        }
        return dinputs;
    }

private:
    // Helper function to clip values
    vector<double> clip(const vector<double> &values, double min, double max)
    {
        vector<double> clipped_values = values;
        for (auto &row : clipped_values)
        {
          row=clamp(row, min, max);
        }
        return clipped_values;
    }

    // Helper function to calculate mean of a vector
    double mean(const vector<double> &v)
    {
        double sum = 0.0;
        for (double val : v)
        {
            sum += val;
        }
        return sum / v.size();
    }
    double clamp(double val, double min, double max)
    {
        if (val < min)
            return min;
        if (val > max)
            return max;
        return val;
    }
};
