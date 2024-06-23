#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer_Dense.cpp"
using namespace std;

class Cross_Entropy_Loss
{
public:
    vector<vector<double>> dinputs;
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
    double calculateLossCategorical(const vector<vector<double>> &output, const vector<int> &y)
    {
        vector<double> sample_losses;
        sample_losses = forwardCatergorical(output, y);
        double data_loss = mean(sample_losses);
        return data_loss;
    }

    double calculateLossOneHotEncoded(const vector<vector<double>> &output, const vector<vector<int>> &y)
    {
        vector<double> sample_losses;
        sample_losses = forwardOneHotEncoded(output, y);
        double data_loss = mean(sample_losses);
        return data_loss;
    }

    // Forward pass for categorical labels
    vector<double> forwardCatergorical(const vector<vector<double>> &y_pred, const vector<int> &y_true)
    {
        int samples = y_pred.size();
        vector<vector<double>> y_pred_clipped = clip(y_pred, 1e-6, 1.0 - 1e-6);
        vector<double> correct_confidences(samples);

        for (int i = 0; i < samples; ++i)
        {
            correct_confidences[i] = y_pred_clipped[i][y_true[i]];
        }

        vector<double> negative_log_likelihoods(samples);
        for (int i = 0; i < samples; ++i)
        {
            negative_log_likelihoods[i] = -log(correct_confidences[i]);
        }

        return negative_log_likelihoods;
    }

    // Forward pass for one-hot encoded labels
    vector<double> forwardOneHotEncoded(const vector<vector<double>> &y_pred, const vector<vector<int>> &y_true)
    {
        int samples = y_pred.size();
        vector<vector<double>> y_pred_clipped = clip(y_pred, 1e-7, 1.0 - 1e-7);
        vector<double> correct_confidences(samples);

        for (int i = 0; i < samples; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < y_pred[i].size(); ++j)
            {
                sum += y_pred_clipped[i][j] * y_true[i][j];
            }
            correct_confidences[i] = sum;
        }

        vector<double> negative_log_likelihoods(samples);
        for (int i = 0; i < samples; ++i)
        {
            negative_log_likelihoods[i] = -log(correct_confidences[i]);
        }
        return negative_log_likelihoods;
    }

    vector<vector<double>> backwardCategorical(const vector<vector<double>> &dvalues, const vector<int> &y_true)
    {
        int samples = dvalues.size();
        int labels = dvalues[0].size();

        // Convert y_true to one-hot encoding
        vector<vector<double>> y_true_one_hot(samples, vector<double>(labels, 0.0));
        for (int i = 0; i < samples; ++i)
        {
            y_true_one_hot[i][y_true[i]] = 1.0;
        }
        // Calculate gradient
        dinputs.resize(samples, vector<double>(labels, 0.0));
        for (int i = 0; i < samples; ++i)
        {
            for (int j = 0; j < labels; ++j)
            {
                dinputs[i][j] = -y_true_one_hot[i][j] / dvalues[i][j];
            }
        }
        // Normalize gradient
        for (int i = 0; i < samples; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < labels; ++j)
            {
                dinputs[i][j] /= samples;
            }
        }
        return dinputs;
    }

    vector<vector<double>> backwardOneHotEncoded(const vector<vector<double>> &dvalues, const vector<vector<int>> &y_true)
    {
        // cout<<"ami eikhane"<<endl;
        int samples = dvalues.size();
        int labels = dvalues[0].size();
        // no need to convert to one-hot encoding
        // Calculate gradient
        dinputs.resize(samples, vector<double>(labels, 0.0));
        // cout<<"samples: "<<samples<<" labels: "<<labels<<endl;
        for (int i = 0; i < samples; ++i)
        {
            for (int j = 0; j < labels; ++j)
            {
                dinputs[i][j] = -y_true[i][j] / dvalues[i][j];
            }
        }
        // Normalize gradient
        for (int i = 0; i < samples; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < labels; ++j)
            {
                dinputs[i][j] /= samples;
            }
        }

        return dinputs;
    }

private:
    // Helper function to clip values
    vector<vector<double>> clip(const vector<vector<double>> &values, double min, double max)
    {
        vector<vector<double>> clipped_values = values;
        for (auto &row : clipped_values)
        {
            for (auto &val : row)
            {
                val = clamp(val, min, max);
            }
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
