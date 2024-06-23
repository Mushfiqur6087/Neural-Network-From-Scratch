#pragma once
#include <iostream>
#include <vector>
#include <algorithm> // For  max
using namespace std;

class Activation_ReLU
{
public:
    vector<vector<double>> input;
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    vector<vector<double>> forward(const vector<vector<double>> &inputs)
    {
        this->input = inputs;
        int n_samples = inputs.size();
        int n_features = inputs[0].size();
        output.resize(n_samples, vector<double>(n_features, 0.0));
        for(int i=0;i<output.size();i++){
            for(int j=0;j<output[i].size();j++){
                output[i][j]=0;
            }
        }
        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < n_features; ++j)
            {
                output[i][j] = max(0.0, inputs[i][j]);
            }
        }
        return output;
    }

    vector<vector<double>> backward(const vector<vector<double>> &dvalues)
    {
        int n_samples = dvalues.size();
        int n_features = dvalues[0].size();
        vector<vector<double>> dinputs(n_samples, vector<double>(n_features, 0.0));

        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < n_features; ++j)
            {
                dinputs[i][j] = dvalues[i][j] * (input[i][j] > 0);
            }
        }
        this->dinputs = dinputs;
        return dinputs;
    }
};
