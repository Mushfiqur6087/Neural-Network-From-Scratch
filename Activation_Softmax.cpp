#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
class Activation_Softmax
{
public:
    vector<vector<double>> output;
    vector<vector<double>> dinput;
    vector<vector<double>> forward(const vector<vector<double>> &inputs)
    {
        int n_samples = inputs.size();
        int n_classes = inputs[0].size();
        output.resize(n_samples, vector<double>(n_classes, 0.0));
        for (int i = 0; i < output.size(); i++)
        {
            for (int j = 0; j < output[i].size(); j++)
            {
                output[i][j] = 0;
            }
        }

        for (int i = 0; i < n_samples; ++i)
        {
            // Find max element in each row for numerical stability
            double max_val = *max_element(inputs[i].begin(), inputs[i].end());
            // Compute softmax
            double exp_sum = 0.0;
            for (int j = 0; j < n_classes; ++j)
            {
                output[i][j] = exp(inputs[i][j] - max_val);
                exp_sum += output[i][j];
            }
            for (int j = 0; j < n_classes; ++j)
            {
                output[i][j] /= exp_sum;
            }
        }
        return output;
    }

    // Backward pass
    vector<vector<double>> backward(const vector<vector<double>> &dvalues)
    {
        dinput.resize(output.size(), vector<double>(output[0].size(), 0.0));
        for (int i = 0; i < dinput.size(); i++)
        {
            for (int j = 0; j < dinput[i].size(); j++)
            {
                dinput[i][j] = 0;
            }
        }

        for (size_t i = 0; i < output.size(); ++i)
        {
            // Flatten output array
            vector<vector<double>> single_output(1, output[i]);
            // Calculate Jacobian matrix of the output
            vector<vector<double>> jacobian_matrix(output[i].size(), vector<double>(output[i].size()));
            for (size_t j = 0; j < output[i].size(); ++j)
            {
                for (size_t k = 0; k < output[i].size(); ++k)
                {
                    if (j == k)
                    {
                        jacobian_matrix[j][k] = output[i][j] * (1 - output[i][k]);
                    }
                    else
                    {
                        jacobian_matrix[j][k] = -output[i][j] * output[i][k];
                    }
                }
            }

            for (size_t m = 0; m < jacobian_matrix.size(); ++m)
            {
                double x = 0;
                for (size_t n = 0; n < dvalues[i].size(); ++n)
                {
                    //cout << "m: " << m << " n: " << n << " jacobian_matrix[m][n]: "<< jacobian_matrix[m][n] << " dvalues[i][n]: " << dvalues[i][n] << endl;
                    x += jacobian_matrix[m][n] * dvalues[i][n];
                    //cout << "x: " << x << endl;
                }
                dinput[i][m] = x;
                //cout << "dinput[i][m]: " << dinput[i][m] << endl;
                
            }
        }

        this->dinput = dinput;
        return dinput;
    }
 };