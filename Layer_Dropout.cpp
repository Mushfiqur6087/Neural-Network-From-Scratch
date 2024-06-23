#pragma once
#include <vector>
#include <random>
#include <iostream>
using namespace std;
vector<vector<int>> generate_binomial_distribution_2D(int n, double p, size_t rows, size_t cols) {
    // Create a random number generator and a binomial distribution
    random_device rd; // Seed for the random number generator
    mt19937 gen(rd()); // Mersenne Twister generator
    binomial_distribution<> d(n, p);
    // Vector to store the generated random numbers
    vector<vector<int>> result(rows, vector<int>(cols));
    // Generate the random numbers
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            result[i][j] = d(gen);
        }
    }

    return result;
}


class Layer_Dropout
{
public:
    vector<vector<int>> mask;
    vector<vector<double>> output;
    vector<vector<double>> dinputs;
    double rate;
    Layer_Dropout(double dropout_rate)
    {
        this->rate = 1.0 - dropout_rate;
    }
    vector<vector<double>> forward(const vector<vector<double>> &inputs)
    {
        mask = generate_binomial_distribution_2D(1, rate, inputs.size(), inputs[0].size());
        output.resize(inputs.size(), vector<double>(inputs[0].size()));
        for(size_t i = 0; i < inputs.size(); ++i) {
            for(size_t j = 0; j < inputs[0].size(); ++j) {
                output[i][j] = inputs[i][j] * mask[i][j]/rate;
            }
        }
        return output;
    }

    vector<vector<double>> backward(const vector<vector<double>> &dvalues)
    {
        dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size()));
        for(size_t i = 0; i < dvalues.size(); ++i) {
            for(size_t j = 0; j < dvalues[0].size(); ++j) {
                dinputs[i][j] = dvalues[i][j] * mask[i][j];
            }
        }
        return dinputs;
    }
};


