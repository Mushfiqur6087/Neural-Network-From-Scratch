#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
class Activation_Sigmoid {
public:
     vector< vector<double>> output;
     vector< vector<double>> dinputs;

    // Forward pass
    void forward(const  vector< vector<double>>& inputs) 
    {
        this->output.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                this->output[i][j] = 1.0 / (1.0 +  exp(-inputs[i][j]));
            }
        }
    }

    // Backward pass
    void backward(const  vector< vector<double>>& dvalues) {
        this->dinputs.resize(dvalues.size());
        for (size_t i = 0; i < dvalues.size(); ++i) {
            for (size_t j = 0; j < dvalues[i].size(); ++j) {
                this->dinputs[i][j] = dvalues[i][j] * (1.0 - this->output[i][j]) * this->output[i][j];
            }
        }
    }
};