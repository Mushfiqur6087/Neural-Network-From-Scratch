#include <bits/stdc++.h>
#include "Activation_ReLU.cpp"
#include "Activation_Softmax_and_Cross_Entropy.cpp"
#include "Optimizer_Adam.cpp"
#include "Optimizer_Adagrad.cpp"
#include "Layer_Dropout.cpp"
using namespace std;

vector<string> split(const string &s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

int argmax(const vector<double> &vec)
{
    return distance(vec.begin(), max_element(vec.begin(), vec.end()));
}

// Function to find the argmax for each row in a 2D vector
vector<int> argmax(const vector<vector<double>> &mat)
{
    vector<int> result(mat.size());
    for (size_t i = 0; i < mat.size(); ++i)
    {
        result[i] = argmax(mat[i]);
    }
    return result;
}
// calculate mean of a vector
double mean(const std::vector<bool> &vec)
{
    return accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}
int main()
{
    ifstream file("spiral_data.csv"); // Open CSV file for reading

    if (!file.is_open())
    {
        cerr << "Error opening file." << endl;
        return 1;
    }

    vector<vector<double>> x_data;
    vector<int> y_data;
    string line;
    bool header = true;
    while (getline(file, line))
    {
        if (header)
        {
            header = false;
            continue; // Skip header row
        }
        vector<string> tokens = split(line, ',');
        // Convert tokens to double and store in x_data
        vector<double> row_x;
        for (size_t i = 0; i < tokens.size() - 1; ++i)
        {
            row_x.push_back(stod(tokens[i]));
        }
        x_data.push_back(row_x);

        // Convert last token to double and store in y_data
        int label = stod(tokens.back());
        y_data.push_back(label);
    }

    file.close();

    // Create Dense layer with 2 input features and 64 output values
    Layer_Dense layer1 = Layer_Dense(2, 512,0,0,0,0);
    Activation_ReLU activation1 = Activation_ReLU();
    Layer_Dense layer2 = Layer_Dense(512, 3);
    //Layer_Dropout dropout = Layer_Dropout(.1);
    Activation_Softmax_Loss_CategoricalCross_Entropy loss_activation = Activation_Softmax_Loss_CategoricalCross_Entropy();
    Optimizer_Adam Adam = Optimizer_Adam(.05,5e-5);
    for (int epoch = 0; epoch < 1000; epoch++)
    {
        layer1.forward(x_data);
        activation1.forward(layer1.output);
        //dropout.forward(activation1.output);
        //layer2.forward(dropout.output);
        layer2.forward(activation1.output);
        double loss = loss_activation.forward(layer2.output,y_data);
        double regularization_loss = loss_activation.regularizationLoss(layer1) + loss_activation.regularizationLoss(layer2);
        double total_loss = loss + regularization_loss;
        cout << "Loss: " << total_loss << endl;
        loss_activation.backward(loss_activation.output, y_data);
        layer2.backward(loss_activation.dinputs);
        //dropout.backward(layer2.dinputs);
        //activation1.backward(dropout.dinputs);
        activation1.backward(layer2.dinputs);
        layer1.backward(activation1.dinputs);
        Adam.pre_update_params();
        Adam.update_params(layer1);
        Adam.update_params(layer2);
        Adam.post_update_params();
    }
    
    return 0;
}
