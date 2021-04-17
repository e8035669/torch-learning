#include <iostream>
#include <ATen/core/stack.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace torch;
using namespace std;

int main(int argc, char** argv) {
    string filename = argv[1];

    torch::jit::Module model = torch::jit::load(filename);

    vector<torch::jit::IValue> input;
    input.push_back(torch::ones({28, 28}));

    Tensor out = model.forward(input).toTensor();

    cout << out << endl;

    return 0;
}
