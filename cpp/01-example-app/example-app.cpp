#include <torch/torch.h>
#include <iostream>

int main(int argc, char** argv) {
    torch::Tensor tensor = torch::rand({2, 3});
    tensor = tensor.cuda();
    std::cout << tensor << std::endl;

    return 0;
}
