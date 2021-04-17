#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/stack.h>
#include <c10/core/DeviceType.h>
#include <c10/util/in_place.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/cuda.h>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/example.h>
#include <torch/data/transforms/stack.h>
#include <torch/jit.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/options/loss.h>
#include <torch/nn/options/pooling.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/sgd.h>
#include <torch/serialize.h>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

#include <cstdio>
#include <iostream>

using namespace std;
using namespace torch::nn;
using namespace torch::optim;
using torch::Tensor;

struct BasicBlockImpl : torch::nn::Module {
    BasicBlockImpl(int inplanes, int planes, int stride, Sequential downsample)
        : conv1(Conv2dOptions(inplanes, planes, 3)
                    .stride(stride)
                    .padding(1)
                    .bias(false)),
          bn1(BatchNorm2dOptions(planes)),
          relu(ReLUOptions(true)),
          conv2(Conv2dOptions(planes, planes, 3)
                    .stride(1)
                    .padding(1)
                    .bias(false)),
          bn2(BatchNorm2dOptions(planes)),
          downsample(downsample) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (!downsample.is_empty()) {
            register_module("downsample", downsample);
        }
    }

    Tensor forward(Tensor x) {
        Tensor identity = x;
        Tensor out = relu(bn1(conv1(x)));
        out = bn2(conv2(out));
        if (!downsample.is_empty()) {
            identity = downsample->forward(x);
        }
        out += identity;
        out = torch::relu(out);
        return out;
    }

    Conv2d conv1, conv2;
    ReLU relu;
    BatchNorm2d bn1, bn2;
    Sequential downsample;
};

TORCH_MODULE(BasicBlock);

struct Resnet18Impl : public torch::nn::Module {
    Resnet18Impl(int num_classes)
        : inplanes(64),
          num_classes(num_classes),
          conv1(Conv2dOptions(1, inplanes, 7).stride(2).padding(3).bias(false)),
          bn1(BatchNorm2dOptions(inplanes)),
          relu(ReLUOptions(true)),
          maxpool(MaxPool2dOptions(3).stride(2).padding(1)),
          layer1(make_layer(64, 2, 1)),
          layer2(make_layer(128, 2, 2)),
          layer3(make_layer(256, 2, 2)),
          layer4(make_layer(512, 2, 2)),
          avgpool(AdaptiveAvgPool2dOptions({1, 1})),
          fc(LinearOptions(512, num_classes)) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("relu", relu);
        register_module("maxpool", maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("avgpool", avgpool);
        register_module("fc", fc);
    }

    Sequential make_layer(int planes, int blocks, int stride) {
        Sequential layers;
        Sequential downsample{nullptr};
        if (stride != 1 || inplanes != planes) {
            downsample = Sequential{Conv2d(Conv2dOptions(inplanes, planes, 1)
                                               .stride(stride)
                                               .bias(false)),
                                    BatchNorm2d(BatchNorm2dOptions(planes))};
        }
        layers->push_back(BasicBlock(inplanes, planes, stride, downsample));
        inplanes = planes;
        for (int i = 1; i < blocks; ++i) {
            layers->push_back(BasicBlock(inplanes, planes, 1, nullptr));
        }
        return layers;
    }

    Tensor forward(Tensor x) {
        x = conv1(x);
        x = bn1(x);
        x = relu(x);
        x = maxpool(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = avgpool(x);
        x = torch::flatten(x, 1);
        x = fc(x);
        return x;
    }

    int inplanes;
    int num_classes;

    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu;
    MaxPool2d maxpool;
    Sequential layer1, layer2, layer3, layer4;
    AdaptiveAvgPool2d avgpool;
    Linear fc;
};

TORCH_MODULE(Resnet18);

int main(int argc, char** argv) {
    int64_t epochs = 30;
    int64_t test_interval = 1;
    int64_t save_interval = 1;

    torch::manual_seed(1);
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        cout << "Training on GPU!" << endl;
        device = torch::kCUDA;
    }

    Resnet18 resnet(10);

    resnet->to(device);

    auto dataset = torch::data::datasets::MNIST("./data").map(
        torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
        move(dataset), torch::data::DataLoaderOptions().batch_size(64));

    auto valid_dataset =
        torch::data::datasets::MNIST("./data",
                                     torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Stack<>());
    auto valid_dataloader = torch::data::make_data_loader(
        move(valid_dataset), torch::data::DataLoaderOptions(64));

    auto loss = CrossEntropyLoss();
    SGD trainner(resnet->parameters(), SGDOptions(0.00001));

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        int64_t index = 0;
        resnet->train();
        Tensor totalloss = torch::zeros(1, device).requires_grad_(false);
        for (torch::data::Example<>& b : *data_loader) {
            Tensor input = b.data.to(device);
            Tensor target = b.target.to(device);

            Tensor output = resnet->forward(input);
            Tensor c_loss = loss->forward(output, target);
            c_loss.backward();
            trainner.step();

            totalloss += c_loss;

            printf("\r[%ld/%ld][%ld] loss: %lf", epoch, epochs, index,
                   c_loss.item<double>());

            index++;
        }
        printf("\n[%ld/%ld] totalloss: %lf\n", epoch, epochs,
               totalloss.item<double>());

        if (epoch % test_interval == 0) {
            resnet->eval();
            int64_t total = 0;
            int64_t correct = 0;
            for (torch::data::Example<>& b : *valid_dataloader) {
                Tensor x = b.data.to(device);
                Tensor target = b.target.to(device);

                Tensor output = resnet->forward(x);
                auto topk = output.topk(1, 1);
                Tensor out_indice = get<1>(topk).flatten();

                Tensor corr = target.eq(out_indice);

                total += b.data.size(0);
                correct += corr.sum(0).item().toLong();
            }
            printf("Epoch: %ld, Accuracy: %lf {%ld/%ld}\n", epoch,
                   (double)correct / total, correct, total);
        }

        if (epoch % save_interval == 0) {
            torch::save(resnet, "resnet-checkpoint.pt");
            torch::save(trainner, "trainner-checkpoint.pt");
            cout << "Save checkpoint" << endl;
        }
    }

    return 0;
}
