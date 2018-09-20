#ifndef NETWORK_H_
#define NETWORK_H_

#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "helper.h"

using namespace Eigen;

// typedef function pointer
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;

typedef  RowMatrixXd(*ActivationFunction) (RowMatrixXd const& x);
typedef  RowMatrixXd(*ActivationFunctionDerivative) (RowMatrixXd const& x);


class NeuralNetwork
{
  public:
    NeuralNetwork();
    ~NeuralNetwork();

    void set_nn_structure(int input_size, int num_layers, int* layer_sizes);
    void set_activation(char* name);
    void set_keep_prob(double* keep_prob);
    void add_weight_bias(double** weight, double* bias, int layer);
    void forward(double * zeta, const int rows, const int cols, const int ensemble_index);
    void backward();

    double get_sum_output() {
      return activOutputLayer_.sum();
    }

    double* get_output() {
      return activOutputLayer_.data();
    }

    double* get_grad_input() {
      return gradInput_.data();
    }

  void set_ensemble_size(int repeat);
  int get_ensemble_size();
  void add_dropout_binary(int ensemble_index, int layer_index, int size, int* binary);


//TODO  for debug purpose delete
    void echo_input() {
      std::cout<<"==================================="<<std::endl;
      std::cout<<"Input data for class NeuralNetwork"<<std::endl;
      std::cout<<"inputSize_: "<<inputSize_<<std::endl;
      std::cout<<"Nlayers_: "<<Nlayers_<<std::endl;
      std::cout<<"Nperceptrons_: ";
      for (size_t i=0; i<layerSizes_.size(); i++) {
        std::cout<< layerSizes_.at(i) <<" ";
      }
      std::cout<<std::endl;

      std::cout<<"weights and biases:"<<std::endl;
      for (size_t i=0; i<weights_.size(); i++) {
        std::cout<<"w_"<<i<<std::endl<<weights_.at(i)<<std::endl;
        std::cout<<"b_"<<i<<std::endl<<biases_.at(i)<<std::endl;
      }

      std::cout<<"==================================="<<std::endl;
      std::cout<<"ensemble_size:"<<ensemble_size_<<std::endl;
      for (size_t i=0; i<row_binary_.size(); i++) {
        for (size_t j=0; j<row_binary_[i].size(); j++) {
          std::cout<<"\n\n@@ i="<<i << " j="<<j<<" size="<<row_binary_[i][j].size() << std::endl;
          std::cout<<row_binary_[i][j]<<std::endl;
        }
      }

    }



  private:
    int inputSize_;  // size of input layer
    int Nlayers_;  // number of layers, including output, excluding input
    std::vector<int> layerSizes_;  // number of perceptrons in each layer
    ActivationFunction activFunc_;
    ActivationFunctionDerivative activFuncDeriv_;
    std::vector<RowMatrixXd> weights_;
    std::vector<RowVectorXd> biases_;
    std::vector<RowMatrixXd> preactiv_;
    std::vector<double> keep_prob_;
    std::vector<RowMatrixXd> keep_prob_binary_;
    RowMatrixXd activOutputLayer_;
    RowMatrixXd gradInput_;

    int ensemble_size_;
    std::vector<std::vector<RowMatrixXd>> row_binary_;

    // dropout
    RowMatrixXd dropout_(RowMatrixXd const& x, int layer, int const ensemble_index);


};


// activation fucntion and derivatives
RowMatrixXd relu(RowMatrixXd const& x);
RowMatrixXd relu_derivative(RowMatrixXd const& x);
RowMatrixXd elu(RowMatrixXd const& x);
RowMatrixXd elu_derivative(RowMatrixXd const& x);
RowMatrixXd tanh(RowMatrixXd const& x);
RowMatrixXd tanh_derivative(RowMatrixXd const& x);
RowMatrixXd sigmoid(RowMatrixXd const& x);
RowMatrixXd sigmoid_derivative(RowMatrixXd const& x);


#endif // NETWORK_H_
