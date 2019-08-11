//
// CDDL HEADER START
//
// The contents of this file are subject to the terms of the Common Development
// and Distribution License Version 1.0 (the "License").
//
// You can obtain a copy of the license at
// http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
// specific language governing permissions and limitations under the License.
//
// When distributing Covered Code, include this CDDL HEADER in each file and
// include the License file in a prominent location with the name LICENSE.CDDL.
// If applicable, add the following below this CDDL HEADER, with the fields
// enclosed by brackets "[]" replaced with your own identifying information:
//
// Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
//
// CDDL HEADER END
//

//
// Copyright (c) 2019, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Mingjian Wen
//

#include "network.h"

#define LOG_ERROR(msg) \
  (std::cerr << "ERROR (NeuralNetwork): " << (msg) << std::endl)

// Nothing to do at this moment
NeuralNetwork::NeuralNetwork() :
    inputSize_(0),
    Nlayers_(0),
    fully_connected_(false),
    ensemble_size_(0)
{
  return;
}

NeuralNetwork::~NeuralNetwork() {}

int NeuralNetwork::read_parameter_file(FILE * const filePointer, int desc_size)
{
  int ier;
  int endOfFileFlag = 0;
  char nextLine[MAXLINE];
  char errorMsg[1024];
  char name[128];
  int numLayers;
  int * numNodes;

  // number of layers
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%d", &numLayers);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read number of layers from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // number of nodes in each layer
  numNodes = new int[numLayers];
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = getXint(nextLine, numLayers, numNodes);
  if (ier)
  {
    sprintf(errorMsg, "unable to read number of nodes from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  set_nn_structure(desc_size, numLayers, numNodes);

  // activation function
  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%s", name);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read `activation function` from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // register activation function
  lowerCase(name);
  if (strcmp(name, "sigmoid") != 0 && strcmp(name, "tanh") != 0
      && strcmp(name, "relu") != 0 && strcmp(name, "elu") != 0)
  {
    sprintf(errorMsg,
            "unsupported activation function. Expecting `sigmoid`, `tanh` "
            " `relu` or `elu`, given %s.\n",
            name);
    LOG_ERROR(errorMsg);
    return true;
  }
  set_activation(name);

  // keep probability
  double * keep_prob;
  AllocateAndInitialize1DArray<double>(keep_prob, numLayers);

  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  ier = getXdouble(nextLine, numLayers, keep_prob);
  if (ier)
  {
    sprintf(errorMsg, "unable to read `keep probability` from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  set_keep_prob(keep_prob);
  Deallocate1DArray(keep_prob);

  // weights and biases
  for (int i = 0; i < numLayers; i++)
  {
    double ** weight;
    double * bias;
    int row;
    int col;

    if (i == 0)
    {
      row = desc_size;
      col = numNodes[i];
    }
    else
    {
      row = numNodes[i - 1];
      col = numNodes[i];
    }

    AllocateAndInitialize2DArray<double>(weight, row, col);
    for (int j = 0; j < row; j++)
    {
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      ier = getXdouble(nextLine, col, weight[j]);
      if (ier)
      {
        sprintf(errorMsg, "unable to read `weight` from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }

    // bias
    AllocateAndInitialize1DArray<double>(bias, col);
    getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
    ier = getXdouble(nextLine, col, bias);
    if (ier)
    {
      sprintf(errorMsg, "unable to read `bias` from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }

    // copy to network class
    add_weight_bias(weight, bias, i);

    Deallocate2DArray(weight);
    Deallocate1DArray(bias);
  }

  delete[] numNodes;

  // everything is OK
  return false;
}

int NeuralNetwork::read_dropout_file(FILE * const filePointer)
{
  int ier;
  int endOfFileFlag = 0;
  char nextLine[MAXLINE];
  char errorMsg[1024];

  getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
  int ensemble_size;
  ier = sscanf(nextLine, "%d", &ensemble_size);
  if (ier != 1)
  {
    sprintf(errorMsg, "unable to read ensemble_size from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  set_ensemble_size(ensemble_size);

  for (int i = 0; i < ensemble_size; i++)
  {
    for (int j = 0; j < Nlayers_; j++)
    {
      int size;
      if (j == 0) { size = inputSize_; }
      else
      {
        size = layerSizes_[j - 1];
      }

      int * row_binary = new int[size];
      getNextDataLine(filePointer, nextLine, MAXLINE, &endOfFileFlag);
      ier = getXint(nextLine, size, row_binary);
      if (ier)
      {
        sprintf(errorMsg, "unable to read dropout binary from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
      add_dropout_binary(i, j, size, row_binary);
      delete[] row_binary;
    }
  }

  // everything is OK
  return false;
}

void NeuralNetwork::set_nn_structure(int size_input,
                                     int num_layers,
                                     int * layer_sizes)
{
  inputSize_ = size_input;
  Nlayers_ = num_layers;
  for (int i = 0; i < Nlayers_; i++) { layerSizes_.push_back(layer_sizes[i]); }

  weights_.resize(Nlayers_);
  biases_.resize(Nlayers_);
  preactiv_.resize(Nlayers_);
  keep_prob_.resize(Nlayers_);
  keep_prob_binary_.resize(Nlayers_);
}

void NeuralNetwork::set_activation(char * name)
{
  if (strcmp(name, "sigmoid") == 0)
  {
    activFunc_ = &sigmoid;
    activFuncDeriv_ = &sigmoid_derivative;
  }
  else if (strcmp(name, "tanh") == 0)
  {
    activFunc_ = &tanh;
    activFuncDeriv_ = &tanh_derivative;
  }
  else if (strcmp(name, "relu") == 0)
  {
    activFunc_ = &relu;
    activFuncDeriv_ = &relu_derivative;
  }
  else if (strcmp(name, "elu") == 0)
  {
    activFunc_ = &elu;
    activFuncDeriv_ = &elu_derivative;
  }
}

void NeuralNetwork::set_keep_prob(double * keep_prob)
{
  for (int i = 0; i < Nlayers_; i++) { keep_prob_[i] = keep_prob[i]; }
}

void NeuralNetwork::add_weight_bias(double ** weight, double * bias, int layer)
{
  int rows;
  int cols;

  if (layer == 0)
  {
    rows = inputSize_;
    cols = layerSizes_[layer];
  }
  else
  {
    rows = layerSizes_[layer - 1];
    cols = layerSizes_[layer];
  }

  // copy data
  RowMatrixXd w(rows, cols);
  RowVectorXd b(cols);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++) { w(i, j) = weight[i][j]; }
  }
  for (int j = 0; j < cols; j++) { b(j) = bias[j]; }

  // store in vector
  weights_[layer] = w;
  biases_[layer] = b;
}

void NeuralNetwork::set_ensemble_size(int repeat)
{
  ensemble_size_ = repeat;
  row_binary_.resize(repeat);
  for (size_t i = 0; i < row_binary_.size(); i++)
  { row_binary_[i].resize(Nlayers_); }
}

void NeuralNetwork::add_dropout_binary(int ensemble_index,
                                       int layer_index,
                                       int size,
                                       int * binary)
{
  RowMatrixXd data(1, size);
  for (int i = 0; i < size; i++) { data(0, i) = binary[i]; }
  row_binary_[ensemble_index][layer_index] = data;
}

void NeuralNetwork::forward(double * zeta,
                            const int rows,
                            const int cols,
                            int const ensemble_index)
{
  RowMatrixXd act;

  // map raw C++ data into Matrix data
  // see: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
  Map<RowMatrixXd> activation(zeta, rows, cols);
  act = activation;

  for (int i = 0; i < Nlayers_; i++)
  {
    // apply dropout
    if (fully_connected_ == false && keep_prob_[i] < 1 - 1e-10)
    {
      act = dropout_(act, i, ensemble_index);  // no aliasing will occur for act
    }

    preactiv_[i] = (act * weights_[i]).rowwise() + biases_[i];

    if (i == Nlayers_ - 1)
    {  // output layer (no activation function applied)
      activOutputLayer_ = preactiv_[i];
    }
    else
    {
      act = activFunc_(preactiv_[i]);
    }
  }
}

void NeuralNetwork::backward()
{
  // our cost (energy E) is the sum of activations at output layer, and no
  // activation function is employed in the output layer
  int rows = preactiv_[Nlayers_ - 1].rows();
  int cols = preactiv_[Nlayers_ - 1].cols();

  // error at output layer
  RowMatrixXd delta = RowMatrixXd::Constant(rows, cols, 1.0);

  for (int i = Nlayers_ - 2; i >= 0; i--)
  {
    // eval() is used to prevent aliasing since delta is both lvalue and rvalue.
    delta = (delta * weights_[i + 1].transpose())
                .eval()
                .cwiseProduct(activFuncDeriv_(preactiv_[i]));

    // apply dropout
    if (fully_connected_ == false && keep_prob_[i + 1] < 1 - 1e-10)
    {
      delta = delta.cwiseProduct(keep_prob_binary_[i + 1]) / keep_prob_[i + 1];
    }
  }

  gradInput_ = delta * weights_[0].transpose();
  // apply dropout
  if (fully_connected_ == false && keep_prob_[0] < 1 - 1e-10)
  {
    gradInput_ = gradInput_.cwiseProduct(keep_prob_binary_[0]) / keep_prob_[0];
  }
}

// dropout
RowMatrixXd NeuralNetwork::dropout_(RowMatrixXd const & x,
                                    int layer,
                                    int const ensemble_index)
{
  RowMatrixXd y;
  double keep_prob = keep_prob_[layer];

  if (fully_connected_ == false && keep_prob < 1 - 1e-10)
  {
    //// do it within model
    //// uniform [-1, 1]
    // RowMatrixXd random = RowMatrixXd::Random(1, x.cols());
    //// uniform [keep_prob, 1+keep_prob] .floor()
    // random =( (random/2.).array() + 0.5 + keep_prob ).floor();

    // read in from file
    RowMatrixXd random = row_binary_[ensemble_index][layer];

    // each row is the same (each atom is treated the same)
    keep_prob_binary_[layer] = random.replicate(x.rows(), 1);

    y = (x / keep_prob).cwiseProduct(keep_prob_binary_[layer]);
  }
  else
  {
    y = x;
  }

  return y;
}

//*****************************************************************************
// activation functions and derivatives
//*****************************************************************************

RowMatrixXd relu(RowMatrixXd const & x) { return x.cwiseMax(0.0); }

RowMatrixXd relu_derivative(RowMatrixXd const & x)
{
  RowMatrixXd deriv(x.rows(), x.cols());

  for (int i = 0; i < x.rows(); i++)
  {
    for (int j = 0; j < x.cols(); j++)
    {
      if (x(i, j) < 0.) { deriv(i, j) = 0.; }
      else
      {
        deriv(i, j) = 1.;
      }
    }
  }
  return deriv;
}

RowMatrixXd elu(RowMatrixXd const & x)
{
  double alpha = 1.0;
  RowMatrixXd e(x.rows(), x.cols());

  for (int i = 0; i < x.rows(); i++)
  {
    for (int j = 0; j < x.cols(); j++)
    {
      if (x(i, j) < 0.) { e(i, j) = alpha * (exp(x(i, j)) - 1); }
      else
      {
        e(i, j) = x(i, j);
      }
    }
  }
  return e;
}

RowMatrixXd elu_derivative(RowMatrixXd const & x)
{
  double alpha = 1.0;
  RowMatrixXd deriv(x.rows(), x.cols());

  for (int i = 0; i < x.rows(); i++)
  {
    for (int j = 0; j < x.cols(); j++)
    {
      if (x(i, j) < 0.) { deriv(i, j) = alpha * exp(x(i, j)); }
      else
      {
        deriv(i, j) = 1.;
      }
    }
  }
  return deriv;
}

RowMatrixXd tanh(RowMatrixXd const & x) { return (x.array().tanh()).matrix(); }

RowMatrixXd tanh_derivative(RowMatrixXd const & x)
{
  return (1.0 - x.array().tanh().square()).matrix();
}

RowMatrixXd sigmoid(RowMatrixXd const & x)
{
  return (1.0 / (1.0 + (-x).array().exp())).matrix();
}

RowMatrixXd sigmoid_derivative(RowMatrixXd const & x)
{
  RowMatrixXd s = sigmoid(x);

  return (s.array() * (1.0 - s.array())).matrix();
}
