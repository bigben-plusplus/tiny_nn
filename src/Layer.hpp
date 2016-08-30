#ifndef __Layer_H__
#define __Layer_H__

#include <iostream>
#include <string>

#include "Activation.hpp"
#include "config.hpp"

// #################
//     Interface
// #################

class Layer {
public:
    virtual void fprop(const mat_t &x, mat_t& y) {};
protected:
    std::string name;

    size_t inputsize;
    size_t outputsize;
};

class HiddenLayer: public Layer {
public:
    HiddenLayer(const char* name, size_t inputsize = 1, size_t outputsize = 1);
    ~HiddenLayer() {
    	delete activation;
    };
    virtual void fprop(const mat_t &x, mat_t& y, mat_t& dy);
    virtual void bprop(const mat_t &x, mat_t& y);

    friend class MultiLayerPerceptron;
protected:
    // weight matrix of size (#outputsize, #inputsize)
    mat_t W;
    // bias vector (#outputsize, 1)
    mat_t b;
private:
	const Activation* activation = NULL;
};

// ################
//  Implementation
// ################

HiddenLayer::HiddenLayer(const char* name, size_t inputsize, size_t outputsize) {
    // construct HiddenLayer with specification: (inputsize, outputsize)
    this->inputsize  = inputsize;
    this->outputsize = outputsize;

    this->name       = std::string(name);
    
    this->activation = ActivationFactory().getActivationInstance(SIGMOID);

    double r = sqrt(6.0 / (inputsize + outputsize));

// #ifdef USE_ARMA
    this->W          = arma::randu<arma::mat>(outputsize, inputsize) * 2 * r - r;
    this->b          = arma::zeros<arma::mat>(outputsize, 1);
// #endif
}

void HiddenLayer::fprop(const mat_t& x, mat_t& y, mat_t& dy) {
    // feed forward input x to the next layer,
    // output to y (activation) and dy (the derivation of activation function)

    sigmoid(this->W * x + arma::repmat(this->b, 1, x.n_cols), y, dy);
}

void HiddenLayer::bprop(const mat_t& x, mat_t& y) {
    // back propagate input x to the previous layer (for BP),
    // output to y

    y = (this->W.t() * x);
}

#endif
