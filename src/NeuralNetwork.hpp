#ifndef __NeuralNetwork_H__
#define __NeuralNetwork_H__

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "config.hpp"
#include <assert.h>

// #################
//    Interface
// #################

class NeuralNetwork {
public:
    NeuralNetwork(const char* name = "nnet") {
        this->name = std::string(name);
    };

    virtual void build(const char* filename) = 0;
    virtual void train(const mat_t& x, const mat_t& y) = 0;
    virtual void save(const char* filename) = 0;
protected:
    std::string name;

    size_t inputsize;
    size_t outputsize;

    std::vector<Layer *> layers;
};

class MultiLayerPerceptron: public NeuralNetwork {
public:
    MultiLayerPerceptron(const char *name, size_t inputsize, size_t outputsize);

    virtual void build(const char* filename) {};
    virtual void build(const std::vector<size_t>& layersize);
    virtual void train(const mat_t& x, const mat_t& y) {};
    virtual void train(const mat_t& x, const mat_t& y, TrainOpts* trainopts);
    virtual void save(const char* filename);

    virtual const mat_t& ff(const mat_t& x);

protected:
    void to_dot(const char* filename);
    void to_json(const char* filename);

    size_t nlayers;

    // mean square error (mse)
    double loss;

    mat_t err;

    // delta in each layer. The local error in back propagation phase
    std::vector<mat_t> d;
    // activation in each layer.
    std::vector<mat_t> y;
    // derivative of activation function in each layer.
    std::vector<mat_t> dy;
};

// ################
//  Implementation
// ################

MultiLayerPerceptron::MultiLayerPerceptron(const char *name, size_t inputsize, size_t outputsize) {
    this->name       = std::string(name);

    this->inputsize  = inputsize;
    this->outputsize = outputsize;
}

void MultiLayerPerceptron::build(const std::vector<size_t>& layersize) {
    // build multilayer perceptron model with hiddenlayer size specification
    // assert(layersize.size() > 0)

    // std::clog << "building MultiLayerPerceptron model ..." << std::endl;

    this->layers.clear();

    size_t _inputsize, _outputsize;
    _inputsize = this->inputsize;

    for (size_t i = 0; i < layersize.size(); ++i) {
        _outputsize = layersize[i];
        this->layers.push_back(new HiddenLayer("layer", _inputsize, _outputsize));
        _inputsize  = layersize[i];
    }

    _outputsize = this->outputsize;
    this->layers.push_back(new HiddenLayer("layer", _inputsize, _outputsize));

    this->nlayers = this->layers.size();
}

void MultiLayerPerceptron::train(const mat_t& x, const mat_t& y, TrainOpts* trainopts) {
    // std::clog << "training MultiLayerPerceptron model ..." << std::endl;
    HiddenLayer *layer;

    assert(x.n_cols == y.n_cols);
    size_t nsamples  = x.n_cols;

    // get train options
    // maxIter: maximum number of iterations
    // lr     : learning rate
    size_t maxIter = trainopts->maxIter;
    double lr      = trainopts->lr;

    // pre-allocate for d, y, and dy
    this->d.clear();
    this->d.resize(nlayers + 1);

    this->y.clear();
    this->y.resize(nlayers + 1);

    this->dy.clear();
    this->dy.resize(nlayers + 1);

    // main loop
    for (size_t j = 0; j < maxIter; ++j) {
        std::clog << "Iteration: " << (std::setw(4)) << (j + 1);

        // Stage1: feed forward
        this->y[0] = x;
        for (size_t i = 0; i < nlayers; ++i) {
            layer = dynamic_cast<HiddenLayer *>(this->layers[i]);
            layer->fprop(this->y[i], this->y[i+1], this->dy[i+1]);
        }

        err  = (this->y[nlayers] - y);
        loss = 0.5 * arma::accu(err % err) / (err.n_elem);
        std::clog << " : loss: " << loss << std::endl;

        this->d[nlayers] = err % this->dy[nlayers];

        // Stage2: back propagation
        for (size_t i = this->nlayers - 1; i > 0; --i) {
            layer = dynamic_cast<HiddenLayer *>(this->layers[i]);
            layer->bprop(this->d[i+1], this->d[i]);
            this->d[i] = this->d[i] % this->dy[i];
        }

        // Stage3: update weight and bias
        for (size_t i = 0; i < this->nlayers; ++i) {
            layer = dynamic_cast<HiddenLayer *>(this->layers[i]);
            layer->W = layer->W - lr * this->d[i+1] * this->y[i].t() / nsamples;
            layer->b = layer->b - lr * arma::sum(this->d[i+1], 1) / nsamples;
        }
    }

}

void MultiLayerPerceptron::save(const char* filename) {
    // save model to dot file or a json file according to the ext
    const char *p = filename;
    while (*p != '.') p++;
    p++;
    if (strcmp(p, "dot") == 0) {
        this->to_dot(filename);
    } else if (strcmp(p, "json") == 0) {
        this->to_json(filename);
    } else {
        return;
    }
}

const mat_t& MultiLayerPerceptron::ff(const mat_t& x) {
    // feedforward nn model
    HiddenLayer *layer;

    this->y[0] = x;
    for (size_t i = 0; i < this->nlayers; ++i) {
        layer = dynamic_cast<HiddenLayer *>(this->layers[i]);
        layer->fprop(this->y[i], this->y[i+1], this->dy[i+1]);
    }

    return this->y[nlayers];
}

void MultiLayerPerceptron::to_dot(const char* filename = "nn_mlp.dot") {
    FILE *fp = fopen(filename, "w");
    if (NULL == fp) {
        fp = stdout;
    }

    fprintf(fp, "digraph %s {\n", "nnet");
    fprintf(fp, "\trankdir = LR\n");
    fprintf(fp, "\tnode [shape=\"circle\" label=\"\"]\n\n");

    HiddenLayer *layer;

    for (size_t i = 0; i < this->nlayers; ++i) {
        fprintf(fp, "\tsubgraph layer%d {\n", i);

        layer = dynamic_cast<HiddenLayer *>(this->layers[i]);

        for (size_t m = 0; m < layer->inputsize; ++m) {
            fprintf(fp, "\t\tneuron_%d_%d -> {", i, m);
            for (size_t n = 0; n < layer->outputsize; ++n) {
                fprintf(fp, " neuron_%d_%d ", i + 1, n);
            }
            fprintf(fp, "}\n");
        }
        fprintf(fp, "\t}\n");
    }
    fprintf(fp, "}\n");

    fclose(fp);
}

void MultiLayerPerceptron::to_json(const char* filename = "nn_mlp.json") {
    FILE *fp = fopen(filename, "w");
    if (NULL == fp) {
        fp = stdout;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "{\t\"%s\" : \"%s\",\n", "name", this->name.c_str());
    fprintf(fp, "\t\"%s\" : %d,\n", "inputsize", this->inputsize);
    fprintf(fp, "\t\"%s\" : %d,\n", "outputsize", this->outputsize);

    fprintf(fp, "\t\"layers\" : [\n");

    HiddenLayer *layer = NULL;

    for (size_t i = 0; i < this->nlayers; ++i) {
        layer = dynamic_cast<HiddenLayer *>(this->layers[i]);

        fprintf(fp, "\t{\n\t\t\"%s\" : \"%s\",\n", "name", layer->name.c_str());
        fprintf(fp, "\t\t\"%s\" : %d,\n", "inputsize", layer->inputsize);
        fprintf(fp, "\t\t\"%s\" : %d,\n", "outputsize", layer->outputsize);
        fprintf(fp, "\t\t\"%s\" : \"%s\",\n", "activation", "sigmoid");
        fprintf(fp, "\t\t\"%s\" : \"%s\",\n", "W", "");
        fprintf(fp, "\t\t\"%s\" : \"%s\",\n", "b", "");

        fprintf(fp, "\t},\n");
    }
    fprintf(fp, "\t]\n");
    fprintf(fp, "}\n");

    fclose(fp);
}

#endif
