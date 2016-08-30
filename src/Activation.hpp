#ifndef __Activation_H__
#define __Activation_H__

#include "config.hpp"
#if 1
#include <armadillo>
typedef arma::mat mat_t;
#endif

typedef enum {
    SIGMOID = 0,
    TANH,
    TANHOPT,
    RELU
} activation_t;

class Activation {
public:
    virtual feed(const mat_t& x, mat_t& y, mat_t& dy) { };
    virtual void operator()(const mat_t &x, mat_t &y) = 0;
    virtual void operator()(const mat_t &x, mat_t &y, mat_t &yd) = 0;
};

class ActSigmoid: public Activation {
public:
    void operator()(const mat_t &x, mat_t &y) {
        y = 1 / (1 + exp(-x));
    };
    void operator()(const mat_t &x, mat_t &y, mat_t &yd) {
        y = 1 / (1 + exp(-x));
        yd = y * (1 - y);
    };
};

class ActTanh: public Activation {
public:
    void operator()(const mat_t &x, mat_t &y) {
        y = tanh(x);
    };
    void operator()(const mat_t &x, mat_t &y, mat_t &yd) {
        y = tanh(x);
        yd = 1 - y % y;
    };
};

class ActTanhOpt: public Activation {
public:
    void operator()(const mat_t &x, mat_t &y) {
    	y = 1.7159 * tanh(2 / 3 * x);
    };
    void operator()(const mat_t &x, mat_t &y, mat_t &yd) {
        y = 1.7159 * tanh(2 / 3 * x);
        yd = 1.7159 * 2 / 3 * (1 - 1 / (1.7159 * 1.7159) * x % x);
    };
};


class ActivationFactory {
public:
    static const Activation* getActivationInstance(activation_t type = SIGMOID);
};

const Activation* ActivationFactory::getActivationInstance(activation_t type) {
    switch (type) {
    case SIGMOID:
        return new ActSigmoid();
        break;
    case TANH:
        return new ActTanh();
        break;
    case TANHOPT:
        return new ActTanhOpt();
        break;
    default:
        break;
    }
}

static void sigmoid(const mat_t& x, mat_t& y, mat_t& dy) {
    y  = 1 / (1 + arma::exp(-x));
    dy = y % (1 - y);
}

static void tanh(const mat_t& x, mat_t& y, mat_t& dy) {
    y  = 1 / (1 + arma::tanh(-x));
    dy = 1 - y % y;
}

#endif
