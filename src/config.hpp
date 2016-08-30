#ifndef __Config_H__
#define __Config_H__

#if 0
#include <armadillo>
typedef arma::mat T;

using namespace arma;

#ifdef ARMA_INCLUDES
#define HAVE_ARMADILLO
#endif

typedef std::pair<T, T> S;

#endif

typedef struct {
    size_t maxIter;
    double lr;
} TrainOpts;


// Configure MatrixOp backend
#define USE_ARMA

#ifdef USE_EIGEN
#include <Eigen/Dense>
using namespace Eigen;

typedef Eigen::MatrixXd mat_t;

#else

#ifdef USE_ARMA
#include <armadillo>
using namespace arma;

typedef arma::mat mat_t;
#endif

#endif

#endif
