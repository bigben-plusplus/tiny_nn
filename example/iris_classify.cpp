#include <iostream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <getopt.h>

#include "Iris.cpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "DataLoader.hpp"
#include "config.hpp"

//#include "matrix.h"

using namespace std;

// utility function: find the index of max value in each column
static std::vector<int> argmax(const mat_t& x)
{
    arma::uvec idx;
    std::vector<int> index(x.n_cols);
    for (size_t i = 0; i < x.n_cols; ++i) {
        idx = arma::sort_index(x.col(i), "descend");
        index[i] = idx(0);
    }

    return index;
}

// utility function: classification performance evaluation
static void evaluate(const mat_t& score, const mat_t& ground)
{
    assert(score.n_cols == ground.n_cols);

    std::cout << "truth: " << std::endl;
    std::vector<int> truth = argmax(ground);
    std::copy(truth.begin(), truth.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "decision: " << std::endl;
    std::vector<int> decision = argmax(score);
    std::copy(decision.begin(), decision.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    int count = 0;
    int total = score.n_cols;
    assert(total > 0);

    for (size_t i = 0; i < total; ++i) {
        if (decision[i] == truth[i]) count++;
    }
    fprintf(stdout, "accuracy: %.3f%%\n", ((double)(count) / (double)(total))*100.0);
}

int main(int argc, char *argv[])
{
    TrainOpts trainopts;
    trainopts.maxIter = 100;
    trainopts.lr      = 1e-1;

    const char *iris_dat = "data/iris.csv";
    int shuffle = 1;

    // parse options
    char ch;
    while ((ch = getopt(argc, argv, "hvsk:r:f:")) != EOF) {
        switch (ch) {
        case 'h':
            fprintf(stdout, "iris_example [options], where options are: \n");
            fprintf(stdout, "  -h\t\t print this help message\n");
            fprintf(stdout, "  -v\t\t print version message\n");
            fprintf(stdout, "  -s\t\t shuffle data before training (default: true)\n");
            fprintf(stdout, "  -k\t\t maximum number of iteration (default: 25)\n");
            fprintf(stdout, "  -r\t\t learning rate (default: 1e-2)\n");
            fprintf(stdout, "  -f\t\t path to iris data file (default: ./data/iris.csv)\n");
            exit(0);

            break;
        case 'v':
            fprintf(stdout, "version 1.0.0\n");
            exit(0);

            break;
        case 's':
            shuffle = 1;
            break;
        case 'k':
            trainopts.maxIter = atoi(optarg);
            break;
        case 'r':
            trainopts.lr      = atof(optarg);
            break;
        case 'f':
            iris_dat = optarg;
            break;
        case '?':
            if (optopt == 'k' || optopt == 'r' || optopt == 'f') {
                fprintf(stderr, "option %c has an argument\n", optopt);
                exit(-1);
            } else if (isprint(optopt)) {
                fprintf(stderr, "invalid option %c\n", optopt);
                exit(-1);
            } else {
                fprintf(stderr, "invalid option \\0x%x\n", optopt);
                exit(-1);
            }
            break;
        default:
            break;
        }
    }

    fprintf(stdout, "Iris classification task using NN model\n\n");

    fprintf(stdout, "loading iris data from %s ...\n", iris_dat);
    CsvDataLoader<Iris> *loader = new CsvDataLoader<Iris>("iris_loader", 0);

    vector<Iris> result;
    result.clear();

    if (false == loader->load(iris_dat, result)) {
        fprintf(stderr, "loading iris data failed, exit ...\n");
        exit(-1);
    }
    //std::copy(result.begin(), result.end(), ostream_iterator<Iris>(std::cout, "\n"));

    mat_t feature, label;

    if (shuffle) {
        srand(time(NULL));
        std::random_shuffle(result.begin(), result.end());
    }

    Iris::load_feature_label(result, feature, label);

    int nsamples = feature.n_rows;
    assert(nsamples > 0);

    int k = (int)(0.6 * nsamples) + 1;

    mat_t x = feature.rows(0, k - 1).t();
    mat_t y = label.rows(0, k - 1).t();

    // create a multi-layer perceptron
    MultiLayerPerceptron* nnet = new MultiLayerPerceptron("mlp", 4, 3);

    std::vector<size_t> arch;
    //arch.push_back(3);
    arch.push_back(5);

    fprintf(stdout, "building MultiLayerPerceptron model ...\n");
    nnet->build(arch);

    fprintf(stdout, "training MultiLayerPerceptron model with options: [maxIter=%d, learning rate=%g]\n", trainopts.maxIter, trainopts.lr);
    nnet->train(x, y, &trainopts);

    fprintf(stdout, "testing MultiLayerPerceptron model ...\n");
    fprintf(stdout, "performance on train set\n");
    mat_t scores = nnet->ff(x);
    evaluate(scores, y);

    fprintf(stdout, "performance on test set\n");
    x = feature.rows(k, nsamples - 1).t();
    y = label.rows(k, nsamples - 1).t();

    scores = nnet->ff(x);
    //scores.save("scores.dat", raw_ascii);
    evaluate(scores, y);

    // save model to .dot or .json file
    fprintf(stdout, "saving model to nn_mlp4iris.dot ...\n");
    nnet->save("nn_mlp4iris.dot");
    fprintf(stdout, "saving model to nn_mlp4iris.json ...\n");
    nnet->save("nn_mlp4iris.json");

    return 0;
}
