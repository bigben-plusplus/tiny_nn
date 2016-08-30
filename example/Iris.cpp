#include <iostream>

#include "config.hpp"
#if 1
#include <armadillo>
typedef arma::mat mat_t;
#endif

class Iris {
public:
    double feature[4];
    std::string label;

    static void load_feature_label(const std::vector<Iris> &v, mat_t& feature, mat_t&label);

    friend std::istream& operator >>(std::istream& in, Iris& iris);
    friend std::ostream& operator <<(std::ostream& out, const Iris& iris);
};

// override input operator >> for class Iris
std::ostream& operator<< (std::ostream& out, const Iris& iris) {
    std::cout << "Iris([" << (iris.feature[0]) << "," << iris.feature[1] << "," \
              << (iris.feature[2]) << "," << iris.feature[3] << "], " << iris.label << ")";
    return out;
}

// override output operator << for class T
std::istream& operator>> (std::istream& in, Iris& iris) {
    std::string str;
    std::getline(in, str);

    if (str != "") {
        const char* buf = str.c_str();
        //double a, b, c, d;
        char label_buf[32];

        sscanf(buf, "%lf,%lf,%lf,%lf,%s", \
               &iris.feature[0], \
               &iris.feature[1], \
               &iris.feature[2], \
               &iris.feature[3], \
               label_buf);

        iris.label = std::string(label_buf);

        // std::cout << iris << std::endl;
    }

    return in;
}

void Iris::load_feature_label(const std::vector<Iris> &v, mat_t& feature, mat_t&label) {
    // std::transform(v.begin(), v.end(), xx);

    size_t nsamples = v.size();
    feature = arma::zeros<arma::mat>(nsamples, 4);
    label   = arma::zeros<arma::mat>(nsamples, 3);

    for (size_t i = 0; i < nsamples; ++i) {
        // parse feature
        feature(i, 0) = v[i].feature[0];
        feature(i, 1) = v[i].feature[1];
        feature(i, 2) = v[i].feature[2];
        feature(i, 3) = v[i].feature[3];

        // parse label
        std::string str = v[i].label;
        if (str == std::string("Iris-setosa")) {
            label(i, 0) = 1;
        } else if (str == std::string("Iris-versicolor")) {
            label(i, 1) = 1;
        } else if (str == std::string("Iris-virginica")) {
            label(i, 2) = 1;
        }

    }
}
