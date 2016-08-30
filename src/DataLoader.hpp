#ifndef __DataLoader_H__
#define __DataLoader_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>

// #################
//    Interface
// #################

template<class T>
class DataLoader {
public:
    virtual bool load(const char* filename, std::vector<T>& result) = 0;
};

// Load csv file
template<class T>
class CsvDataLoader: public DataLoader<T> {
public:
    CsvDataLoader(const char* delimiter = ",", size_t skip = 0);
    bool load(const char* filename, std::vector<T>& result);
protected:
    const char* delimiter;
    size_t skip;
};

// ##################
//   Implementation
// ##################

template<class T>
CsvDataLoader<T>::CsvDataLoader(const char* delimiter, size_t skip) {
    this->delimiter = std::string(delimiter).c_str();
    this->skip      = skip;
}

template<class T>
bool CsvDataLoader<T>::load(const char* filename, std::vector<T>& result) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "can not locate csv data file " << filename << std::endl;
        return false;
    }

    std::copy(std::istream_iterator<T>(ifs), std::istream_iterator<T>(), std::back_inserter(result));

    ifs.close();
    return true;
}

template<>
bool CsvDataLoader<std::string>::load(const char* filename, std::vector<std::string>& result) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "can not locate csv data file " << filename << std::endl;
        return false;
    }

    copy(std::istream_iterator<std::string>(ifs), std::istream_iterator<std::string>(), std::back_inserter(result));

    ifs.close();
    return true;
}

#endif
