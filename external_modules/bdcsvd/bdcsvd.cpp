#include <Eigen/Dense>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;

// The core function for computing SVD
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> computeSVD(const Eigen::MatrixXd &mat) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    return {svd.matrixU(), svd.singularValues(), svd.matrixV().transpose()};
}

namespace py = pybind11;

PYBIND11_MODULE(svd_module, m) {
    m.def("computeSVD", &computeSVD, "Compute the SVD of a matrix");
}
