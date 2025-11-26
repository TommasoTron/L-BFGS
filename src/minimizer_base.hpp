#pragma once

#include "common.hpp"
#include <eigen3/Eigen/Eigen>

template<typename V, typename M>
class MinimizerBase {
public:
  int iterations() const noexcept {
    return _iters;
  }

  double tolerance() const noexcept {
    return _tol;
  }
  
  void setMaxIterations(int max_iters) noexcept {
    _max_iters = max_iters;
  }

  void setTolerance(double tol) noexcept { _tol = tol; }

  virtual V solve(V x, M b, VecFun<V,double> f, GradFun<V> &Gradient) = 0;
  
protected:
  unsigned int _max_iters = 1000;
  unsigned int _iters = 0;
  double _tol = 1.e-6;
};
