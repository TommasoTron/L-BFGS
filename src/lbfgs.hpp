#pragma once

#include <eigen3/Eigen/Eigen>
#include "common.hpp"
#include "minimizer_base.hpp"

template <typename V, typename M> class LBFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  // V vector, M matrix
  V solve(V x, M B, VecFun<V, double> f, GradFun<V> &Gradient) override {
    
  }

private:
  
};

