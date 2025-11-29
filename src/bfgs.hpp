#pragma once

#include <eigen3/Eigen/Eigen>
#include "common.hpp"
#include "minimizer_base.hpp"
#include <limits>


template <typename V, typename M> class BFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  // V vector, M matrix
  V solve(V x, M B, VecFun<V, double> f, GradFun<V> &Gradient) override {
    Eigen::ConjugateGradient<M> solver;

    for (_iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol; ++_iters) {
      solver.compute(B);
      check(solver.info() == Eigen::Success, "conjugate gradient solver error");
     
      V p = solver.solve(-Gradient(x));
      double alpha = 1.0;
      


      alpha = line_search(x, p, f, Gradient);

      V s = alpha * p;
      V x_next = x + s;

      V y = Gradient(x_next) - Gradient(x);

      M b_prod = B * s;
      B = B + (y * y.transpose()) / (y.transpose() * s) -
          (b_prod * b_prod.transpose()) / (s.transpose() * B * s);
      x = x_next;
    }

    return x;
  }

double line_search(V x, V p, VecFun<V, double> &f, GradFun<V> &Gradient) {
    double f_x = f(x);
    double grad_f_old = Gradient(x).dot(p);
    
    double inf = std::numeric_limits<double>::infinity();
    double alpha_min = 0.0;
    double alpha_max = inf;

    alpha = 1.0;
  
    for (int i=0; i< max_line_iters;++i){
      V x_new = x + alpha * p;
      double f_new = f(x_new);
      double grad_f_new = Gradient(x_new).dot(p);
      
    

      for (int armijo_iter =0; armijo_iter < armijo_max_iter; ++armijo_iter){
      if (f_new > f(x) + c1 * alpha * grad_f_old){
        alpha_max = alpha;
        alpha= rho * (alpha_min + alpha_max);
      }
      else {
        break;
      }
      }

      if (grad_f_new >= c2* grad_f_old){
        if (alpha_max < inf){
          alpha_min =alpha;
          alpha_max=rho * (alpha_min + alpha_max);
        }
        else {
          alpha_min=alpha;
          alpha= 5 * alpha;
        }
      }


    

    }
    return alpha;
  }


private:
  double alpha = 1.0;
  double c1 = 1e-4;
  double c2 = 0.9;
  double rho = 0.5;
  double armijo_max_iter = 20;
  double max_line_iters = 20;     
};
