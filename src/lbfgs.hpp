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
  V solve(V x, VecFun<V, double> f, GradFun<V> &Gradient) override {
    
    std::vector<V> s_list; 
    std::vector<V> y_list; 
    std::vector<double> rho_list; 

    V grad= Gradient(x);
    V p= -grad;
    V x_new=x;


    V s=x; //initializatione a caso giusto per la dimensione
    V y=grad;



for ( _iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol; ++_iters) {


    grad=Gradient(x);
    p= compute_direction(grad, s_list, y_list, rho_list);

    alpha_wolfe = line_search(x, p, f, Gradient);

      x_new= x + alpha_wolfe * p;
      V s = x_new - x;
      V y = Gradient(x_new) - grad;
      
      x=x_new;

    double rho = 1.0 / y.dot(s);
    s_list.push_back(s);
    y_list.push_back(y);
    rho_list.push_back(rho);

    if (s_list.size() > m) {
      s_list.erase(s_list.begin());
      y_list.erase(y_list.begin());
      rho_list.erase(rho_list.begin());
    }
  }
}


 
V compute_direction(V grad, std::vector<V> &s_list,std::vector<V> &y_list, std::vector<double> &rho_list) {
    if (s_list.size() == 0) {
      return -grad;
    }
  
  V z = V::Zero(grad.size()); //random initialization
      V q = grad;
      std::vector<double> alpha_list(s_list.size());

      double gamma=1.0;



      for (int i = s_list.size() - 1; i >= 0; --i) {
        alpha_list[i] = rho_list[i] * s_list[i].dot(q);
        q -= alpha_list[i] * y_list[i];
      }

    gamma= s_list.back().dot(y_list.back()) / y_list.back().dot(y_list.back());
      
    M H0 = gamma * M::Identity(grad.size(), grad.size());

      z = H0 * q;

      for (size_t i = 0; i < s_list.size(); ++i) {
        double beta = rho_list[i] * y_list[i].dot(z);
        z += s_list[i] * (alpha_list[i] - beta);
      }

      
      z= -z;
      return z;
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
  int m=5;
  double alpha_wolfe = 1e-3;
  double c1 = 1e-4;
  double c2 = 0.9;
  double rho = 0.5;
  double armijo_max_iter = 20;
  double max_line_iters = 20;     

  };

