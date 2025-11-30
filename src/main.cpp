#include "bfgs.hpp"
#include "lbfgs.hpp"
#include "minimizer_base.hpp"
#include <eigen3/Eigen/Eigen>
#include <memory>

//Trying to automate derivative computation
#include <autodiff/reverse/var.hpp> //we could also use /forward/ but reverse should be more efficient for gradients like O(1) vs O(n) (sempre se fossi attento a quella lezione di naml)
#include <autodiff/reverse/var/eigen.hpp> //reverse is also easier to write 



using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {
  VecFun<Vec, double> f = [](Vec v) {
    double x1 = v(0);
    double x2 = v(1);
    return 100 * std::pow((x2 - x1 * x2), 2.0) + std::pow((1 - x1), 2.0);
  };

  auto f_var=[](Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1> x) {
    return 100 * pow((x(1) - x(0) * x(1)), 2.0) + pow((1.0 - x(0)), 2.0);
  };

  GradFun<Vec> grad = [f_var](Vec v) {

    Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1> x(v.size());
    for (int i=0;i<v.size();++i)
      x(i)=v(i);

autodiff::var y=f_var(x);
    Vec dy_dv = autodiff::gradient(y, x);
    
    return dy_dv;


  };

  Vec v(2);
  v(0) = -1.5;
  v(1) = -15;

  Mat m(2, 2);
  m.setIdentity();

  using minimizerPtr = std::unique_ptr<MinimizerBase<Vec, Mat>>;

  minimizerPtr bfgs = std::make_unique<BFGS<Vec, Mat>>();
  minimizerPtr lbfgs = std::make_unique<LBFGS<Vec, Mat>>();

  auto assert_solver = [&](minimizerPtr &solver, std::string name) {
    solver->setMaxIterations(4000);
    solver->setTolerance(1.e-12);

    Vec result = solver->solve(v, m, f, grad);
    std::cout << "========" << name << "========" << std::endl;
    std::cout << "computed result: " << std::endl
              << result << std::endl
              << std::endl;
    std::cout << "iterations: " << solver->iterations() << std::endl;
    std::cout << "tolerance: " << solver->tolerance() << std::endl;
    std::cout << "error norm: " << grad(result).norm() << std::endl;
    std::cout << std::endl;
  };

  assert_solver(bfgs, "BFGS");
  assert_solver(lbfgs, "LBFGS");

  return 0;
}
