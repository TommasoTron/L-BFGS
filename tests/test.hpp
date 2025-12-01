#include "../src/minimizer_base.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <iostream>
#include <chrono>

namespace Tests {

template <typename V, typename M>
class TestSuite {
  using minimizerPtr = std::shared_ptr<MinimizerBase<V, M>>;
  using testFunction = std::function<void(minimizerPtr &)>;

public:
  TestSuite() {
    impls = std::map<std::string, minimizerPtr>();
    tests = std::vector<std::pair<std::string, testFunction>>();
  }
  
  void addImplementation(minimizerPtr ptr, std::string name) {
    impls[name] = ptr;
  }

  void addTest(std::string name,  testFunction fun) {
    tests.push_back(std::make_pair(name, fun));
  }

  void runTests(){
    for(std::pair<std::string, testFunction>& test : tests){
      std::cout << "======================"
                   "RUNNING TEST:"
                << test.first << "======================" << std::endl;

      for (auto &impl : impls) {
        std::cout << "  implementation: " << impl.first << std::endl;
        auto before = std::chrono::steady_clock::now();

        test.second(impl.second);

        auto after = std::chrono::steady_clock::now();
        auto delta_us = std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();

        std::cout << "\t time elapsed: " << delta_us << " us" << std::endl;
        std::cout << "\t iterations:   " << impl.second->iterations() << std::endl;
        std::cout << "\t tolerance:    " << impl.second->tolerance() << std::endl;
      }
      
    }
  }

private:
  std::map<std::string, minimizerPtr> impls;
  std::vector<std::pair<std::string, testFunction>> tests;

  
};

} // namespace Tests
