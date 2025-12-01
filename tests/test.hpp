#include "../src/minimizer_base.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace Tests {

template <typename V, typename M>
class TestSuite {
  using minimizerPtr = std::unique_ptr<MinimizerBase<V, M>>;

public:
  void addImplementation(minimizerPtr &ptr, std::string name) {
    impls[name] = ptr;
  }

  void addTest(std::string name, std::function<void(minimizerPtr &)>) {
    tests.push_back()
  }

private:
  std::map<std::string, minimizerPtr &> impls;
  std::vector<std::pair<std::string, std::function<void(minimizerPtr&)>>> tests;

  
};

} // namespace Tests
