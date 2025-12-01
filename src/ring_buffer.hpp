#include <deque>
#include "common.hpp"

template <typename T>
class CircularBuffer {
private:
    std::deque<T> data;
    size_t max_size;

public:
    CircularBuffer(size_t size) : max_size(size) {
      check((size > 0), "size should be positive");
    }

    void push_back(const T& value) {
        data.push_back(value);
        if (data.size() > max_size) 
            data.pop_front(); 
    }

    size_t size() const { return data.size(); }
    
    const T& operator[](size_t index) const {
        check((index >= 0 && index < data.size()), "index out of bounds");
        return data[index];
    }
    
    typename std::deque<T>::iterator begin() { return data.begin(); }
    typename std::deque<T>::iterator end() { return data.end(); }
    typename std::deque<T>::const_iterator begin() const { return data.cbegin(); }
    typename std::deque<T>::const_iterator end() const { return data.cend(); }
};
