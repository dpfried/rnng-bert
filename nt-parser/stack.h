#ifndef STACK_H
#define STACK_H

#include <algorithm>
#include <assert.h>
#include <memory>
#include <stdexcept>
#include <vector>

template <class T>
class Stack {
public:
  Stack(const T& value)
      : data(std::make_shared<Data>(value)) {}

  Stack(const T& value, const Stack& previous)
      : data(std::make_shared<Data>(value, previous)) {}

  Stack(const std::vector<T>& values) {
    assert(!values.empty());
    Stack stack(values.front());
    for (auto it = values.begin() + 1; it != values.end(); ++it)
      stack = Stack(*it, stack);
    data = stack.data;
  }

  const T& back() const {
    return data->value;
  }

  Stack push_back(const T& value) const {
    return Stack(value, *this);
  }

  Stack pop_back() const {
    if (!data->previous)
      throw std::runtime_error("Invalid operation: cannot call pop_back() on a stack of size 1.");
    return Stack(data->previous);
  }

  unsigned size() const {
    return data->size;
  }

  std::vector<T> values() const {
    std::vector<T> values;
    std::shared_ptr<Data> data(this->data);
    while (data) {
      values.push_back(data->value);
      data = data->previous;
    }
    std::reverse(values.begin(), values.end());
    return values;
  }

private:
  struct Data {
    Data(const T& value)
        : value(value), size(1) {}

    Data(const T& value, const Stack& previous)
        : value(value), previous(previous.data), size(previous.data->size + 1) {}

    const T value;
    const std::shared_ptr<Data> previous;
    const unsigned size;
  };

  Stack(const std::shared_ptr<Data>& data)
      : data(data) {}

  std::shared_ptr<Data> data;
};

#endif
