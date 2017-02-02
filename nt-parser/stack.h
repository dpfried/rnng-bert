#ifndef STACK_H
#define STACK_H

#include <algorithm>
#include <assert.h>
#include <memory>
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
    return Stack(data->previous);
  }

  unsigned size() const {
    return data->size;
  }

  std::vector<T> values() const {
    std::vector<T> values;
    Stack stack(*this);
    while (stack.data) {
      values.push_back(stack.back());
      stack = stack.pop_back();
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
