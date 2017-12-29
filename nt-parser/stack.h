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
  Stack() {}

  Stack(const T& value)
      : data(std::make_shared<Data>(value)) {}

  Stack(const T& value, const Stack& previous)
      : data(std::make_shared<Data>(value, previous)) {}

  Stack(const std::vector<T>& values) {
    if (!values.empty()) {
      Stack stack(values.front());
      for (auto it = values.begin() + 1; it != values.end(); ++it)
        stack = Stack(*it, stack);
      data = stack.data;
    }
  }

  bool empty() const {
    return !((bool) data);
  }

  const T& back() const {
    if (empty()) {
      throw std::runtime_error("Invalid operation: cannot call back() on an empty stack.");
    }
    return data->value;
  }

  Stack push_back(const T& value) const {
    if (empty())
      return Stack(value);
    else
      return Stack(value, *this);
  }

  Stack pop_back() const {
    if (empty()) {
      throw std::runtime_error("Invalid operation: cannot call pop_back() on an empty stack.");
    }
    if (data->previous)
      return Stack(data->previous);
    else {
      // throw std::runtime_error("Invalid operation: cannot call pop_back() on a stack of size 1.");
      return Stack();
    }
  }

  unsigned size() const {
    if (data)
      return data->size;
    else
      return 0;
  }

  std::vector<T> values(int limit = -1) const {
    unsigned to_take = (limit >= 0) ? (unsigned) limit : size();
    std::vector<T> values;
    std::shared_ptr<Data> data(this->data);
    while (data && to_take > 0) {
      to_take--;
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
        : value(value), previous(previous.data), size(previous.size() + 1) {}

    const T value;
    const std::shared_ptr<Data> previous;
    const unsigned size;
  };

  Stack(const std::shared_ptr<Data>& data)
      : data(data) {}

  std::shared_ptr<Data> data;
};

#endif
