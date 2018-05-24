#pragma once
#include "darknetcxx/blas.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace darknet {
void
error(const char* s);
float
sec(clock_t clocks);

template<typename T>
void
make_normaldist_array(T* x, int n, T mean)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> d(mean, 2);
    std::for_each(x, x + n, [&](T& val) { val = d(gen); });
}

// TODO initialize seed
inline float
rand_uniform(float min, float max)
{
    if (max < min)
        return rand_uniform(max, min);
    return (static_cast<float>(rand()) / RAND_MAX * (max - min)) + min;
}

float
sum_array(float* a, int n);

void
random_init(float* ptr, size_t size, float scale);

/*******************************************************************************
 * wrapper for cpu raw pointers
 ******************************************************************************/

class NonCopyable
{
  public:
    NonCopyable();
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable(NonCopyable&&) = delete;
    NonCopyable& operator=(NonCopyable&&) = delete;
};

template<typename T>
class Ptr : NonCopyable
{
  public:
    struct deleter
    {
        void operator()(T* p)
        {
            if (p) {
                delete[] p;
            }
        }
    };

    Ptr()
      : m_size(0)
      , m_ptr(nullptr, deleter())
    {
    }

    explicit Ptr(size_t size)
      : m_size(size)
      , m_ptr(new T[size], deleter())
    {
#ifdef DARKNET_MALLOCATED
        m_allocated += sizeof(T) * size / 1000000.;
// fprintf(stderr, "CPU init %lu bytes, all %lf MB\n", sizeof(T) * size, m_allocated);
#endif
        // std::cout << "Ptr_cpu ctor" << std::endl;
    }

    void resize(size_t size)
    {
#ifdef DARKNET_MALLOCATED
        m_allocated -= sizeof(T) * m_size;
        m_allocated += sizeof(T) * size;
// printf("CPU resized to %lu bytes, all %lf MB\n", sizeof(T) * size,
//        m_allocated);
#endif
        auto p = m_ptr.release();
        if (p)
            delete[] p;
        m_size = size;
        m_ptr.reset(new T[size]);
        // std::cout << "ptr cpu resized to " << size << std::endl;
    }

    T* get()
    {
        return m_ptr.get();
    }

    T& operator[](size_t i)
    {
        return m_ptr[i];
    }

    size_t size()
    {
        return m_size;
    }

    void fill(T x)
    {
        fill_cpu(size(), x, get());
    }

#ifdef DARKNET_MALLOCATED
    static double m_allocated;
#endif

  private:
    size_t m_size;
    std::unique_ptr<T[], deleter> m_ptr;
};

/*********
 * Utils *
 *********/

float
mag_array(float* a, int n);

std::string
fix_path(const std::string& filepath);

int
max_index(float* first, float* last);

void
log_layer_output(float* output, const int& size, const int& idx);

void
free_ptrs(void** ptrs, int n);

class InputParser
{
  public:
    InputParser(const int& argc, const char** argv)
    {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    /// @author iain
    const std::string& getCmdOption(const std::string& option) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }
    /// @author iain
    bool cmdOptionExists(const std::string& option) const
    {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

  private:
    std::vector<std::string> tokens;
};
} // namespace darknet
