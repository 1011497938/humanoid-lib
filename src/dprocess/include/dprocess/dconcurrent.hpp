// Load all threads at first and run in async way
// Avoid overhead of std::async of always launching new threads
// Created on: May 2, 2017
//     Author: Wenxing Mei <mwx36mwx@gmail.com>
#pragma once
#include <array>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace dprocess {
static const int MAX_THREADS = 10;

class _Thread
{
  public:
    _Thread();
    void stop();
    void setf(std::function<void(void)> f);
    void spinOnce();
    void join();

  private:
    std::mutex lock_in;
    std::condition_variable cond_in;
    std::mutex lock_out;
    std::condition_variable cond_out;
    bool ready;
    bool done;
    bool terminate;
    std::function<void(void)> func;
    std::thread thread;
};

class DConcurrent
{
  public:
    DConcurrent();
    ~DConcurrent();
    void push(std::function<void(void)> f);
    // spin once
    void spinOnce();
    void join();

  private:
    // using array because std::vector needs destruct when expanding space, and it terminates the threads
    std::array<_Thread, MAX_THREADS> m_threads;
    int m_num;
};
}