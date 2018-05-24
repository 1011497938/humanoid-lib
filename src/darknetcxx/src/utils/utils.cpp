#include "darknetcxx/utils.hpp"
#include "darknetcxx/blas.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

namespace darknet {
#ifdef DARKNET_MALLOCATED
template<>
double Ptr<float>::m_allocated = 0;
template<>
double Ptr<int>::m_allocated = 0;
template<>
double Ptr<unsigned char>::m_allocated = 0;
#endif

NonCopyable::NonCopyable() = default;

void
error(const char* s)
{
    perror(s);
    assert(0);
    exit(-1);
}

float
sec(clock_t clocks)
{
    return static_cast<float>(clocks) / CLOCKS_PER_SEC;
}

void
random_init(float* ptr, size_t size, float scale)
{
    for (unsigned int i = 0; i < size; ++i) {
        ptr[i] = scale * rand_uniform(-1, 1);
    }
}

float
sum_array(float* a, int n)
{
    float sum = 0;
    for (int i = 0; i < n; ++i)
        sum += a[i];
    return sum;
}

float
mag_array(float* a, int n)
{
    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

std::string
fix_path(const std::string& filepath)
{
    // std::string upper_path = ;
    if (std::ifstream(filepath)) {
        return filepath;
    } else if (std::ifstream("../" + filepath)) {
        return "../" + filepath;
    } else {
        return fix_path("../" + filepath);
    }
}

int
max_index(float* first, float* last)
{
    return std::distance(first, std::max_element(first, last));
}

void
log_layer_output(float* output, const int& size, const int& idx)
{
    std::string filename = "output_layer_" + std::to_string(idx) + ".log";
    FILE* fout = fopen(filename.c_str(), "w");
    for (int i = 0; i < size; i++) {
        fprintf(fout, "%5f\n", output[i]);
    }
    fprintf(fout, "\n");
    fclose(fout);
    printf("write to log, lines: %d\n", size);
}

void
free_ptrs(void** ptrs, int n)
{
    for (int i = 0; i < n; ++i)
        free(ptrs[i]);
    free(ptrs);
}
} // namespace darknet
