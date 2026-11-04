/** @file floating_point_epsilon.h */

#ifndef FLOATING_POINT_EPSILON_H_
#define FLOATING_POINT_EPSILON_H_

template <typename T>
struct fp {
    static constexpr T epsilon = 0;
    static constexpr T min = 0;
    static constexpr T max = INFINITY;
    static constexpr T lowest = -INFINITY;
};

template <>
struct fp<long double> {
    static constexpr long double epsilon = 1.0842e-19L;
    static constexpr long double min = 3.3621e-4932L;
    static constexpr long double lmin = -11355.1371L;
    static constexpr long double max = 1.1897e+4932L;
    static constexpr long double lowest = -1.1897e+4932L;
};

template <>
struct fp<double> {
    static constexpr double epsilon=2.22045e-16;
    static constexpr double min = 2.22507e-308;
    static constexpr double lmin = -708.39642;
    static constexpr double max = 1.79769e+308;
    static constexpr double lowest = -1.79769e+308;
};

template <>
struct fp<float> {
    static constexpr float epsilon=1.19209e-07f;
    static constexpr float min = 1.17549e-38f;
    static constexpr float lmin = -87.33654f;
    static constexpr float max = 3.40282e+38f;
    static constexpr float lowest = -3.40282e+38f;
};

#endif