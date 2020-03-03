#ifndef AVX_H
#define AVX_H

#include "../../plumbing/defs.h"
#include <immintrin.h>
#include "../../vectorclass/vectorclass.h"
#include "../../vectorclass/vectormath_exp.h"
#include "../../vectorclass/vectormath_trig.h"
#include "../../vectorclass/vectormath_hyp.h"

#include "../../plumbing/memory.h"

#define VECTORIZED


// Define random number generator
#define seed_random(seed) seed_mersenne(seed)
inline double hila_random(){ return mersenne(); }

// Trivial synchronization
inline void synchronize_threads(){}



/// Implements test for basic in types, similar to 
/// std::is_arithmetic, but allows the backend to add
/// it's own basic tyes (such as AVX vectors)
template< class T >
struct is_arithmetic : std::integral_constant<
  bool,
  std::is_arithmetic<T>::value ||
  std::is_same<T,Vec4d>::value ||
  std::is_same<T,Vec8f>::value ||
  std::is_same<T,Vec8i>::value ||
  std::is_same<T,Vec8d>::value ||
  std::is_same<T,Vec16f>::value ||
  std::is_same<T,Vec16i>::value 
> {};




/*** The next section contains basic operations for vectors ***/

// Norm squared
inline Vec4d norm_squared(Vec4d val){
  return val*val;
}

inline Vec8f norm_squared(Vec8f val){
  return val*val;
}

inline Vec8i norm_squared(Vec8i val){
  return val*val;
}

inline Vec8d norm_squared(Vec8d val){
  return val*val;
}

inline Vec16f norm_squared(Vec16f val){
  return val*val;
}

inline Vec16i norm_squared(Vec16i val){
  return val*val;
}

// Reductions
inline double reduce_sum(Vec4d v){
  double sum = 0;
  double store[4];
  v.store(&(store[0]));
  for(int i=0; i<4; i++)
    sum += store[i];
  return sum;
}

inline double reduce_sum(Vec8f v){
  double sum = 0;
  float store[8];
  v.store(&(store[0]));
  for(int i=0; i<8; i++)
    sum += store[i];
  return sum;
}

inline double reduce_sum(Vec8i v){
  double sum = 0;
  int store[8];
  v.store(&(store[0]));
  for(int i=0; i<8; i++)
    sum += store[i];
  return sum;
}

inline double reduce_sum(Vec8d v){
  double sum = 0;
  double store[8];
  v.store(&(store[0]));
  for(int i=0; i<8; i++)
    sum += store[i];
  return sum;
}

inline double reduce_sum(Vec16f v){
  double sum = 0;
  float store[16];
  v.store(&(store[0]));
  for(int i=0; i<16; i++)
    sum += store[i];
  return sum;
}

inline double reduce_sum(Vec16i v){
  double sum = 0;
  int store[16];
  v.store(&(store[0]));
  for(int i=0; i<16; i++)
    sum += store[i];
  return sum;
}

inline double reduce_prod(Vec4d v){
  double sum = 1;
  double store[4];
  v.store(&(store[0]));
  for(int i=0; i<4; i++)
    sum *= store[i];
  return sum;
}

inline double reduce_prod(Vec8f v){
  double sum = 0;
  float store[8];
  v.store(&(store[0]));
  for(int i=0; i<8; i++)
    sum *= store[i];
  return sum;
}

inline double reduce_prod(Vec8i v){
  double sum = 0;
  int store[8];
  v.store(&(store[0]));
  for(int i=0; i<8; i++)
    sum *= store[i];
  return sum;
}

inline double reduce_prod(Vec8d v){
  double sum = 1;
  double store[8];
  v.store(&(store[0]));
  for(int i=0; i<8; i++)
    sum *= store[i];
  return sum;
}

inline double reduce_prod(Vec16f v){
  double sum = 0;
  float store[16];
  v.store(&(store[0]));
  for(int i=0; i<16; i++)
    sum *= store[i];
  return sum;
}

inline double reduce_prod(Vec16i v){
  double sum = 0;
  int store[16];
  v.store(&(store[0]));
  for(int i=0; i<16; i++)
    sum *= store[i];
  return sum;
}


// Define modulo operator for integer vector
inline Vec16i operator%( const Vec16i &lhs, const int &rhs)
{
  Vec16i r;
  int tvec1[16], tvec2[16];
  lhs.store(&(tvec1[0]));
  for(int i=0; i<16; i++)
    tvec2[i] = tvec1[i] % rhs;
  r.load(&(tvec2[0]));
  return r;
}


inline Vec8i operator%( const Vec8i &lhs, const int &rhs)
{
  Vec8i r;
  int tvec1[8], tvec2[8];
  lhs.store(&(tvec1[0]));
  for(int i=0; i<8; i++)
    tvec2[i] = tvec1[i] % rhs;
  r.load(&(tvec2[0]));
  return r;
}

inline Vec4i operator%( const Vec4i &lhs, const int &rhs)
{
  Vec4i r;
  int tvec1[4], tvec2[4];
  lhs.store(&(tvec1[0]));
  for(int i=0; i<4; i++)
    tvec2[i] = tvec1[i] % rhs;
  r.load(&(tvec2[0]));
  return r;
}


// Random numbers
inline Vec4d hila_random_Vec4d(){
  Vec4d r;
  double tvec[4];
  for(int i=0; i<4; i++){
    tvec[i] = mersenne();
  }
  r.load(&(tvec[0]));
  return r;
}

inline Vec8f hila_random_Vec8f(){
  Vec8f r;
  float tvec[8];
  for(int i=0; i<8; i++){
    tvec[i] = mersenne();
  }
  r.load(&(tvec[0]));
  return r;
}


inline Vec8d hila_random_Vec8d(){
  Vec8d r;
  double tvec[8];
  for(int i=0; i<8; i++){
    tvec[i] = mersenne();
  }
  r.load(&(tvec[0]));
  return r;
}

inline Vec16f hila_random_Vec16f(){
  Vec16f r;
  float tvec[16];
  for(int i=0; i<16; i++){
    tvec[i] = mersenne();
  }
  r.load(&(tvec[0]));
  return r;
}








/// Utility for returning mapping a field element type into 
/// a corresponding vector. This is not used directly as a type
template <typename T>
struct field_info{
  constexpr static int vector_size = 1;
  constexpr static int base_type_size = 1;
  constexpr static int elements = 1;

  using base_type = double;
  using base_vector_type = Vec4d;
  using vector_type = Vec4d;
};





#endif