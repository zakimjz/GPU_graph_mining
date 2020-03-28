#ifndef __GSL_RNG_GEN_HPP__
#define __GSL_RNG_GEN_HPP__

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <rnd_gen.hpp>


class GSL_rnd_generator : public rnd_generator {
  static gsl_rng * rng;
public:
  GSL_rnd_generator(int seed);
  GSL_rnd_generator();
  virtual double  generate_uniform(double min, double max);
  virtual unsigned long generate_uniform(unsigned long max);
  virtual unsigned int  generate_poisson(double avg);
};


#endif
