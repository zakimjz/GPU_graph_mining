#include <rnd_gen.hpp>
#include <gsl_rnd_gen.hpp>
#include <sys/time.h>

//new GSL_rnd_generator();

gsl_rng *GSL_rnd_generator::rng = 0;

GSL_rnd_generator::GSL_rnd_generator(int seed)
{
  if(rng == 0) {
    rng = gsl_rng_alloc(gsl_rng_mt19937); // taus !?!?!?!?!
    gsl_rng_set(rng, seed);
  }

}

GSL_rnd_generator::GSL_rnd_generator()
{
  if(rng == 0) {
    timeval tv;
    gettimeofday(&tv, 0);
    int seed = tv.tv_usec;
    rng = gsl_rng_alloc(gsl_rng_mt19937); // taus !?!?!?!?!
    gsl_rng_set(rng, seed);
  }

}

double  GSL_rnd_generator::generate_uniform(double min, double max)
{
  return gsl_ran_flat(rng, min, max);
}


unsigned long  GSL_rnd_generator::generate_uniform(unsigned long max)
{
  return gsl_rng_uniform_int(rng, max);
}

unsigned int  GSL_rnd_generator::generate_poisson(double avg)
{
  return gsl_ran_poisson(rng, avg);
}
