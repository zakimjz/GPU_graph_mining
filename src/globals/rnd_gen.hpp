#ifndef __RND_GEN_HPP__
#define __RND_GEN_HPP__

class rnd_generator {
public:
  virtual double        generate_uniform(double min, double max) = 0;
  virtual unsigned long generate_uniform(unsigned long max) = 0;
  virtual unsigned int  generate_poisson(double avg) = 0;
};


#endif
