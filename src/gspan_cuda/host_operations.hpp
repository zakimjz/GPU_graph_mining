#ifndef __HOST_OPERATIONS_HPP__
#define __HOST_OPERATIONS_HPP__

template<class __transform_operator__, class __associative_operator__, class init_t>
init_t host_transform_reduce(int from, int to, __transform_operator__ transform_op, init_t init, __associative_operator__ assoc_op)
{
  //init_t val = init;
  //int i = from;
  init_t result = init;

  for(int i = from + 1; i < to; i++) {
    result = assoc_op(result, transform_op(i));
  } // for i

  return result;
}


template<class __operator__>
void host_for_each(int from, int to, __operator__ op)
{
  for(int i = from; i < to; i++) {
    op(i);
  }
}

template<class value_type, class __operator__>
void host_exclusive_scan(value_type *from, value_type *to, value_type *result, value_type init, __operator__ op)
{
  if(from == to) return;

  *result = init;
  result++;
  for(value_type *it = from + 1; it != to; it++) {
    *result = op(*(result - 1), *it);
  } // for it
} // host_exclusive_scan


#endif


