#include "MemCorrupter.h"
#include <random>

void MemCorrupter::corrupt() {
// Setup new random distribution
  char* p = (char*)begin;
  int length = (char*)end - p;
  std::random_device seeder;
  std::default_random_engine r(seeder());

  std::poisson_distribution<> bits_to_flip(8 * length * prob);
  std::uniform_int_distribution<> which_bit(0, 8 * length - 1);

  counter=0;

  while( state != TO_TERMINATE ){
    std::unique_lock<std::mutex> lk(m);                                     
    while (state==TO_STOP || state == STOPPED) {
      if ( state == TO_STOP ) state = STOPPED;
      notify_back();
      cv.wait(lk);
    }
    lk.unlock();

    lk.lock();
    if ( state == TO_RUN ){
//      state = set_state( RUNNING );
      state =  RUNNING ;
      notify_back();
    }
    lk.unlock();

    while ( state == RUNNING ) {
      int bits = bits_to_flip(r);
/* if ( bits != 0)
        printf("%d.", bits); 
*/

      for(int b = 0; b < bits; ++b)
      {
        int bit_index = which_bit(r);
        __sync_fetch_and_xor(p + bit_index / 8, 1 << (bit_index % 8));
        counter++;
      }
    } // RUNNING
  } // !TO_TERMINATE
  set_state(TERMINATED);
  notify_back();
}
