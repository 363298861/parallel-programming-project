#ifndef __MEM_CORRUPTOR_H__
#define __MEM_CORRUPTOR_H__
#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <condition_variable>

class MemCorrupter{
public:
    static MemCorrupter* getInstance(){
      static MemCorrupter *mc;
      if (mc==NULL) mc = new MemCorrupter;
      return mc;
    }

    unsigned long long exit(){
      kill();
//      unsigned long long ret = this->counter;
      //delete this;
      return counter;
    }

protected:
    MemCorrupter() { 
        counter = 0;
        state = TO_STOP; 
        t = std::thread( &MemCorrupter::corrupt, this );
    }

public:
    ~MemCorrupter() { if (state != TERMINATED) kill(); }

    void startCorrupting(void *begin, void *end, double prob){
        this->begin=begin; 
        this->end=end; 
        this->prob=prob;
        start();
    }

    unsigned long long stopCorrupting(){
        stop();
        return counter;
    };

    void corrupt();

protected:
    // helpers for thread control
    void start(){ 
        change_state(TO_RUN);
    }

    void stop() { 
        change_state(TO_STOP);
    }

    void kill() { 
      change_state(TO_TERMINATE);

      if (t.joinable()) t.join(); 
    }


private:
    std::mutex m;
    std::mutex ms;
    std::thread t;
    std::condition_variable cv;
    std::condition_variable cvs;

    volatile unsigned int state;
    static const unsigned int TO_STOP       = 0;
    static const unsigned int STOPPED       = 1;
    static const unsigned int TO_RUN        = 2;
    static const unsigned int RUNNING       = 3;
    static const unsigned int TO_TERMINATE  = 4;
    static const unsigned int TERMINATED    = 5;

    volatile void *begin, *end;
    volatile double prob;
    volatile unsigned long long counter;

    void set_state( unsigned int new_state){
        std::lock_guard<std::mutex> lk(m);
        state = new_state;
    }

    void change_state( unsigned int new_state){
        set_state( new_state);
        cv.notify_one();

        std::unique_lock<std::mutex> lks(ms);
        while (state != new_state+1){
            cvs.wait(lks);
        }
    }

    void notify_back(){
        std::unique_lock<std::mutex> lks(ms);
        cvs.notify_one();
    }


};
#endif //__MEM_CORRUPTOR_H__
/*
void MemoryCorrupter::corrupt() {
  std::mutex m;
  std::unique_lock<std::mutex> lk(m);

  while( state != TERMINATE ){
    while ( state == STOPPED ) { 
      //st.notify_one();
        cv.wait(lk); 
    }// STOPPED 
    
    // Setup new random distribution
    char* p = (char*)begin;
    int length = (char*)end - p;
    std::random_device seeder;
    std::default_random_engine r(seeder());

    std::poisson_distribution<> bits_to_flip(8 * length * prob);
    std::uniform_int_distribution<> which_bit(0, 8 * length - 1);

    counter=0;
    while ( state == RUNNING ) {
      int bits = bits_to_flip(r);
      for(int b = 0; b < bits; ++b)
      {
        int bit_index = which_bit(r);
        __sync_fetch_and_xor(p + bit_index / 8, 1 << (bit_index % 8));
      }
      counter++;
    } // RUNNING
  } // !TERMINATE
}

int main(){
    MemoryCorrupter *h = MemoryCorrupter::getInstance();
    double *u = new double[1024*1024];

    std::this_thread::sleep_for(std::chrono::seconds(1));

    h->startCorrupting((void*)u, (void*)&u[1024*1024-1], 0.0001);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    long x = h->stopCorrupting();
    std::cout<< "Counter: " << x << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    h->startCorrupting((void*)u, (void*)&u[1024*1024-1], 0.0001);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    x = h->stopCorrupting();

    std::cout<< "Counter: " << x << std::endl;

    return 0;
}
*/
