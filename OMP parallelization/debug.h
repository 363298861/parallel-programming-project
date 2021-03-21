#ifndef __DEBUG_H__
#define __DEBUG_H__
#include <stdio.h>

#ifdef _BDEBUG_
#define debug_out(fmt,arg...) fprintf(stderr,fmt,arg)
#else
#define debug_out(fmt,arg...) ((void)0)
#endif

#endif //__DEBUG_H__
