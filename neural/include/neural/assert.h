#ifndef NDEBUG

	#include <intrin.h>
	
	#define ASSERT(expr) \
		if (expr) { } \
		else \
		{ \
			__debugbreak(); \
		}
#else
	#define ASSERT(expr)
#endif
