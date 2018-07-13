#ifndef NDEBUG

    #ifdef _MSC_VER
        #include <intrin.h>
    #endif

    #ifdef _MSC_VER
        #define ASSERT(expr) \
            if (expr) { } \
            else \
            { \
                __debugbreak(); \
            }
    #elif __APPLE__
        #define ASSERT(expr) \
            if (expr) { } \
            else \
            { \
                __asm__("int $3"); \
            }
    #else
        #define ASSERT(expr)
    #endif

#else
    #define ASSERT(expr)
#endif // NDEBUG
