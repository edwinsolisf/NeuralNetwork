#ifndef stm_debug_h
#define stm_debug_h

/*
* DEBUG TOOLS
* Setting up debug macros
* 
* Add "DEBUG" to debug configuration
*/

#ifdef DEBUG

	#ifdef _MSC_VER
		#define debug_break __debugbreak
	#else
		#if __has_builtin(__builtin_debugtrap)
			#define debug_break __builtin_debugtrap
		#endif
	#endif

	#define stm_assert(x) if(!(x)) { debug_break(); }

#else

	#define stm_assert(x) 

#endif /* DEBUG */

#define TEST(x) { long long start = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
for (unsigned int i = 0; i < 10000; ++i) { x; }\
long long end = std::chrono::high_resolution_clock::now().time_since_epoch().count();\
std::cout << (end - start)/10000 << std::endl; }

#endif /* stm_debug_h */