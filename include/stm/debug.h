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

#endif /* stm_debug_h */