#pragma once

/*
* DEBUG TOOLS
* Setting up debug macros
* 
* Add "DEBUG" to debug configuration
*/

#ifdef DEBUG
#define assert(x) if(!(x)) { }
#else
#define assert(x) 
#endif