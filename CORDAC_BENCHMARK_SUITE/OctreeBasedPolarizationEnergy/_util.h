/*
@author: Coded by Jesmin Jahan Tithi, jtithi@cs.stonybrook.edu
With the help of codes provided by Professor Rezaul Chowdhury

*/

#ifndef UTIL_H
#define UTIL_H


#include<iostream>
#include<fstream>
#include<istream>
#include<string>
#include<math.h>
#include<iomanip>
#include<limits>
#include<cstdio>
#include<cstdlib>
#include<string.h>
#include<malloc.h>
#include<sys/timeb.h>
#include<time.h>
#include<sys/types.h>
using namespace std;

//structure for holding quardrature points


/*Necessary Utility Functions*/
double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks)/(CLOCKS_PER_SEC);
	return diffms;
}
#ifdef zeroIfLess   
   #undef zeroIfLess
#endif
#define zeroIfLess( a, b ) ( ( ( a ) < ( b ) ) ? 0 : 1 )

#define transform( ox, oy, oz, M, nx, ny, nz ) {                                                                      \
                                                 nx = ( ox ) * M[ 0 ] + ( oy ) * M[ 1 ] + ( oz ) * M[  2 ] + M[  3 ]; \
                                                 ny = ( ox ) * M[ 4 ] + ( oy ) * M[ 5 ] + ( oz ) * M[  6 ] + M[  7 ]; \
                                                 nz = ( ox ) * M[ 8 ] + ( oy ) * M[ 9 ] + ( oz ) * M[ 10 ] + M[ 11 ]; \
                                               }

//#ifndef M_PI
//   #define M_PI 3.1415926535897932384626433832795
//#endif
//
//#ifndef INV_SQRT_TWO
//   #define INV_SQRT_TWO 0.70710678118654752440084436210485
//#endif
//
//#define Inv_4PI 1/(4*M_PI)
int skipWhiteSpaces( char *buf, int i )
{
   int j = i;
   
   while ( buf[ j ] && isspace( buf[ j ] ) ) j++;
   
   return j;
}

int getInt( char *buf, int i, int *v )
{ 
   char s[ 100 ];
   int j = skipWhiteSpaces( buf, i );
   
   int k = 0;
   
   if ( buf[ j ] == '-' ) s[ k++ ] = buf[ j++ ];
   
   while ( buf[ j ] && ( isdigit( buf[ j ] ) ) ) s[ k++ ] = buf[ j++ ];
   
   s[ k ] = 0;
   
   *v = atoi( s );
   
   return j;
}
int getAlphaString( char *buf, int i, char *s )
{ 
   int j = skipWhiteSpaces( buf, i );
   
   int k = 0;
   
   while ( buf[ j ] && isalpha( buf[ j ] ) ) s[ k++ ] = buf[ j++ ];
   
   s[ k ] = 0;
   
   return j;
}


int getString( char *buf, int i, char *s )
{ 
   int j = skipWhiteSpaces( buf, i );
   
   int k = 0;
   
   while ( buf[ j ] && ( isalnum( buf[ j ] ) || ispunct( buf[ j ] ) ) ) s[ k++ ] = buf[ j++ ];
   
   s[ k ] = 0;
   
   return j;
}


int getDouble( char *buf, int i, double *v )
{ 
   char s[ 100 ];
   int j = skipWhiteSpaces( buf, i );
   
   int k = 0;
   
   if ( buf[ j ] == '-' ) s[ k++ ] = buf[ j++ ];
   
   while ( buf[ j ] && ( isdigit( buf[ j ] ) || ( buf[ j ] == '.' ) ) ) s[ k++ ] = buf[ j++ ];
   
   s[ k ] = 0;
   
   *v = atof( s );
   
   return j;
}


#endif
