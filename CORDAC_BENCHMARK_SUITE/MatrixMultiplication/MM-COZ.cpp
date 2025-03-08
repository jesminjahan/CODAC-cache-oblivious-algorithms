#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <iostream>
#include <pthread.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <math.h>

using namespace std;

#ifndef TYPE
   #define TYPE float
#endif

#ifndef maxN
   #define maxN 16384
#endif

#ifndef BASECPU
   #define BASECPU 64
#endif

int N, B;
TYPE *X, *U, *V; 
TYPE *XO, *UO, *VO;

#ifdef debug
TYPE A[maxN*maxN], BB[maxN*maxN], C[maxN*maxN];

void MatrixMultiplication(){
	//A[:]=0;
	for(int i=0;i<maxN; i++)
	{
		for(int j=0;j<maxN;j++)
		{   A[i*maxN+j]=0;
			for(int k=0;k<maxN;k++)
			{
				A[i*maxN+j]=A[i*maxN+j]+(BB[i*maxN+k]*C[k*maxN+j]);
			
			}
		
		}
	}

}
#endif

int getMilliCount( void )
{
   timeb tb;
   
   ftime( &tb );
   
   int nCount = tb.millitm + ( tb.time & 0xfffff ) * 1000;
   
   return nCount;
}

int getMilliSpan( int nTimeStart )
{
   int nSpan = getMilliCount( ) - nTimeStart;
   
   if ( nSpan < 0 ) nSpan += 0x100000 * 1000;
   
   return nSpan;
}



void conv_RM_2_ZM_RM( TYPE *x, TYPE *xo, int n, int no )
{
   if ( n <= B )
     {
       for ( int i = 0; i < n; i++ )
         {
           for ( int j = 0; j < n; j++ )
              ( *x++ ) = ( *xo++ );

           xo += ( no - n );
         }
     }
   else
     {
       int nn = ( n >> 1 );
	   int nn2 = nn * nn;

       const int m11 = 0;
       int m12 = m11 + nn2;
       int m21 = m12 + nn2;
       int m22 = m21 + nn2;

       conv_RM_2_ZM_RM( x, xo, nn, no );
       conv_RM_2_ZM_RM( x+m12, xo + nn, nn, no );
       conv_RM_2_ZM_RM( x+m21, xo + n * nn, nn, no );
       conv_RM_2_ZM_RM( x+m22, xo + n * nn + nn, nn, no );

     }
}


/*
   ZM_RM : hybrid Z-Morton row-major
   RM: row-major
*/

void conv_ZM_RM_2_RM( TYPE *x, TYPE *xo, int n, int no )
{
   if ( n <= B )
     {
       for ( int i = 0; i < n; i++ )
         {
           for ( int j = 0; j < n; j++ )
              ( *xo++ ) = ( *x++ );

           xo += ( no - n );
         }
     }
   else
     {
       int nn = ( n >> 1 );
       int nn2 = nn * nn;
	   const int m11 = 0;
       int m12 = m11 + nn2;
       int m21 = m12 + nn2;
       int m22 = m21 + nn2;

       conv_ZM_RM_2_RM( x, xo, nn, no );
       conv_ZM_RM_2_RM( x+ m12, xo + nn, nn, no );
       conv_ZM_RM_2_RM( x+ m21, xo + n * nn, nn, no );
       conv_ZM_RM_2_RM( x+ m22, xo + n * nn + nn, nn, no );

     }
}


/*
   RM: row-major
   ZM_CM : hybrid Z-Morton column-major
*/

void conv_RM_2_ZM_CM( TYPE *x, TYPE *xo, int n, int no )
{
   if ( n <= B )
     {
       for ( int i = 0; i < n; i++ )
         {
           TYPE *xx = x + i;

           for ( int j = 0; j < n; j++ )
             {
               ( *xx ) = ( *xo++ );
               xx += n;
             }


           xo += ( no - n );
         }
     }
   else
     {
       int nn = ( n >> 1 );
	   int nn2 = nn * nn;
	   const int m11 = 0;
       int m12 = m11 + nn2;
       int m21 = m12 + nn2;
       int m22 = m21 + nn2;

       conv_RM_2_ZM_CM( x, xo, nn, no );
       conv_RM_2_ZM_CM( x+ m12, xo + nn, nn, no );
       conv_RM_2_ZM_CM( x+ m21, xo + n * nn, nn, no );
       conv_RM_2_ZM_CM( x+ m22, xo + n * nn + nn, nn, no );
     }
}
void funcD( TYPE *x, TYPE *u, TYPE *v, int n )
{
   if ( n <= B )
     {
       TYPE *xx = x;     
       TYPE *uu = u;
#pragma parallel      
       for ( int i = 0; i < n; i++ )
         {
           TYPE *vv = v;
                      
#pragma parallel         
           for ( int j = 0; j < n; j++ )
             {
               TYPE t = 0;
#pragma ivdep
#pragma vector always
#pragma vector aligned
               for ( int k = 0; k < n; k++ )
                  t += uu[ k ] * vv[ k ];
                  
               ( *xx++ ) += t;   
               vv += n;                 
             }
             
           uu += n;
         }  
     }
   else
     {
       int nn = ( n >> 1 );
       int nn2 = nn * nn;
       
       const int m11 = 0;
       int m12 = m11 + nn2;
       int m21 = m12 + nn2;
       int m22 = m21 + nn2;
       
       cilk_spawn funcD( x + m11, u + m11, v + m11, nn );
       cilk_spawn funcD( x + m12, u + m11, v + m12, nn );
       cilk_spawn funcD( x + m21, u + m21, v + m11, nn );
                  funcD( x + m22, u + m21, v + m12, nn );

       cilk_sync;

       cilk_spawn funcD( x + m11, u + m12, v + m21, nn );
       cilk_spawn funcD( x + m12, u + m12, v + m22, nn );
       cilk_spawn funcD( x + m21, u + m22, v + m21, nn );
                  funcD( x + m22, u + m22, v + m22, nn );       

       cilk_sync;
     }  
}


int main ( int argc , char *argv[ ] )
{
   N = 0;
   if ( argc > 1 ) N = atoi( argv[ 1 ] );   
   if ( N <= 0 ) N = maxN;
   
   int k = 0;
   
   while ( N )
     {
       N >>= 1;
       k++;
     }

   if ( k > 1 ) N = ( 1 << ( k - 1 ) );
   else N = 1; 

   B = 0;
   if ( argc > 2 ) B = atoi( argv[ 2 ] );   
   if ( B <= 0 ) B = BASECPU;

   k = 0;
   
   while ( B )
     {
       B >>= 1;
       k++;
     }

   if ( k > 1 ) B = ( 1 << ( k - 1 ) );
   else B = 1; 

   if ( B > N ) B = N;
   
   printf( "N = %d, B = %d\n", N, B );	

   X = ( TYPE * ) _mm_malloc( N * N * sizeof( TYPE ), 64 );
   UO = ( TYPE * ) _mm_malloc( N * N * sizeof( TYPE ), 64 );
   VO = ( TYPE * ) _mm_malloc( N * N * sizeof( TYPE ), 64 );

  
   if ( ( X == NULL ) || ( UO == NULL ) || ( VO == NULL ) )   
     {
       printf( "\nError: Allocation failed!\n\n" );
       
       if ( X != NULL ) _mm_free( X ); 
       if ( UO != NULL ) _mm_free( U );
       if ( VO != NULL ) _mm_free( VO ); 
       
       exit( 1 );
     }
     
   srand( time( NULL ) );  
     
   for ( int i = 0; i < N * N; i++ )
      {
        X[ i ] = 0;
        UO[ i ] = ( rand( ) % 5000 );
        VO[ i ] = ( rand( ) % 5000 );
#ifdef debug
		BB[i]=UO[ i ];
		C[i]=VO[ i ];
#endif
      }
     
   U = ( TYPE * ) _mm_malloc( N * N * sizeof( TYPE ), 64 );
   if( U == NULL ) 
   {  
	   printf( "\nError: Allocation failed!\n\n" );
       if ( U != NULL ) _mm_free( U ); 
   }
   
   conv_RM_2_ZM_RM( U, UO,  N, N );
   if ( UO != NULL ) _mm_free( UO ); 



   V = ( TYPE * ) _mm_malloc( N * N * sizeof( TYPE ), 64 );
   if( V == NULL )
   {
	   printf( "\nError: Allocation failed!\n\n" );
       if ( V != NULL ) _mm_free( V ); 
   }
   conv_RM_2_ZM_CM( V, VO,  N, N );
   if ( VO != NULL ) _mm_free( VO ); 
         
   float tstart = getMilliCount( ); 
   
   funcD ( X, U, V, N );
     
   float tend = getMilliSpan( tstart );
   XO = ( TYPE * ) _mm_malloc( N * N * sizeof( TYPE ), 64 );
  
   if(XO == NULL ) 
   {      printf( "\nError: Allocation failed!\n\n" );
          if ( XO != NULL ) _mm_free( XO ); 
   
   }
   XO[0:N*N]=0;
   conv_ZM_RM_2_RM( X, XO,  N, N );

#ifdef debug
MatrixMultiplication();
for(int i=0;i<maxN;i++)
{
	for(int j=0;j<maxN;j++)
	{
		if(A[i*maxN+j]!=XO[i*maxN+j])
		{
			cout<<"Wrong at:"<<i<<","<<j<<endl;		
			break;
		
		}
	
	}

}
#endif

    

   printf( "p = %d, T_p = %lf ( ms )\n\n", __cilkrts_get_nworkers( ), tend );

   _mm_free( X ); 
   
   _mm_free( V );

   _mm_free( U); 

   _mm_free( XO ); 


   return 0;	
}
