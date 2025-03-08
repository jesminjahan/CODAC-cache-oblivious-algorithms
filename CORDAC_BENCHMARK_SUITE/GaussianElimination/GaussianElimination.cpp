/*
LICENSING: Copyright (c) 2013, Jesmin Jahan Tithi and Dr. Rezaul Chowdhury.

LICENSING: Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

About this code:
A very naive implementation of recursive divide and conquer algorithm for Gaussian Elimination. It can be optimized to get 10 - 30x speedup.
icpc -o gauss Gauss_rec.cpp -O3 -Wall -Werror -ansi-alias -ip -opt-subscript-in-range  -restrict -vec-report -parallel -xhost
*
* */

#include<cilk/cilk.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include<iomanip>

using namespace std;
#ifndef BASECPU
	#define BASECPU 64
#endif

#ifndef SIZE
	#define SIZE 8192
#endif

#ifndef TYPE
	#define TYPE double
#endif

void GaussianD( TYPE *X , int xi , int xj , TYPE *U , int ui  ,int uj ,
	TYPE *V , int vi , int vj , TYPE *W , int wi  ,int wj ,int n )
{

	if( n  <= BASECPU ) 
	{
	 	for(int k = wi  ;k < wi + n   ;k++ )
		 for(int i = xi ;i < xi + n  ;i++ )
		 {
			for(int j = xj  ;j < xj +n ; j++ )
			{
                              
				if(k<i && k<j && i< SIZE -1 )
				{
         				*(X+ i*SIZE + j) =  *(X+ i*SIZE + j) - (TYPE) ( *(U+ i*SIZE + k)/ (*(W+ k*SIZE+ k))) * ( *(V+ k*SIZE + j));
				//	X[i][j] = X[i][j] - (TYPE) U[i][k]/W[k][k] * V[k][j];
				}
                             
			}
		}
		return;
	}

	//F( X11, U11, V11, W11);  
	cilk_spawn GaussianD( X,xi,xj, U, ui,uj , V,vi,vj, W,wi,wj ,n/2); 
	
	//F( X12, U11, V12, W11);  
	cilk_spawn GaussianD( X,xi, xj + n/2, U,ui,uj , V ,vi ,vj +n/2, W,wi, wj ,n/2 );  
	
	//F ( X21, U21, V11, W11);   
	cilk_spawn GaussianD( X, xi +n/2, xj    , U,ui +n/2, uj ,  V,vi, vj , W,wi, wj ,n/2); 

	//F ( X22, U21, V12, W11);      
	GaussianD( X, xi + n/2 , xj + n/2 , U,ui +n/2, uj, V ,vi ,vj + n/2 , W,wi,wj,n/2 );  
	

	cilk_sync;
	/////////////////////////////////////////////

          
	//F( X11, U12, V21, W22); 
	cilk_spawn GaussianD( X, xi , xj  , U, ui  , uj + n/2, V, vi + n/2 , vj  , W, wi + n/2 , wj + n/2 ,n/2); 
	
	//F( X12, U12, V22, W22);   
	cilk_spawn GaussianD( X, xi, xj+n/2, U, ui  , uj + n/2 ,  V,vi +n/2, vj+n/2, W, wi + n/2 , wj + n/2 ,n/2); 


	//F ( X21, U21, V21, W22)changed to U22;  
	cilk_spawn GaussianD( X, xi+n/2 ,xj , U,ui+n/2 ,uj+n/2,V, vi + n/2 , vj  ,  W, wi + n/2 , wj + n/2 ,n/2); 


	//F ( X22, U22, V22, W22)	
	 GaussianD( X,xi+n/2, xj+n/2 , U, ui+n/2 ,uj +n/2 ,  V,vi +n/2, vj+n/2,  W, wi + n/2 , wj + n/2 ,n/2); 

	cilk_sync;
	
	return ;
	
	
}


void GaussianC( TYPE *X , int xi , int xj , TYPE *U , int ui  ,int uj ,
	TYPE *V , int vi , int vj , TYPE *W , int wi  ,int wj ,int n )
{

	if( n  <= BASECPU ) 
	{
		for(int k = wi;k < wi + n   ;k++ )
		 for(int i = xi ;i < xi + n  ;i++ )
		 {
			for(int j = xj  ;j < xj +n ; j++ )
			{
				
			 	if(k<i && k<j && i< SIZE -1 )
				{
					
         				*(X+ i*SIZE + j) =  *(X+ i*SIZE + j) - (TYPE)  *(U+ i*SIZE + k)/ *(W+ k*SIZE+ k) *  *(V+ k*SIZE + j);
					//X[i][j] = X[i][j] - (TYPE) U[i][k]/W[k][k] * V[k][j];
				}
                        
			}
		}

		
		return;
	}



	//F( X11, U11, V11, W11);  
	cilk_spawn GaussianC( X,xi,xj, U, ui,uj , V,vi,vj, W,wi,wj ,n/2); 
	
	//F ( X21, U21, V11, W11);   
	 GaussianC( X, xi +n/2, xj    , U,ui +n/2, uj ,  V,vi, vj , W,wi, wj ,n/2); 

	cilk_sync ;

	//F( X12, U11, V12, W11);  
	cilk_spawn GaussianD( X,xi, xj + n/2, U,ui,uj , V ,vi ,vj +n/2, W,wi, wj ,n/2 );  
	
	//F ( X22, U21, V12, W11);      
	GaussianD( X, xi + n/2 , xj + n/2 , U,ui +n/2, uj, V ,vi ,vj + n/2 , W,wi,wj,n/2 );  
	
	cilk_sync ;

	/////////////////////////////////////////////

	
	//F ( X12, U12, V22, W22);  
	cilk_spawn GaussianC( X, xi ,xj +n/2, U,ui ,uj+n/2,V, vi + n/2 , vj + n/2 ,  W, wi + n/2 , wj + n/2 ,n/2); 

	//F( X22, U22, V22, W22); 
	 GaussianC( X, xi + n/2 , xj + n/2 , U, ui + n/2 , uj + n/2, V, vi + n/2 , vj + n/2 , W, wi + n/2 , wj + n/2 ,n/2); 
	
	cilk_sync ;

	//F( X21, U22, V21, W22);   
	cilk_spawn GaussianD( X, xi +n/2, xj, U, ui + n/2 , uj + n/2 ,  V,vi +n/2, vj, W, wi + n/2 , wj + n/2 ,n/2); 

	//F ( X11, U12, V21, W22)	
	GaussianD( X,xi, xj , U, ui ,uj +n/2 ,  V,vi +n/2, vj,  W, wi + n/2 , wj + n/2 ,n/2); 
	cilk_sync;
	
	return ;
}

void GaussianB( TYPE *X , int xi , int xj , TYPE *U , int ui  ,int uj ,
	TYPE *V , int vi , int vj , TYPE *W , int wi  ,int wj ,int n )
{

	if( n  <= BASECPU ) 
	{
	
		for(int k = wi ;k < wi + n   ;k++ )
		 for(int i = xi ;i < xi + n  ;i++ )
		 {
			for(int j = xj  ;j < xj +n ; j++ )
			{
				
				//if(k<i)
				if(k<i && k<j && i < SIZE -1)
				{
         				*(X+ i*SIZE + j) =  *(X+ i*SIZE + j) - (TYPE)  *(U+ i*SIZE + k)/ *(W+ k*SIZE+ k) *  *(V+ k*SIZE + j);
					//X[i][j] = X[i][j] - (TYPE) U[i][k]/W[k][k] * V[k][j];
				} 
			}
		}

		return;
	}


	//F( X11, U11, V11, W11);  
	cilk_spawn GaussianB( X,xi,xj, U, ui,uj , V,vi,vj, W,wi,wj ,n/2); 
	
	//F( X12, U11, V12, W11);  
	GaussianB( X,xi, xj + n/2, U,ui,uj , V ,vi ,vj +n/2, W,wi, wj ,n/2 );  
	
        cilk_sync ;

	//F ( X21, U21, V11, W11);   
	cilk_spawn GaussianD( X, xi +n/2, xj    , U,ui +n/2, uj ,  V,vi, vj , W,wi, wj ,n/2); 

	//F ( X22, U21, V12, W11);      
	GaussianD( X, xi + n/2 , xj + n/2 , U,ui +n/2, uj, V ,vi ,vj + n/2 , W,wi,wj,n/2 );  
	
	cilk_sync ;
	/////////////////////////////////////////////

	//F( X22, U22, V22, W22); 
	cilk_spawn GaussianB( X, xi + n/2 , xj + n/2 , U, ui + n/2 , uj + n/2, V, vi + n/2 , vj + n/2 , W, wi + n/2 , wj + n/2 ,n/2); 
	
	//F( X21, U22, V21, W22);   
	GaussianB( X, xi +n/2, xj, U, ui + n/2 , uj + n/2 ,  V,vi +n/2, vj, W, wi + n/2 , wj + n/2 ,n/2); 

	cilk_sync;

	//F ( X12, U12, V22, W22);  
	cilk_spawn GaussianD( X, xi ,xj +n/2, U,ui ,uj+n/2,V, vi + n/2 , vj + n/2 ,  W, wi + n/2 , wj + n/2 ,n/2); 


	//F ( X11, U12, V21, W22)	
	GaussianD( X,xi, xj , U, ui ,uj +n/2 ,  V,vi +n/2, vj,  W, wi + n/2 , wj + n/2 ,n/2); 

	cilk_sync ;
	return ;
	
	
}

void GaussElimination( TYPE *X , int xi , int xj , TYPE *U , int ui  ,int uj ,
	TYPE *V , int vi , int vj , TYPE *W , int wi  ,int wj ,int n )
{

	if( n  <= BASECPU ) 
	{
	
		for(int k = wi ;k < wi + n   ;k++ )
		 for(int i = xi ;i < xi + n  ;i++ )
		 {
			for(int j = xj  ;j < xj +n ; j++ )
			{
				
				if(k<i && k<j && i < SIZE -1 )
				{
					
         				*(X+ i*SIZE + j) =  *(X+ i*SIZE + j) - (TYPE)  *(U+ i*SIZE + k)/ *(W+ k*SIZE+ k) *  *(V+ k*SIZE + j);
					//X[i][j] = X[i][j] - (TYPE) U[i][k]/W[k][k] * V[k][j];
				}
			}
		}

		return;
	}


	//F( X11, U11, V11, W11);  
	GaussElimination ( X,xi,xj, U, ui,uj , V,vi,vj, W,wi,wj ,n/2); 
	
	//F( X12, U11, V12, W11);  
	cilk_spawn GaussianB ( X,xi, xj + n/2, U,ui,uj , V ,vi ,vj +n/2, W,wi, wj ,n/2 );  
	
	//F ( X21, U21, V11, W11);   
	 GaussianC ( X, xi +n/2, xj    , U,ui +n/2, uj ,  V,vi, vj , W,wi, wj ,n/2); 

	cilk_sync;


	//F ( X22, U21, V12, W11);      
	 GaussianD ( X, xi + n/2 , xj + n/2 , U,ui +n/2, uj, V ,vi ,vj + n/2 , W,wi,wj,n/2 );  
	

	/////////////////////////////////////////////

	//F( X22, U22, V22, W22); 
       	GaussElimination ( X, xi + n/2 , xj + n/2 , U, ui + n/2 , uj + n/2, V, vi + n/2 , vj + n/2 , W, wi + n/2 , wj + n/2 ,n/2); 
	
	//F( X21, U22, V21, W22);   
         cilk_spawn	GaussianB ( X, xi +n/2, xj, U, ui + n/2 , uj + n/2 ,  V,vi +n/2, vj, W, wi + n/2 , wj + n/2 ,n/2); 

        
	//F ( X12, U12, V22, W22);  
	 GaussianC ( X, xi ,xj +n/2, U,ui ,uj+n/2,V, vi + n/2 , vj + n/2 ,  W, wi + n/2 , wj + n/2 ,n/2); 

	cilk_sync;
      
	//F ( X11, U12, V21, W22)	
         GaussianD( X,xi, xj , U, ui ,uj +n/2 ,  V,vi +n/2, vj,  W, wi + n/2 , wj + n/2 ,n/2); 


	return ;
	
	
}

int main(int argc , char  *argv[] )
{

	int size ;	
	//cin>>size; 
	
	    
	//SIZE = size;
	size=SIZE;
	TYPE * input;
	TYPE value;

	// allocation
	input = (TYPE*)_mm_malloc(size*size*sizeof(TYPE), 64);;

        for(int i = 0 ; i < size ; i++ )
        {
          for(int j = 0 ; j< size; j++ )
           {
		        //cin>>value;
			     value=i+j+10;
                *(input + i* size + j )= value  ;
                }
        }
		
	
	GaussElimination(input,0,0,input,0,0,input,0,0,input,0,0,size);
	
	
	 for(int i = 0 ; i < size ; i++ )
        {
	  cout << endl;
          for(int j = 0 ; j< size; j++ )
           {
               value =  *(input + i* size + j )  ;
			cout << setw (10);
	               cout<<setprecision (2)<<fixed<<showpoint << value;
		 }
        }
        cout<<endl;
	
	//std::cout<<"\n" << "Time Taken in Seconds:" << cv.accumulated_milliseconds() / 1000.f << " seconds" << "\n" ;
	return 0;	
}
