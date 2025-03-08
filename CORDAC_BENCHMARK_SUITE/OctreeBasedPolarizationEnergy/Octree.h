#include"_util.h"
using namespace std;
#ifndef M_PI
   #define M_PI 3.1415926535897932384626433832795
#endif

#ifndef INV_SQRT_TWO
   #define INV_SQRT_TWO 0.70710678118654752440084436210485
#endif

#define Inv_4PI 0.07957747154 //1/(4*M_PI)
#define real float

typedef struct quadPoint{
 real rx,ry,rz;
 real nx,ny,nz;
 real w;
 
}QPoint;

typedef struct{
	
	int id; //i of the atom
    real x, y, z, q, r,ra; //
	
} Atom;
#ifdef freeMem
   #undef freeMem
#endif
#define freeMem( ptr ) { if ( ptr != NULL ) free( ptr ); }
typedef union 
     {
       unsigned long ix;
       float fx;
     } UL_F_union;
        
// N=0, C=1, O=2, H=3, S=4, P=5  

typedef struct on{
     real cx, cy, cz;				//center of the circle bounding the node
     real cr;						//radious of the bounding circle
	 real nxq,nyq,nzq;			//these are sum of normals for the quad points
     bool leaf;						//tells whether the node is a leaf 
	 int cPtr[ 8 ];					//pointers to 8 children
	 int  atomsStartID, atomsEndID; // starting and ending id of the sAtoms
	 						//sum of the charges under this node
	                    //approximated sum
	 int numOfAtom;					//number of sAtoms;
     int id;
	 //i think it also hold array of atoms here and also calculate the sum of charges while building the octree
	 on(){
	 cx= cy= cz=cr=0;
	 atomsStartID=atomsEndID=numOfAtom=0;
	 nxq=nyq=nzq=0;
	 memset(cPtr, -1, 8*sizeof(int));
	 //sQ=NULL;
	 
	 }
}octreeNode;
/*Class of octree*/

class octree{
	  
public: 
    //these pointers can be passed to born radii
    int numStaticAtomsOctreeNodes;    //Number of  octree nodes in the static atoms octree
	int numMovingAtomsOctreeNodes;    //Number of octree nodes in the moving atoms octree
	octreeNode* sOctree;             //pointer to octree nodes for static nodes
	octreeNode* mOctree;             //pointer to octree nodes for moving nodes
	real Rmin;                     //min dimension
	int maxLeafSize;                 //max leaf size
	
	real BRmax, BRmin;
	
    real vdwRad[ 6 ];
	Atom * satms;                    //static atoms array
	QPoint * matms;                  //moving atoms array
	int nStaticAtom;                 //number of static atoms
	int nMovingAtom;                 //number of moving atoms
	int curNode, numNodes,maxNode, sOctreeRoot,mOctreeRoot;  ////Root of the the octrees
	real Err;
    real absErr;
    real perE;
    real tao;
    real epsilon1;               //epsilon parameter
    real epsilon2; 
    real epsilon1Plus1;
    real epsilon2Plus1;
    real thresold;   
    real onePlustwobyEps2;
	real onePlusTwobyEps;
    int MEpsilon;
    bool useApproxMath;
	real *bornRadi;
    real *sA; 
	real *sa; 
	real *sQ;
	octree(char *staticAtomsFile, char *movingAtomsFile){
	    //read pqr files
		initOctree();
	    nStaticAtom=readPQRfile(staticAtomsFile);
        nMovingAtom=readQUADfile(movingAtomsFile );
		buildOctrees();		
		
    }
	octree(char *staticAtomsFile){
	    //read pqr files
		initOctree();
		nStaticAtom=readPQRfile(staticAtomsFile);
        buildOctrees();		
		
    }

    void initOctree(){
	    
       satms=NULL;
	   matms=NULL;
	   sOctree=NULL;
	   mOctree=NULL;
	   Rmin=2;
	 
	   maxLeafSize=10;
	   nStaticAtom=0;
	   nMovingAtom=0;
	   
	   BRmax=0;
	   BRmin=1000;
       vdwRad[0] = 1.55;
	   vdwRad[1]= 1.70;
	   vdwRad[2]=1.40;
	   vdwRad[3]=1.20;
	   vdwRad[4]= 1.85;
	   vdwRad[5]=1.90;
	   numStaticAtomsOctreeNodes=0;
	   numMovingAtomsOctreeNodes=0;
	   sOctreeRoot=0;
	   mOctreeRoot=0;
	   Err=0;
       absErr=0;
       perE=0;

       tao=1-(1.0/80.0);
       epsilon1=0.9;               //epsilon parameter
       epsilon2=0.9; 
       epsilon1Plus1=1+epsilon1;
       epsilon2Plus1=1+epsilon2;
       thresold=pow(epsilon1Plus1,(1/6.000))  ;   
       onePlustwobyEps2=1.0+(2.0/epsilon2);
       MEpsilon=0;
       useApproxMath=true;
	}

	void setEpsilon1(real v){ 
	epsilon1=v;
	epsilon1Plus1=1+epsilon1;
    thresold=pow(epsilon1Plus1,(1/6.000))  ; 


	}
	
	void setEpsilon2(real v){ 
	epsilon2=v;
	epsilon2Plus1=1+epsilon2; 
	onePlustwobyEps2=1.0+(2.0/epsilon2);

	}
	void initFreeNodeServer( int numNodes )
	{
		curNode = 0;
		maxNode = numNodes;
	}
   //return next free node id
int nextFreeNode( void )
{
	   int nextNode = -1;	   

	   if ( curNode < maxNode ) nextNode = curNode++;

	   return nextNode;
}
	
/*
utility functions
*/
	int readPQRfile(char *file){
   //real startT = getTime( );
   clock_t begin1=clock();
   FILE *fp;
  // fstream ofile;
  // ofile.open ("result.txt", fstream::out | fstream::app); 
   fp = fopen( file, "rt" );
   
   if ( fp == NULL )
     {
      printf( "Failed to open PQR file (%s)!", file ); exit(1);
      return false;
     }
    int indx=0;
	char line [101];
    while ( fgets( line, 100, fp ) != NULL )
    {   char tmp[ 100 ];    
	    int i = 0;
        i = getAlphaString( line, i, tmp );   // get 'ATOM'/'HETATM', and ignore
		if ( !strcmp( tmp, "ATOM" ) )
			indx++;
    }
//	cout<<"Num of Atoms:"<<indx;
    satms=(Atom*)malloc(sizeof (Atom)*indx);
	indx=0;
	rewind(fp);
    while ( fgets( line, 100, fp ) != NULL )
     {
      int i = 0, j=0;
	  double v=0;
	 
      char tmp[ 100 ];    
	    
      i = getAlphaString( line, i, tmp );   // get 'ATOM'/'HETATM', and ignore
      
	  if ( !strcmp( tmp, "ATOM" ) ) {
		   
			  i = getInt( line, i, &j );            // get atom number, and ignore


			  i = getString( line, i, tmp );       //get atom name
			  char aName=tmp[0];
			  i = getAlphaString( line, i, tmp );   // get residue name, and ignore
			  

              i = getInt( line, i, &j );            // get residue number, and ignore

              i = getDouble( line, i, &v );         // get X coordinate
			  satms[ indx].x = v;  
			  
			  
		      satms[ indx].id=indx+1;
			  i = getDouble( line, i, &v );         // get Y coordinate
			  satms[ indx].y = v;    
			 

			  i = getDouble( line, i, &v );         // get Z coordinate
			  satms[ indx].z = v;    
			 
			  i = getDouble( line, i, &v );         // get charge
			  satms[ indx].q = v;  
			

			  i = getDouble( line, i, &v );         // get radius
			  if(aName=='N') satms[ indx].r =vdwRad[0];
			  else if(aName=='C') satms[ indx].r =vdwRad[1];
			  else if(aName=='O') satms[ indx].r =vdwRad[2];
			  else if(aName=='H') satms[ indx].r =vdwRad[3];
			  else if(aName=='S') satms[ indx].r =vdwRad[4];
			  else if(aName=='P') satms[ indx].r =vdwRad[5];
			  else satms[ indx].r = v;  
			  satms[ indx].ra = 0; 
			 // ofile<<satms[ indx].id<<" "<<satms[ indx].x <<" "<<satms[ indx].y<< " " << satms[ indx].z<<" "<<satms[ indx].q<<" "<<satms[ indx].r<<endl;
              indx++;
			

	  }
   
	}
	fclose( fp );
   // ofile.close();
    clock_t end1=clock();
    printf( "done ( %lf milisec, read %d sAtoms )\n", real(diffclock(end1,begin1)), indx );
 return indx;
}

	/*
Fields of .pqr files
	Field_name Atom_number Atom_name Residue_name Chain_ID Residue_number X Y Z Charge Radius 
*/


int readQUADfile(char *file){
  // real startT = getTime( );
   clock_t begin1=clock();
   FILE *fp;
  // fstream ofile;
  // ofile.open ("result.txt", fstream::out | fstream::app); 
   fp = fopen( file, "rt" );
   
   if ( fp == NULL )
     {
      printf( "Failed to open QUAD file (%s)!", file );exit(1);
      return false;
     }
   int indx=0;
   char line [101];
   while ( fgets( line, 100, fp ) != NULL )
   {
		indx++;
   }
    matms=(QPoint *)malloc(sizeof(QPoint)*indx);
	cout<<"Reading Quadrature points: "<<indx<<"\n";
	indx=0;
	rewind(fp);

    while ( fgets( line, 100, fp ) != NULL )
     {
      int i = 0;
	  double v=0;
	
           
     		  i = getDouble( line, i, &v );         // get X coordinate
			  matms[ indx].rx = v;  
			
		    
			  i = getDouble( line, i, &v );         // get Y coordinate
			  matms[ indx].ry = v;    
			

			  i = getDouble( line, i, &v );         // get Z coordinate
			  matms[ indx].rz = v;    
			 
			  i = getDouble( line, i, &v );         // get nx
			  matms[ indx].nx = v;  
			 

			  i = getDouble( line, i, &v );         // get ny
			  matms[ indx].ny = v;  

			  i = getDouble( line, i, &v );         // get nz
			  matms[ indx].nz = v;  


			   i = getDouble( line, i, &v );         // get weight
			  matms[ indx].w = v;  

			 // ofile<<matms[ indx].rx<<" "<<matms[ indx].ry <<" "<<matms[ indx].rz<< " " << matms[ indx].nx<<" "<<matms[ indx].ny<<" "<<matms[ indx].nz<<matms[ indx].w<<endl;
              indx++;
			
	   
	}
	fclose( fp );
	//ofile.close();
    clock_t end1=clock();
    printf( "done ( %lf miliec, read %d qPoints )\n", real(diffclock(end1,begin1)), indx );
 return indx;
}
	
	/*
	build the octree from the provided atoms file, initilize static atoms and moving atoms 
	*/
	//constructors
    
   //finds how many nodes you needed for this octree and sort the atoms to the octree node
   void countAtomsOctreeNodesAndSortAtoms( Atom *sAtoms, int atomsStartID, int atomsEndID, Atom *atomsT, int *numNodes)
   {
  
   real minX = sAtoms[ atomsStartID ].x, minY = sAtoms[ atomsStartID ].y, minZ = sAtoms[ atomsStartID ].z;
   real maxX = sAtoms[ atomsStartID ].x, maxY = sAtoms[ atomsStartID ].y, maxZ = sAtoms[ atomsStartID ].z;

  /*Find the minimum of */
     for ( int i = atomsStartID; i <= atomsEndID; i++ )
     { 
          if ( sAtoms[ i ].x < minX ) minX = sAtoms[ i ].x;      
          if ( sAtoms[ i ].x > maxX ) maxX = sAtoms[ i ].x;      
      
          if ( sAtoms[ i ].y < minY ) minY = sAtoms[ i ].y;      
          if ( sAtoms[ i ].y > maxY ) maxY = sAtoms[ i ].y;      

          if ( sAtoms[ i ].z < minZ ) minZ = sAtoms[ i ].z;      
          if ( sAtoms[ i ].z > maxZ ) maxZ = sAtoms[ i ].z;      
      } 
    /*Find the center of atoms*/
   real cx = ( minX + maxX ) / 2,
          cy = ( minY + maxY ) / 2,
          cz = ( minZ + maxZ ) / 2;

   /*I think we need to change it should be minX or maxX
     Its finding the maximum radious
   */
   real r2 = ( sAtoms[ atomsStartID ].x - cx ) * ( sAtoms[ atomsStartID ].x - cx )
             + ( sAtoms[ atomsStartID ].y - cy ) * ( sAtoms[ atomsStartID ].y - cy )
             + ( sAtoms[ atomsStartID ].z - cz ) * ( sAtoms[ atomsStartID ].z - cz );


      for ( int i = atomsStartID; i <= atomsEndID; i++ )
        { 
          real r2T = ( sAtoms[ i ].x - cx ) * ( sAtoms[ i ].x - cx )
                     + ( sAtoms[ i ].y - cy ) * ( sAtoms[ i ].y - cy )
                     + ( sAtoms[ i ].z - cz ) * ( sAtoms[ i ].z - cz );
         
          if ( r2T > r2 ) r2 = r2T;        
        }
	
  //have root of rsquare + and add minimum atomic distance on both sides  
   real cr = sqrt( r2 );
   /*counting number of atoms*/
   int numAtoms = 0;

     if ( atomsEndID >= atomsStartID ) 
        numAtoms = ( atomsEndID- atomsStartID + 1 ); 
   
	 /*Counting number of octree Nodes*/
   *numNodes = 1;
   //dividing the big octree to a number of children   
   if ( ( numAtoms > maxLeafSize ) && ( cr > Rmin ) )
   {
      int atomsCount[ 8 ]= { 0, 0, 0, 0, 0, 0, 0, 0 }; //counts how many atoms goes to each children 
	//  int atomsCountAllTypes[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      
      for ( int i = atomsStartID; i <= atomsEndID; i++ )
      {
           atomsT[ i ] = sAtoms[ i ];
           //finding the index 
           int j = ( zeroIfLess( sAtoms[ i ].z, cz ) << 2 )
                 + ( zeroIfLess( sAtoms[ i ].y, cy ) << 1 )
                 + ( zeroIfLess( sAtoms[ i ].x, cx ) );
           
           atomsCount[ j ]++;
		 //  atomsCountAllTypes[ j ]++;
         
        }
      //keeping track of all the startindex and endindex of atoms that goes to all the childrens
      int atomsStartIndex[ 8 ], atomsEndIndex[ 8 ];
	  //keeping track of current index
      int atomsCurIndex[ 8 ];              
      
      
      atomsCurIndex[ 0 ] = atomsStartIndex[ 0 ] = atomsStartID;
      for ( int i = 1; i < 8; i++ )
            atomsCurIndex[ i ]= atomsStartIndex[ i ] = atomsStartIndex[ i - 1 ] + atomsCount[ i - 1 ];
     //here atoms cur index is keeping track of the start index
     //sorting the atoms based on the node they are going to go.

      for ( int i = atomsStartID; i <= atomsEndID; i++ )
          {        
          int j = ( zeroIfLess( atomsT[ i ].z, cz ) << 2 )
                  + ( zeroIfLess( atomsT[ i ].y, cy ) << 1 )
                  + ( zeroIfLess( atomsT[ i ].x, cx ) );
             
            sAtoms[ atomsCurIndex[ j ]] = atomsT[ i ];
            atomsCurIndex[ j ]++;  
			
         }                    
           
        
     for ( int i = 0; i < 8; i++ ) 
        if ( atomsCount[ i ] > 0 )
          {
           int numNodesT = 0;
           
              atomsEndIndex[ i ] = atomsStartIndex[ i ] + atomsCount[ i ]- 1;

		   //this is a recursive function call. we can make it parallel
            countAtomsOctreeNodesAndSortAtoms( sAtoms, atomsStartIndex[ i ], atomsEndIndex[ i ], atomsT, &numNodesT );
         
           *numNodes += numNodesT;
          } 
		 
   }
}
  
  int constructAtomsOctree( int atomsStartID, int atomsEndID, Atom *atoms, octreeNode *atomsOctree ){
   
	  //take the next free node
   int nodeID = nextFreeNode( );
   
   if ( nodeID < 0 ) return nodeID;
      
   //determine the start atom id and end atom id for the atoms of the node
      atomsOctree[ nodeID ].id=nodeID;
	  atomsOctree[ nodeID ].nxq=0;
	  atomsOctree[ nodeID ].nyq=0;
	  atomsOctree[ nodeID ].nzq=0;
	
      atomsOctree[ nodeID ].atomsStartID = atomsStartID;
      atomsOctree[ nodeID ].atomsEndID = atomsEndID;
    

   real minX = atoms[ atomsStartID ].x, minY = atoms[ atomsStartID ].y, minZ = atoms[ atomsStartID ].z;
   real maxX = atoms[ atomsStartID ].x, maxY = atoms[ atomsStartID].y, maxZ = atoms[ atomsStartID ].z;

   //determining the minimum of x, y, z
     for ( int i = atomsStartID; i <= atomsEndID; i++ )
        { 
          if ( atoms[ i ].x < minX ) minX = atoms[ i ].x;      
          if ( atoms[ i ].x > maxX ) maxX = atoms[ i ].x;      
      
          if ( atoms[ i ].y < minY ) minY = atoms[ i ].y;      
          if ( atoms[ i ].y > maxY ) maxY = atoms[ i ].y;      

          if ( atoms[ i ].z < minZ ) minZ = atoms[ i ].z;      
          if ( atoms[ i ].z > maxZ ) maxZ = atoms[ i ].z;  

		  //accumulating charge values

		 // atomsOctree[ nodeID ].sQ=atomsOctree[ nodeID ].sQ+atoms[ i ].q;
        } 
   //determining the center of the octree Node
   //I do not know why I need to do this two times
   real cx = atomsOctree[ nodeID ].cx = ( minX + maxX ) / 2;
   real cy = atomsOctree[ nodeID ].cy = ( minY + maxY ) / 2;
   real cz = atomsOctree[ nodeID ].cz = ( minZ + maxZ ) / 2;

   //determining the dimension of the node
   real r2 = ( atoms[ atomsStartID ].x - cx ) * ( atoms[ atomsStartID ].x - cx )
             + ( atoms[ atomsStartID ].y - cy ) * ( atoms[ atomsStartID ].y - cy )
             + ( atoms[ atomsStartID ].z - cz ) * ( atoms[ atomsStartID ].z - cz );

   for ( int i = atomsStartID; i <= atomsEndID; i++ )
        { 
          real r2T = ( atoms[ i ].x - cx ) * ( atoms[ i ].x - cx )
                     + ( atoms[ i ].y - cy ) * ( atoms[ i ].y - cy )
                     + ( atoms[ i ].z - cz ) * ( atoms[ i ].z - cz );
         
          if ( r2T > r2 ) r2 = r2T;        
        }
   
   real cr = atomsOctree[ nodeID ].cr = sqrt( r2 );
  // atomsOctree[ nodeID ].dim=cr;
   int numAtoms = 0;

      if ( atomsEndID >= atomsStartID ) 
         atomsOctree[ nodeID ].numOfAtom = atomsEndID - atomsStartID + 1;
      else 
         atomsOctree[ nodeID ].numOfAtom = 0;   
      numAtoms = atomsOctree[ nodeID ].numOfAtom; 
    
//when the octree node becomes a leaf
   if ( ( numAtoms <= maxLeafSize ) || ( cr <= Rmin ) )
      atomsOctree[ nodeID ].leaf = true;

   else //if not, then again subdivide it recursively
   {
      atomsOctree[ nodeID ].leaf = false;     

      int atomsCount[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      //why do not  i recursively call it again?
       for ( int i = atomsStartID; i <= atomsEndID; i++ )
      {
         int j = ( zeroIfLess( atoms[ i ].z, cz ) << 2 )
                 + ( zeroIfLess( atoms[ i ].y, cy ) << 1 )
                 + ( zeroIfLess( atoms[ i ].x, cx ) );
           
           atomsCount[ j ]++;
          }
        
      int atomsStartIndex[ 8 ], atomsEndIndex[ 8 ];

      
         atomsStartIndex[ 0 ] = atomsStartID;
         for ( int i = 1; i < 8; i++ )
            atomsStartIndex[ i ] = atomsStartIndex[ i - 1 ] + atomsCount[ i - 1 ];
          
      
      for ( int i = 0; i < 8; i++ ) 
        if ( atomsCount[ i ] > 0 )
          {
           
           atomsEndIndex[ i ] = atomsStartIndex[ i ] + atomsCount[ i ] - 1;

		   //this is again a recursive function call. You can spawn it.

           int j = constructAtomsOctree( atomsStartIndex[ i ], atomsEndIndex[ i ], atoms, atomsOctree );

		   //setting the child pointer
           atomsOctree[ nodeID ].cPtr[ i ] = j; 
          }
        else atomsOctree[ nodeID ].cPtr[ i ] = -1;  
     }  
     
   return nodeID;  
}
bool buildStaticAtomsOctree( void ){
   
   clock_t begin1=clock();
   Atom *atomsT=NULL;
   if ( nStaticAtom > 0 )
   {
         atomsT = ( Atom * ) malloc( nStaticAtom * sizeof( Atom ) );
   
         if ( atomsT == NULL )
           {
            printf( "Failed to allocate temporary memory for static sAtoms!" );
            return false;
           }
    }
      int atomsStartIndex=0, atomsEndIndex=nStaticAtom-1;  //may be we need to change this.

	  //counts octree node..............
	  //I am still not clear why atomsT is needed.
	  countAtomsOctreeNodesAndSortAtoms( satms, atomsStartIndex, atomsEndIndex, atomsT, &numStaticAtomsOctreeNodes );
   
	  freeMem( atomsT );
      //previous call seems to count how many octree node we actually need 
      octreeNode *atomsOctreeT;
   
      atomsOctreeT = ( octreeNode * ) malloc( numStaticAtomsOctreeNodes * sizeof( octreeNode ) );
 
   if ( atomsOctreeT == NULL )
     {
     // printf( "Unable to %s static atoms octree - memory allocation failed!", ( staticAtomsOctreeBuilt ) ? "rebuild" : "build" );
     // if ( !staticAtomsOctreeBuilt ) exit( 1 );
	  cout<<"Error! Do not have enough memory!\n";
      return false;
     }
 
   freeMem( sOctree );
   sOctree = atomsOctreeT; 

   initFreeNodeServer( numStaticAtomsOctreeNodes );
   
   sOctreeRoot = constructAtomsOctree( atomsStartIndex, atomsEndIndex, satms, sOctree );

  
   /* clock_t end1=clock();
	real t=real(diffclock(end1,begin1)/1000);
	if (t<60)
    printf( "done Time to build the octree( %lf sec, calculated )\n", t);
	else 
	printf( "done Time to build the octree( %lf min, calculated )\n", t/60);*/
   
   return true;
   
  }  
/********************************************************************************/


  //finds how many nodes you needed for this octree and sort the atoms to the octree node
   void countAtomsOctreeNodesAndSortQPoints( QPoint*sAtoms, int atomsStartID, int atomsEndID, QPoint *atomsT, int *numNodes)
   {
  
   real minX = sAtoms[ atomsStartID].rx, minY = sAtoms[ atomsStartID ].ry, minZ = sAtoms[ atomsStartID ].rz;
   real maxX = sAtoms[ atomsStartID ].rx, maxY = sAtoms[atomsStartID ].ry, maxZ = sAtoms[ atomsStartID ].rz;

  /*Find the minimum of */
     for ( int i = atomsStartID; i <= atomsEndID; i++ )
     { 
          if ( sAtoms[ i ].rx < minX ) minX = sAtoms[ i ].rx;      
          if ( sAtoms[ i ].rx > maxX ) maxX = sAtoms[ i ].rx;      
      
          if ( sAtoms[ i ].ry < minY ) minY = sAtoms[ i ].ry;      
          if ( sAtoms[ i ].ry > maxY ) maxY = sAtoms[ i ].ry;      

          if ( sAtoms[ i ].rz < minZ ) minZ = sAtoms[ i ].rz;      
          if ( sAtoms[ i ].rz > maxZ ) maxZ = sAtoms[ i ].rz;      
      } 
    /*Find the center of atoms*/
   real cx = ( minX + maxX ) / 2,
          cy = ( minY + maxY ) / 2,
          cz = ( minZ + maxZ ) / 2;

   /*I think we need to change it should be minX or maxX
     Its finding the maximum radious
   */
   real r2 = ( sAtoms[ atomsStartID ].rx - cx ) * ( sAtoms[ atomsStartID ].rx - cx )
             + ( sAtoms[ atomsStartID ].ry - cy ) * ( sAtoms[ atomsStartID ].ry - cy )
             + ( sAtoms[ atomsStartID ].rz - cz ) * ( sAtoms[atomsStartID ].rz - cz );


      for ( int i = atomsStartID; i <= atomsEndID; i++ )
        { 
          real r2T = ( sAtoms[ i ].rx - cx ) * ( sAtoms[ i ].rx - cx )
                     + ( sAtoms[ i ].ry - cy ) * ( sAtoms[ i ].ry - cy )
                     + ( sAtoms[ i ].rz - cz ) * ( sAtoms[ i ].rz - cz );
         
          if ( r2T > r2 ) r2 = r2T;        
        }
	 // dim=r2;
  //have root of rsquare + and add minimum atomic distance on both sides  
   real cr = sqrt( r2 );
  // atomsOctree[ nodeID ].dim=cr;
   /*counting number of atoms*/
   int numAtoms = 0;

     if ( atomsEndID >= atomsStartID ) 
        numAtoms = ( atomsEndID- atomsStartID + 1 ); 
   
	 /*Counting number of octree Nodes*/
   *numNodes = 1;
   //dividing the big octree to a number of children   
   if ( ( numAtoms > maxLeafSize ) && ( cr > Rmin ) )
   {
      int atomsCount[ 8 ]= { 0, 0, 0, 0, 0, 0, 0, 0 }; //counts how many atoms goes to each children 
	//  int atomsCountAllTypes[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      
      for ( int i = atomsStartID; i <= atomsEndID; i++ )
      {
           atomsT[ i ] = sAtoms[ i ];
           //finding the index 
           int j = ( zeroIfLess( sAtoms[ i ].rz, cz ) << 2 )
                 + ( zeroIfLess( sAtoms[ i ].ry, cy ) << 1 )
                 + ( zeroIfLess( sAtoms[ i ].rx, cx ) );
           
           atomsCount[ j ]++;
		 //  atomsCountAllTypes[ j ]++;
         
        }
      //keeping track of all the startindex and endindex of atoms that goes to all the childrens
      int atomsStartIndex[ 8 ], atomsEndIndex[ 8 ];
	  //keeping track of current index
      int atomsCurIndex[ 8 ];              
      
      
      atomsCurIndex[ 0 ] = atomsStartIndex[ 0 ] = atomsStartID;
      for ( int i = 1; i < 8; i++ )
            atomsCurIndex[ i ]= atomsStartIndex[ i ] = atomsStartIndex[ i - 1 ] + atomsCount[ i - 1 ];
     //here atoms cur index is keeping track of the start index
     //sorting the atoms based on the node they are going to go.

      for ( int i = atomsStartID; i <= atomsEndID; i++ )
          {        
          int j = ( zeroIfLess( atomsT[ i ].rz, cz ) << 2 )
                  + ( zeroIfLess( atomsT[ i ].ry, cy ) << 1 )
                  + ( zeroIfLess( atomsT[ i ].rx, cx ) );
             
            sAtoms[ atomsCurIndex[ j ]] = atomsT[ i ];
            atomsCurIndex[ j ]++;  
			
         }                    
           
        
     for ( int i = 0; i < 8; i++ ) 
        if ( atomsCount[ i ] > 0 )
          {
           int numNodesT = 0;
           
              atomsEndIndex[ i ] = atomsStartIndex[ i ] + atomsCount[ i ]- 1;

		   //this is a recursive function call. we can make it parallel
           
           countAtomsOctreeNodesAndSortQPoints( sAtoms, atomsStartIndex[ i ], atomsEndIndex[ i ], atomsT, &numNodesT );
         
           *numNodes += numNodesT;
          } 
		 
   }
}
  
  int constructQOctree( int atomsStartID, int atomsEndID, QPoint *atoms, octreeNode *atomsOctree ){
   
	  //take the next free node
   int nodeID = nextFreeNode( );
   
   if ( nodeID < 0 ) return nodeID;
      
   //determine the start atom id and end atom id for the atoms of the node
      atomsOctree[ nodeID ].id=nodeID;
	  atomsOctree[ nodeID ].nxq=0;
	  atomsOctree[ nodeID ].nyq=0;
	  atomsOctree[ nodeID ].nzq=0;
	
      atomsOctree[ nodeID ].atomsStartID = atomsStartID;
      atomsOctree[ nodeID ].atomsEndID = atomsEndID;
    

   real minX = atoms[ atomsStartID ].rx, minY = atoms[ atomsStartID ].ry, minZ = atoms[ atomsStartID ].rz;
   real maxX = atoms[ atomsStartID ].rx, maxY = atoms[ atomsStartID ].ry, maxZ = atoms[ atomsStartID ].rz;

   //determining the minimum of x, y, z
     for ( int i = atomsStartID; i <= atomsEndID; i++ )
        { 
          if ( atoms[ i ].rx < minX ) minX = atoms[ i ].rx;      
          if ( atoms[ i ].rx > maxX ) maxX = atoms[ i ].rx;      
      
          if ( atoms[ i ].ry < minY ) minY = atoms[ i ].ry;      
          if ( atoms[ i ].ry > maxY ) maxY = atoms[ i ].ry;      

          if ( atoms[ i ].rz < minZ ) minZ = atoms[ i ].rz;      
          if ( atoms[ i ].rz > maxZ ) maxZ = atoms[ i ].rz;  

		  //accumulating in nxq, nyq and nzq

		    atomsOctree[ nodeID ].nxq+=atoms[ i ].w*atoms[ i ].nx;
			atomsOctree[ nodeID ].nyq+=atoms[ i ].w*atoms[ i ].ny;
			atomsOctree[ nodeID ].nzq+=atoms[ i ].w*atoms[ i ].nz;

		
			
        } 
   //determining the center of the octree Node
   //I do not know why I need to do this two times
   real cx = atomsOctree[ nodeID ].cx = ( minX + maxX ) / 2;
   real cy = atomsOctree[ nodeID ].cy = ( minY + maxY ) / 2;
   real cz = atomsOctree[ nodeID ].cz = ( minZ + maxZ ) / 2;

   //determining the dimension of the node
   real r2 = ( atoms[ atomsStartID ].rx - cx ) * ( atoms[ atomsStartID ].rx - cx )
             + ( atoms[ atomsStartID ].ry - cy ) * ( atoms[ atomsStartID ].ry - cy )
             + ( atoms[ atomsStartID ].rz - cz ) * ( atoms[ atomsStartID ].rz - cz );

   for ( int i = atomsStartID; i <= atomsEndID; i++ )
        { 
          real r2T = ( atoms[ i ].rx - cx ) * ( atoms[ i ].rx - cx )
                     + ( atoms[ i ].ry - cy ) * ( atoms[ i ].ry - cy )
                     + ( atoms[ i ].rz - cz ) * ( atoms[ i ].rz - cz );
         
          if ( r2T > r2 ) r2 = r2T;        
        }
   
   real cr = atomsOctree[ nodeID ].cr = sqrt( r2 );
 //  atomsOctree[ nodeID ].dim=cr;
   int numOfAtom = 0;

      if ( atomsEndID >= atomsStartID ) 
         atomsOctree[ nodeID ].numOfAtom = atomsEndID - atomsStartID + 1;
      else 
         atomsOctree[ nodeID ].numOfAtom = 0;   
      numOfAtom = atomsOctree[ nodeID ].numOfAtom; 
    
//when the octree node becomes a leaf
   if ( ( numOfAtom <= maxLeafSize ) || ( cr <= Rmin ) )
      atomsOctree[ nodeID ].leaf = true;

   else //if not, then again subdivide it recursively
   {
      atomsOctree[ nodeID ].leaf = false;     

      int atomsCount[ 8 ] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      //why do not  i recursively call it again?
      for ( int i = 0; i < 8; i++ )
            atomsCount[ i ]= 0;

      for ( int i = atomsStartID; i <= atomsEndID; i++ )
      {
         int j = ( zeroIfLess( atoms[ i ].rz, cz ) << 2 )
                 + ( zeroIfLess( atoms[ i ].ry, cy ) << 1 )
                 + ( zeroIfLess( atoms[ i ].rx, cx ) );
           
           atomsCount[ j ]++;
          }
        
      int atomsStartIndex[ 8 ], atomsEndIndex[ 8 ];

      
         atomsStartIndex[ 0 ] = atomsStartID;
         for ( int i = 1; i < 8; i++ )
            atomsStartIndex[ i ] = atomsStartIndex[ i - 1 ] + atomsCount[ i - 1 ];
          
      
      for ( int i = 0; i < 8; i++ ) 
        if ( atomsCount[ i ] > 0 )
          {
           
           atomsEndIndex[ i ] = atomsStartIndex[ i ] + atomsCount[ i ] - 1;

		   //this is again a recursive function call. You can spawn it.

           int j = constructQOctree( atomsStartIndex[ i ], atomsEndIndex[ i ], atoms, atomsOctree );

		   //setting the child pointer
           atomsOctree[ nodeID ].cPtr[ i ] = j; 
          }
        else atomsOctree[ nodeID ].cPtr[ i ] = -1;  
     }  
     
   return nodeID;  
}
/*******************************************************************************/
bool buildMovingAtomsOctree( void ){
   clock_t begin1=clock();
   QPoint *atomsT=NULL;
   if ( nMovingAtom > 0 )
   {
         atomsT = ( QPoint * ) malloc( nMovingAtom * sizeof( QPoint ) );
   
         if ( atomsT == NULL )
           {
            printf( "Failed to allocate temporary memory for static sAtoms!" );
            return false;
           }
    }
      int atomsStartIndex=0, atomsEndIndex=nMovingAtom-1;  

	  //counts octree node..............
	
	  countAtomsOctreeNodesAndSortQPoints( matms, atomsStartIndex, atomsEndIndex, atomsT, &numMovingAtomsOctreeNodes );
   
	  freeMem( atomsT );
      //previous call seems to count how many octree node we actually need 
      octreeNode *atomsOctreeT;
   
      atomsOctreeT = ( octreeNode * ) malloc( numMovingAtomsOctreeNodes * sizeof( octreeNode ) );
 
   if ( atomsOctreeT == NULL )
     {
     // printf( "Unable to %s static atoms octree - memory allocation failed!", ( staticAtomsOctreeBuilt ) ? "rebuild" : "build" );
     // if ( !staticAtomsOctreeBuilt ) exit( 1 );
	  cout<<"Error! Do not have enough memory!\n";
      return false;
     }
 
   freeMem( mOctree );
   mOctree = atomsOctreeT; 

   initFreeNodeServer( numMovingAtomsOctreeNodes );
   
   mOctreeRoot = constructQOctree( atomsStartIndex, atomsEndIndex, matms, mOctree );

  
   return true;

}          
   bool buildOctrees( void ){
	if (buildStaticAtomsOctree()==true && buildMovingAtomsOctree()==true)
	  			return true;
		return false;
   }         
   int getSubtreeSize( int nodeID, octreeNode *octree );
   int getOctreeSize( octreeNode *octree );
	 
   /**
   Traverse the tree pointed to by 'octree', and print
   info on each node. 
*/
void traverse_octree( int node_id, octreeNode *oNode)
{   
	octreeNode node =  oNode[node_id];
 
    cout<<node_id;
	cout<<"  ";
	cout<<node.numOfAtom;
	cout<<"  ";
	cout<<node.atomsStartID;
	cout<<"  ";
	cout<<node.atomsEndID;
	cout<<"Center+Radius:";
	cout<<node.cx;
	cout<<"  ";
	cout<<node.cy;
	cout<<"  ";
	cout<< node.cz;
	cout<<"  ";
	cout<<node.cr;
	cout<<"Childrens:";
   if ( !node.leaf )
      for ( int i = 0; i < 8; i++ )
        if ( node.cPtr[ i ] >= 0 ) cout<<node.cPtr[ i ];
     
   cout<<"\n" ;  

   if ( !node.leaf )
      for ( int i = 0; i < 8; i++ )
        if ( node.cPtr[ i ] >= 0 ) 
           traverse_octree( node.cPtr[ i ], oNode);
}


/**
   Print the tree pointed to by 'octree'. 
*/
void print_octree(octreeNode *oNode )
{  
   //ofile.open ("Octree.txt");
   traverse_octree( 0, oNode );
   
}  





void printBornRadi(string s){
	ofstream ofile;
	ofile.open (s.c_str());
	for (int i=0;i<nStaticAtom;i++){
	//	ofile<<setw(2)<<fixed;
		ofile<<i<<" :"<<sa[i]<<endl;
	
	}
    ofile.close();
}
void printBornRadi(){
	ofstream ofile, ofile2;
	ofile.open ("Octree.txt");
	ofile2.open ("Naive.txt");
	for (int i=0;i<nStaticAtom;i++){
		ofile<<setw(2)<<fixed;
		ofile<<i<<" :"<<satms[i].ra<<endl;

		ofile2<<setw(2)<<fixed;
		ofile2<<i<<" :"<<bornRadi[i]<<endl;
	
	}
    ofile.close();
	ofile2.close();
}

void calculateBornRadi(){

 
 real maxBornRadius=1000;
 bornRadi= new real[nStaticAtom];
 memset(bornRadi,0,sizeof(real)*nStaticAtom);

 for(int i=0;i<nStaticAtom;i++)
 {
	 real v=0;
	 real x = satms[i].x;
	 real y = satms[i].y;
	 real z = satms[i].z;
     for(int k=0;k<nMovingAtom;k++)
	 {   real m=0;
         real dx = (matms[k].rx-x);
		 real dy = matms[k].ry-y;
		 real dz = (matms[k].rz-z);
		 
	     m=dx*matms[k].nx+(dy)*matms[k].ny+dz*matms[k].nz;
	     real n=0;

		 n=(dx)*(dx)+(dy)*(dy)+(dz)*(dz);
		 v=v+matms[k].w*m/(n*n*n);
		
		  
	     }
		
	 if (v>0)
	  bornRadi[i]=pow((v*Inv_4PI),-(.33333333333));
	 if(satms[i].r>bornRadi[i])
         bornRadi[i]=satms[i].r; 

     else if ( bornRadi[i]> maxBornRadius ) bornRadi[i]= maxBornRadius; 
	
	
	
}

}


 /*void memfree2(){
	for (int i=0;i<numStaticAtomsOctreeNodes;i++)
	{
	 freeMem(sOctree[i].sQ);
	 freeMem(sOctree[i].cPtr);

	 }

  for (int i=0;i<numMovingAtomsOctreeNodes;i++)
	{
	 freeMem(mOctree[i].sQ);
	 freeMem(mOctree[i].cPtr);

	}
 }
*/
 void calBornRadiErr(){
 
 for(int i=0;i<nStaticAtom;i++){
  real d=bornRadi[i]-satms[i].ra;
  perE=perE+fabs(d*100)/bornRadi[i];
  Err=Err+d;
  absErr=absErr+fabs(d);

 }
 
 }
 inline float invSqrt( float x )
{
   if ( x < 0 ) return INFINITY;
   
   volatile UL_F_union xx;
   
   xx.ix = 0;      
   xx.fx = x;
   xx.ix = ( 0xBE6EB50CUL - xx.ix ) >> 1;   
      
   return ( 0.5f * xx.fx * ( 3.0f - x * xx.fx * xx.fx ) );
}

inline float fastExp( float x )
{
   if ( x > 85 ) return INFINITY;
   if ( x < -85 ) return 0.0;   
   
   volatile UL_F_union xx;
         
   // 12102203.16156148555068672305845f = ( 2^23 ) / ln(2);      
   unsigned long i = ( unsigned long ) ( x * 12102203.16156f );
   
   // 361007 = ( 0.08607133 / 2 ) * ( 2^23 )         
   xx.ix = i + ( 127L << 23 ) - 361007;
   
   return xx.fx;
}



real NaiveEPol(){
	  real Epol=0;
	  real dx,dy,dz,r2ij,f_ijGB, RiRj;
	  for(int i=0;i<nStaticAtom;i++){
	      real x = satms[i].x;
		  real y = satms[i].y;
		  real z = satms[i].z;
		  for(int j=0;j<nStaticAtom;j++){
			    dx=x-satms[j].x;
                dy=y-satms[j].y;
                dz=z-satms[j].z;
			    r2ij=dx*dx+dy*dy+dz*dz;
			    RiRj=bornRadi[i]*bornRadi[j];
                /*if ( useApproxMath ) 
					f_ijGB=invSqrt((float)r2ij+RiRj*fastExp(-(float)(r2ij/(4*RiRj))));
				else */
					f_ijGB=sqrt(r2ij+RiRj*exp(-(r2ij/(4*RiRj))));
			    Epol=Epol+(satms[i].q*satms[j].q)/f_ijGB;

		  }
	  }
   return -Epol*tao/2;
  
  }
//end of class definations
};
