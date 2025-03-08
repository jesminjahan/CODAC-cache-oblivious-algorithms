#include<cilk.h>
#include"Octree.h"


using namespace std;
//octree ob;

real BRmin2;
real *epsilon2Plus1P;
/******************************************************************************************************
Function for calculating born radius
**************************************************************************************************/
void approxIntegrals(octree &ob,octreeNode *aNode, int anode_id, octreeNode *qNode, int qnode_id){

	octreeNode* qptN=&qNode[qnode_id];
	octreeNode* qptM=&aNode[anode_id];
	real dx=qptN->cx-qptM->cx;
	real dy=qptN->cy-qptM->cy;
    real dz=qptN->cz-qptM->cz;
   
	real d2=dx * dx + dy * dy + dz * dz;

    real sumOfRadius=qptM->cr+qptN->cr;

    real r_AQ=sqrt(d2);

    real d=r_AQ-(sumOfRadius);


	if(d>(sumOfRadius)&&((r_AQ+sumOfRadius)/d)>ob.thresold) //far enough to approximate
	{   
		real m=(qptN->nxq*(dx)+qptN->nyq*(dy)+qptN->nzq*(dz))/(d2*d2*d2);
		ob.sA[anode_id]=ob.sA[anode_id]+m;  // r_AQ=d2

	
	}
	else if(qptM->leaf==true &&qptN->leaf==true)
	{
	   for(int j=qptM->atomsStartID;j<=qptM->atomsEndID;j++)
	   {    real saTemp=0; 
	        Atom* atm=&(ob.satms[j]);
			for(int i=qptN->atomsStartID;i<=qptN->atomsEndID;i++)
			{   QPoint *qpt=&(ob.matms[i]);
				real dX=(qpt->rx-(atm->x));
				real dY=(qpt->ry-(atm->y));
				real dZ=(qpt->rz-(atm->z));
				real raq=dX*dX+dY*dY+dZ*dZ;
				saTemp+=((qpt->w*(qpt->nx*dX+qpt->ny*dY+qpt->nz*dZ))/(raq*raq*raq));
			
			}
				ob.sa[j]+=saTemp;
	    }
	}
	//make it parallel
	else if(qptM->leaf==true){
	   for(int i=0;i<8;i++)
	   { 
		   if(qptN->cPtr[i]>=0)
		   {
				approxIntegrals(ob,aNode, anode_id, qNode,qptN->cPtr[i]);
		   }
	   
	   }
	
	}

	else if(qptN->leaf==true){
	   cilk_for(int i=0;i<8;i++)
	   { 
		   if(qptM->cPtr[i]>=0)
		   {
				approxIntegrals(ob,aNode, qptM->cPtr[i], qNode, qnode_id);
		   }
	   
	   }
	
	}
   else{
	   cilk_for(int i=0;i<8;i++)
	   { 
		   if(qptM->cPtr[i]>=0)
		   {
				for(int j=0;j<8;j++)
				{
				   if(qptN->cPtr[j]>=0)
					approxIntegrals(ob,aNode, qptM->cPtr[i], qNode,qptN->cPtr[j]);
				}
			   
		   }
	   
	   }
	
	}
	
  
	return;
}

/****************************************************************************************************/

void pushIntegralsToAtoms(octree &ob,octreeNode *aNode,int anode_id, real S){
     if ( anode_id < 0 ) return;
	 octreeNode * an=&aNode[anode_id];
     S=S+ob.sA[anode_id];
	 if(an->leaf==true){
		for(int j=an->atomsStartID;j<=an->atomsEndID;j++){
            real s=(ob.sa[j]+S)*Inv_4PI;
			real d=0;
			if(s>0){d=((pow(s,-(.33333333333))));}
			Atom * atm=&(ob.satms[j]);
			if ((atm->r) > d)
				atm->ra=atm->r;
			else
			{
				atm->ra=d;
			}
	  if ( atm->ra> 1000 ) atm->ra= 1000; 
      if(ob.BRmax<atm->ra)
		ob.BRmax=atm->ra;
      if(ob.BRmin>atm->ra)
		ob.BRmin=atm->ra;
			
		}
		
	}
	else 
	{ 
		cilk_for(int i=0;i<8;i++){
			//if(aNode[anode_id].cPtr[i]>=0)
			{   
				pushIntegralsToAtoms(ob,aNode, an->cPtr[i], S);
			}
		}
	}

	 return;
}

void calSQ(octree &ob){
ob.sQ=new real [ob.MEpsilon*ob.numStaticAtomsOctreeNodes];
memset(ob.sQ, 0, ob.MEpsilon*ob.numStaticAtomsOctreeNodes*sizeof(real));
cilk_for(int uI=0;uI<ob.numStaticAtomsOctreeNodes;uI++){
	int index;
	int start=uI*ob.MEpsilon;
	for (int k=0;k<ob.MEpsilon;k++)
	  {   index=start+k;
	      ob.sQ[index]=0;
		  real v=0;
		  for(int u=ob.sOctree[uI].atomsStartID;u<=ob.sOctree[uI].atomsEndID;u++)
		  { 
		      real d1=ob.BRmin*(pow(ob.epsilon2Plus1,real(k)));
			  real d2=d1*ob.epsilon2Plus1;
			  if(((ob.satms[u]).ra >=d1)&&((ob.satms[u]).ra <d2))
		        v+=ob.satms[u].q;
		  }
		  ob.sQ[index]+=v;
	    }
    }
	 BRmin2=ob.BRmin*ob.BRmin;
	 int mm=ob.MEpsilon*ob.MEpsilon;
	 epsilon2Plus1P=new real [mm];
	 memset(epsilon2Plus1P, 0, sizeof(real)*mm);
	 for(int i=0;i<ob.MEpsilon;i++)
		   for(int j=0;j<ob.MEpsilon;j++)
		   {  
			 
			   epsilon2Plus1P[i*ob.MEpsilon+j]=BRmin2*pow(ob.epsilon2Plus1,real(i+j));
			  
		  
	        } 
return;
}
real ApproxE_pol(octree &ob,octreeNode *U, int uI, octreeNode *V, int vI, real energy){
 if( (uI>vI)) return 0;
 octreeNode *atmN=&U[uI];
 octreeNode *atmM=&V[vI];
 real Dx=atmN->cx-atmM->cx;
 real Dy=atmN->cy-atmM->cy;
 real Dz=atmN->cz-atmM->cz;
 real RuPlusRv=atmN->cr+atmM->cr;
 real  R2_UV= Dx * Dx + Dy * Dy + Dz * Dz;

 real th=(RuPlusRv*(ob.onePlustwobyEps2));


 
  if(atmN->leaf==true&&atmM->leaf==true){
	 // if( (uI>vI)) return 0;
	 	  
	  for(int u=atmN->atomsStartID;u<=atmN->atomsEndID;u++){
		 Atom *atm1=&(ob.satms[u]);
		  for(int v=atmM->atomsStartID;v<=atmM->atomsEndID;v++){
           Atom *atm2=&(ob.satms[v]);
			 real RuRv=atm1->ra*atm2->ra;
             
			 real dx=atm2->x-atm1->x;
			 real dy=atm2->y-atm1->y;
			 real dz=atm2->z-atm1->z;
			 real r2_uv=dx * dx + dy * dy + dz * dz;
			 if ( ob.useApproxMath )                  
                      energy += ( atm1->q*atm2->q * ob.invSqrt( ( float ) ( r2_uv + RuRv * ob.fastExp( ( float ) -(r2_uv/(4*RuRv) ) ) )));  
			 else 
		     energy=energy+(atm1->q*atm2->q)/sqrt(r2_uv+RuRv*exp(-(r2_uv/(4*RuRv))));
			 			
		  }
		  
	  
	  }
	  
	   if(uI<vI)
		return energy;
	   else return energy*0.5;
 }
  else if(R2_UV>th*th){
	 // if( (uI>vI)) return 0;
 // else if(R_UV-RuPlusRv>(2/epsilon2)*RuPlusRv){
	  int idxi=uI*ob.MEpsilon,idxjj=vI*ob.MEpsilon, idxj; 

	  for(int i=0;i<ob.MEpsilon;i++){
	       
		   if (ob.sQ[idxi]!=0){
		   idxj=idxjj;
		   for(int j=0;j<ob.MEpsilon;j++)
		   {  
			  if(ob.sQ[idxj]!=0){
			   real m=epsilon2Plus1P[i*ob.MEpsilon+j];
			   if ( ob.useApproxMath )                  
                           energy += (ob.sQ[idxi]*ob.sQ[idxj] * ob.invSqrt( ( float ) ( R2_UV + m * ob.fastExp( ( float ) ( - R2_UV/(4*m) ) ) ) ) ); 
			   else 
						   energy=energy+(ob.sQ[idxi]*ob.sQ[idxj])/sqrt(R2_UV+m*exp(-(R2_UV/(4*m))));
		  }
		  idxj++;
		 }
	   }
	   idxi++;
	  }
	  if(uI<vI)
		return energy;
	  else return energy*0.5;
  }
  else if(atmN->leaf==true)
  {  
	real partialE[8];
	cilk_for(int i=0;i<8;i++) partialE[i]=0;
	cilk_for(int i=0;i<8;i++)

	  {   
		  if(atmM->cPtr[i]>=0)
		  {
            partialE[i]=ApproxE_pol(ob,U, uI, V,  atmM->cPtr[i],0);
		  }
	  } 
     real gpol=0;
     for(int i=0;i<8;i++) gpol+=partialE[i];
     return gpol;
	 
  }

  else if(atmM->leaf==true)
  {
	  real partialE[8];
	 cilk_for(int i=0;i<8;i++) partialE[i]=0;
	 cilk_for(int i=0;i<8;i++)
	 
	  {
		  if(atmN->cPtr[i]>=0){
		    partialE[i]=ApproxE_pol(ob,U, atmN->cPtr[i], V,  vI,0);
		  }
	  }
     //return patrialE[atmN->cPtr[i]*vI];
	 real gpol=0;
     for(int i=0;i<8;i++) gpol+=partialE[i];
     return gpol;
	 
  }

  else 
  {   
	real partialE[64];
	cilk_for(int i=0;i<64;i++) partialE[i]=0;
	cilk_for(int i=0;i<8;i++)
	{   
		if(atmN->cPtr[i]>=0)
		cilk_for(int j=0;j<8;j++)
		 
		  {   
			  if(atmM->cPtr[j]>=0){
				partialE[i*8+j]=ApproxE_pol(ob,U, atmN->cPtr[i], V,  atmM->cPtr[j],0);
			  }
		  } 
	  }
	 real gpol=0;
     for(int i=0;i<64;i++) gpol+=partialE[i];

	 /*
	 cilk_for(int i=0;i<64;i=i<<3)
	 {
	   for(int j=i+1;j<i+8;j++)
	   {
	    partialE[i]+=partialE[j];
	   
	   }
	 }
     for(int i=0;i<64;i=i<<3)
	 gpol+=partialE[i];
	 
	 */
     return gpol;
  }
 }
void calculateBornRadi(octree &ob){
 
 real maxBornRadius=1000;
 for(int i=0;i<ob.nStaticAtom;i++)
 {
	 real v=0;
     for(int k=0;k<ob.nMovingAtom;k++)
	 {   
		 real dx=ob.matms[k].rx-ob.satms[i].x;
		 real dy=ob.matms[k].ry-ob.satms[i].y;
		 real dz=ob.matms[k].rz-ob.satms[i].z;

		 real m=(dx)*ob.matms[k].nx+(dy)*ob.matms[k].ny+(dz)*ob.matms[k].nz;
	     real n=dx*dx+dy*dy+dz*dz;
		 v=v+ob.matms[k].w*m/(n*n*n);
	 }
	  if (v>0)
	  ob.bornRadi[i]=pow((v*Inv_4PI),-(.33333333333));
	  else ob.bornRadi[i]=0; 
	  if(ob.satms[i].r>ob.bornRadi[i])
         ob.bornRadi[i]=ob.satms[i].r;
	  else if ( ob.bornRadi[i]> maxBornRadius ) ob.bornRadi[i]= maxBornRadius; 

	
 }
 
}

  real NaiveEPol(octree &ob){
	  real Epol=0;
	  real dx,dy,dz,r2ij,f_ijGB, RiRj;
	  for(int i=0;i<ob.nStaticAtom;i++){
		  for(int j=0;j<ob.nStaticAtom;j++){
			   dx=ob.satms[i].x-ob.satms[j].x;
               dy=ob.satms[i].y-ob.satms[j].y;
               dz=ob.satms[i].z-ob.satms[j].z;
			   r2ij=dx*dx+dy*dy+dz*dz;
			   RiRj=ob.bornRadi[i]*ob.bornRadi[j];
			   f_ijGB=sqrt(r2ij+RiRj*exp(-(r2ij/(4*RiRj))));
			   Epol=Epol+(ob.satms[i].q*ob.satms[j].q)/f_ijGB;

		  }
	  }
   return -Epol*ob.tao/2;
  
  }
