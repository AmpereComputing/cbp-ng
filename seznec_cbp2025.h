

// Developped by A. Seznec
//This simulator is derived from the simulator provided with  the slide set "TAGE: an engineering cookbook by Andre Seznec, November 2024" and the CBP2016 winner as coded in the CBP2025 framework

// The two  initial simulators were  developped to fit CBP2016 framework
// In this file : added Local Prediction + Different threshold
// Added global interleaving of logical TAGE tables


// The  loop predictor is present in the file, but is not enabled
// It brings only marginal accuracy benefit



#ifndef _TAGE_PREDICTOR_H_
#define _TAGE_PREDICTOR_H_
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <array>
#include <iostream>

#include <vector>

using uint = unsigned int;

//To get the predictor storage budget on stderr  uncomment the next line
#define PRINTSIZE

#define LOGSCALE 4
#define LOGT  (7+LOGSCALE)	/* logsize of a logical  TAGE tables */
#define LOGB (11+LOGSCALE)	// log of number of entries in bimodal predictor
#define LOGBIAS (7+ LOGSCALE)  // logsize of tables in SC

#define MINHIST 3
#define MAXHIST 1000
// Did not explore exactly which history length could be the best one, but it seems that 1000 is not bad :-)


//parameters of the loop predictor
#define LOGL 5
#define WIDTHNBITERLOOP 10  // we predict only loops with less than 1K iterations
#define LOOPTAG 10      //tag width in the loop predictor


#define NHIST 28
#define NBANK 28
#define TBITS 14


#define UWIDTH 2
#define LOGASSOC 1 // but partial skewed associativity (option PSK) might be interesting


#define LOGG (LOGT-LOGASSOC) // size of way in a logical TAGE table
#define ASSOC (1<<LOGASSOC)

#define HYSTSHIFT 1 // bimodal hysteresis shared among (1<< HYSTSHIFT) entries
#define BIMWIDTH 3  //  with of the counter in the bimodal predictor
//A. Seznec: I just played using 3-bit counters in the simulator, using 2-bit counters but HYSTSHIFT=0 brings similar accuracy



/////////////////////////////////////////////
//Options  for optimizations of TAGE

int BANK1;
// we implement a banked interleaved predictor: all the logical tagged tables share the whole space


/////////////////////////////////////////////////
// the replacement/allocation policies described in the slide set
#define OPTTAGE
#ifdef OPTTAGE
#define OPTGEOHIST // we can do better than geometric series
#define FILTERALLOCATION 1	
#define FORCEU 1  //don't work if only one U  bit	

#if (LOGASSOC==1)
// A. Seznec: partial skewed associativity, remember that I invented it in 1993 :-)
#define PSK 1
#define REPSK 1 // this optimization is funny, if no "useless" entry, move the entry on the other way to make room,it brings a little bit of accuracy
// for caches it was published as the Elbow cache in 2003, or the Zcache in 2010.
#else
//A. Seznec:  I do not have implemented skewed associativity for more than two-way 
#define PSK 0
#define REPSK 0 
#endif
#define PROTECTRECENTALLOCUSEFUL 1	// Recently allocated entries  are protected against the smart u reset: 
#define UPDATEALTONWEAKMISP 1	// When the Longest match is weak and wrong, one updates also the alternate prediction and HCPred : 

#else

#define PSK 0
#define REPSK 0 
#endif
//////////////////////////////////////////////


/////////////////////////////////////////////
/// For the SC component
#define SC                    // Enables the statistical corrector
#define FORCEONHIGHCONF  //   if TAGE is high conf and SC very low conf then use TAGE
//Add the extra SC tables that uses some information that do not flow directly from TAGE
#define SCMEDIUM  
#ifdef SCMEDIUM
#define SCGH
#define SCIMLI

#define LOCALH // enables local history
#ifdef LOCALH           
//#define LOOPPREDICTOR   //loop predictor enable
//Not enabled
#define LOCALS          //enable the 2nd local history
#define LOCALT          //enables the 3rd local history
#endif
#endif
#define PERCWIDTH 6		//Statistical corrector counter width: 
/////////////////////////////////////////////////








//////////////////////////////////
////////The statistical corrector components

//The base table  in the SC component indexed with only PC + information flowing out from  TAGE
// In order to  allow computing SCSUM in parallel with TAGE check, only LongestMatchPred and HCpred are used. 4 SCSUM are computed, and a final 4-to-1 selects the correct prediction:   each extra bit of information (confidence, etc) would necessitate  doubling the number of computed SCSUMs and double the width of the final MUX

// if only PC-based SC these ones are useful
int8_t BiasGEN;
int8_t BiasAP[2];
int8_t BiasLM[2];
//////

int8_t BiasLMAP[4];
int8_t BiasPC[1 << (LOGBIAS)];
int8_t BiasPCLMAP[(1 << (LOGBIAS))];

#define LOGINB (LOGBIAS)
int Im = LOGBIAS;
int8_t IBIAS[(1 << LOGINB)];
int8_t IIBIAS[(1 << LOGINB)];
int8_t ISBIAS[(1 << LOGINB)];
int8_t IISBIAS[(1 << LOGINB)];


//indices for the  SC tables
#define INDBIASLMAP (LongestMatchPred+ (HCpred<<1))
//#define PSNUM 0
#define PCBL PCBR
#define INDBIASPC (((((PCBL ^(PCBL>>(LOGBIAS-5))))) & ((1<<(LOGBIAS)) -1)) )
#define INDBIASPCLMAP (((INDBIASPC) ^((LongestMatchPred  ^(HCpred <<1))<< (LOGBIAS-2)))) &((1<<(LOGBIAS)) -1) 
// a single  physical table but  two logic tables: indices agree on all the bits except 2





#define INDBIASIMLIBR (((((PCBL ^hist_to_use.F_BrIMLI ^(PCBL>>(LOGBIAS-6))))) & ((1<<LOGINB) -1)) )
#define INDBIASIMLITA ((((((PCBL >> 4) ^hist_to_use.F_TaIMLI ^(PCBL<< (LOGBIAS-4))))) & ((1<<LOGINB) -1)))

#define INDSBIASIMLIBR (((((PCBL  ^ (PCBL >> 6) ^hist_to_use.F_BrSIMLI ^(PCBL>>(LOGBIAS-6))))) & ((1<<LOGINB) -1)) )
#define INDSBIASIMLITA ((((((PCBL >> 5) ^hist_to_use.F_TaSIMLI ^(PCBL<< (LOGBIAS-5))))) & ((1<<LOGINB) -1)))

#define OTHERTABLES
// just use outputs of TAGE
#ifdef OTHERTABLES
int8_t BiasAlt[(1 << 7)];
#define DIFFBANK (HitBank-AltBank)
#define INDBIASALTTAGE ((LongestMatchPred==alttaken) + (TAGECONF<<1)+ ((ALTCONF>=TAGECONF)<<3) + (LongestMatchPred<<4) +  (((DIFFBANK>4) + (DIFFBANK>9)+ (DIFFBANK>13))<<5) )
int8_t BiasBIM[(1 << 6)];
#define INDBIASBIM ((LongestMatchPred==BIMPRED) + ((BIMCONF>=TAGECONF)<<1)+ (((HitBank>0) + (HitBank>7)+ (HitBank>13))<<2)+( (AltBank>0)<<4))+(LongestMatchPred<<5)
int8_t BiasLMHCALT[(1<<9)];// in practice less entries !!
#define INDBIASLMHCALT (LongestMatchPred+ (HCpred<<1)+ (alttaken <<2) + (TAGECONF<< 3) + (ALTCONF <<5) +(HCCONF<<7))
#endif



#define LOGSCALEB (LOGSCALE+1)
// from CBP2016
//first local history
#define LOGLNB  (LOGSCALEB+6)   
#define LNB 4
int Lm[LNB] = { 25, 16, 10, 5 };
int8_t LGEHLA[LNB][(1 << LOGLNB)] = { {0} };

int8_t *LGEHL[LNB];
#define  LOGLOCAL 7
#define NLOCAL (1<<LOGLOCAL)

// second local history:in fact history within the same instruction block of instruction
#define LOGSNB (LOGSCALEB +6)      
#define SNB 4
int Sm[SNB] = {30, 18, 9, 4};
int8_t SGEHLA[SNB][(1 << LOGSNB)] = { {0} };

int8_t *SGEHL[SNB];
#define LOGSECLOCAL 5
#define NSECLOCAL (1<<LOGSECLOCAL)

// third local history 
#define LOGTNB (LOGSCALEB +6)      
#define TNB 3
int Tm[SNB] = {17, 8,3};
int8_t TGEHLA[TNB][(1 << LOGTNB)] = { {0} };

int8_t *TGEHL[TNB];
#define LOGTLOCAL 5
#define NTLOCAL (1<<LOGTLOCAL)


//fourth local history  incorporates 2 bits of global history and the current direction in the "local history"
#define LOGQNB (LOGSCALEB + 6)
#define QNB 3
int Qm[QNB] = {  45, 24, 12 };
int8_t QGEHLA[QNB][(1 << LOGQNB)] = { {0} };
int8_t *QGEHL[QNB];
#define LOGQLOCAL 4
#define NQLOCAL (1<<LOGQLOCAL)  


// Global history tables
//normal global/path history

// History of 256 bytes blocks
#define LOGPNB (LOGSCALEB + 5)
#define PNB 1
int Pm[PNB] = {  12 };
int8_t PGEHLA[PNB][(1 << LOGPNB)] = { {0} };
int8_t *PGEHL[PNB];


#define LOGGNB   (LOGSCALEB + 6) 
#define GNB 5
int Gm[GNB]={53,39,24,11,4};
int8_t GGEHLA[GNB][(1 << LOGGNB)] = { {0} };
int8_t *GGEHL[GNB];

// with LongestMatchPred and HCpred
#define LOGANB  (LOGSCALEB + 6)   
#define ANB 4
int Am[ANB]={31,17,7,2};
int8_t AGEHLA[ANB][(1 << LOGANB)] = { {0} };
int8_t *AGEHL[ANB];


// backward history
#define LOGBNB  (LOGSCALEB + 6)    
#define BNB 1
int Bm[BNB] = {11};
int8_t BGEHLA[BNB][(1 << LOGBNB)] = { {0} };
int8_t *BGEHL[BNB];
// forward history
#define LOGFNB  (LOGSCALEB + 6)
#define FNB 1
int Fm[FNB] = {10};
int8_t FGEHLA[FNB][(1 << LOGFNB)] = { {0} };
int8_t *FGEHL[FNB];



//update threshold for the statistical corrector

//more than one update threshold
#define EXTRAW
#define WIDTHRES 15
#define WIDTHRESP 11

#ifdef EXTRAW
#define LOGSIZEUPS 5 
#else
#define LOGSIZEUPS 0
#endif

int updatethreshold;
int Pupdatethreshold[(1 << LOGSIZEUPS)];
int Supdatethreshold[(1 << LOGSIZEUPS)];
//two tables to save  space


#define INDUPDS ((PCBR ^ (PCBR >>2)) & ((1 << (LOGSIZEUPS)) - 1))
#define INDUPDS0 ((PCBR ^ (PCBR >>5)^ (PCBR>>3)) & ((1 << (LOGSIZEUPS)) - 1))

//provide extra multiplicative weight to emphasise/deemphasize the contribution of IMLI and local components
int8_t WL[(1 << LOGSIZEUPS)];
int8_t WL0[(1 << LOGSIZEUPS)];
int8_t WI[(1 << LOGSIZEUPS)];
int8_t WI0[(1 << LOGSIZEUPS)];

// marginal benefit

int SUMGHIST;
int SUMSC;
int SUMFULL;

bool predTSC;
bool predSC;
bool pred_inter;

////  FOR TAGE //////

#define HISTBUFFERLENGTH 8192	// we use a 8K entries history buffer to store the branch history: 5000 are needed in the simulator + 5 * the number of inflight branches
uint8_t ghist[HISTBUFFERLENGTH];


#define BORNTICK 4096
//for the allocation policy
// utility class for index computation
// this is the cyclic shift register for folding 
// a long global history into a smaller number of bits; see P. Michaud's PPM-like predictor at CBP-1

class folded_history
{
public:


  unsigned comp;
  int CLENGTH;
  int OLENGTH;
  int OUTPOINT;
  int INTEROUT;

  folded_history ()
  {
  }


  void init (int original_length, int compressed_length, int N)
  {
    comp = 0;
    OLENGTH = original_length;
    CLENGTH = compressed_length;
    OUTPOINT = OLENGTH % CLENGTH;


  }

  void update (uint8_t * h, int PT)
  {
    comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];

    comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
    comp ^= (comp >> CLENGTH);
    comp = (comp) & ((1 << CLENGTH) - 1);
  }


};

class bentry			// TAGE bimodal table entry  
{
public:
  int8_t hyst;
  int8_t pred;
  bentry ()
  {
    pred = 0;
    hyst = 1;
  }
};

class gentry			// TAGE global table entry
{
public:
  int8_t ctr;
  uint tag;
  int8_t u;


  gentry ()
  {
    ctr = 0;
    u = 0;
    tag = 0;


  }
};




bool alttaken;			// alternate   TAGE prediction if the longest match was not hitting: needed for updating the u bit
bool HCpred;			// longest not low confident match or base prediction if no confident match

bool tage_pred;			// TAGE prediction
bool LongestMatchPred;
int HitBank;			// longest matching bank
int AltBank;			// alternate matching bank
int HCpredBank;		// longest non weak  matching bank
int HitAssoc;
int AltAssoc;
int HCpredAssoc;
int Seed;			// for the pseudo-random number generator
int8_t BIM; // the bimodal prediction


//Counters to guide allocation/replacement in TAGE
int8_t CountMiss11 = -64;		// more or less than 11% of misspredictions

#define LOGCOUNT 6
#define INDCOUNT ((PCBR ^ (PCBR >>  LOGCOUNT)) & ((1<< LOGCOUNT)-1))
int8_t COUNT50[(1<<LOGCOUNT)][(NHIST/4)+1] = {{7}}; // more or less than 50%  misprediction on weak LongestMatchPred
int8_t COUNT16_31[(1<<LOGCOUNT)][(NHIST/4)+1] = {{7}}; // more or less than 16/31th  misprediction on weak LongestMatchPred
int8_t COUNT8_15[(1<<LOGCOUNT)][(NHIST/4)+1] = {{7}};
int8_t Count50[(NHIST/4)+1]= {7};
int8_t Count16_31[(NHIST/4)+1]= {7};
int8_t Count8_15[(NHIST/4)+1]= {7};


int TAGECONF; // TAGE confidence  from 0 (weak counter) to 3 (saturated)
int ALTCONF;
int HCCONF;
int BIMCONF;
bool BIMPRED;
#define PHISTWIDTH 27		// width of the path history used in TAGE
#define CWIDTH 3		// predictor counter width on the TAGE tagged tables

//the counter(s) to chose between longest match and alternate prediction on TAGE when weak counters: only plain TAGE
#define ALTWIDTH 5
int8_t use_alt_on_na;
int TICK, TICKH;		// for the reset of the u counter
// TICKH  used only if (UWIDTH =1)


class lentry            //loop predictor entry
{
public:
  uint16_t NbIter;        //10 bits
  uint8_t confid;     // 4bits
  uint16_t CurrentIter;       // 10 bits

  uint16_t TAG;           // 10 bits
  uint8_t age;            // 4 bits
  bool dir;           // 1 bit

  //39 bits per entry    
  lentry ()
  {
    confid = 0;
    CurrentIter = 0;
    NbIter = 0;
    TAG = 0;
    age = 0;
    dir = false;
  }
};
//For the TAGE predictor
bentry *btable;			//bimodal TAGE table
gentry *gtable[NHIST + 1];	// tagged TAGE tables
int m[NHIST + 1];
uint GI[NHIST + 1];		// indexes to the different tables are computed only once
uint GGI[ASSOC][NHIST + 1];		// indexes to the different ways  for tables  in TAGE
uint GTAG[NHIST + 1];		// tags for the different tables are computed only once  
int BI;				// index of the bimodal table
bool pred_taken;		// prediction

//COPYPASTE
using tage_index_t = std::array<folded_history, NHIST+1>;
using tage_tag_t = std::array<folded_history, NHIST+1>;


struct cbp_hist_t
{
  // Begin Conventional Histories
     
  //      std::array<uint8_t, HISTBUFFERLENGTH> ghist;No need to checkpoint that
  uint64_t phist;      //path history
  int ptghist;
  tage_index_t ch_i;
  std::array<tage_tag_t, 2> ch_t;

  std::array<long long, NLOCAL> L_shist;
  std::array<long long, NSECLOCAL> S_slhist;
  std::array<long long, NTLOCAL> T_slhist;
  
  std::array<long long, NQLOCAL> Q_slhist;

  int GH; //  another form of path history
  // Needs for computing the "histories" for IMLI and backward/forward histories
  uint64_t  LastBack;
  uint64_t  LastBackPC;

  uint64_t  SLastBack;
  uint64_t  SLastBackPC;
  uint64_t PCBLOCK;
  //////////////////////IMLI RELATED and backward/Forward history////////////////////////////////////
  uint64_t TaIMLI;		// use to monitor the iteration number (based on target locality for backward branches) Region 64 bytes
  uint64_t BrIMLI;		// use to monitor the iteration number (a second version based on backward branch locality)) Region 64 bytes
  uint64_t F_TaIMLI;		// use to monitor the iteration number,BHIST if TaIMLI = 0
  uint64_t F_BrIMLI;		// use to monitor the iteration number (a second version), FHIST if BrIMLI = 0


  uint64_t TaSIMLI;		// use to monitor the iteration number (based on target locality for backward branches) Region 4 bytes
  uint64_t BrSIMLI;		// use to monitor the iteration number (a second version based on backward branch locality)) Region 4 bytes
  uint64_t F_TaSIMLI;		// use to monitor the iteration number,BHIST if TaSIMLI = 0 
  uint64_t F_BrSIMLI;		// use to monitor the iteration number (a second version), FHIST if BrSIMLI = 0 
  uint64_t BHIST; // history of backward taken branches 
  uint64_t FHIST; // history of forward taken branches 
  uint64_t PHIST;// history of 256B current blocks
  uint64_t PrevRegion; // 256B curent block
#ifdef LOOPPREDICTOR
  std::vector<lentry> ltable;
  int8_t WITHLOOP;
#endif
  cbp_hist_t()
  {
#ifdef LOOPPREDICTOR
    ltable.resize(1 << (LOGL));
    WITHLOOP = -1;
#endif
  }
};





int
incval (int8_t ctr)
{
     
  return (2 * ctr + 1);
  //to center the sum
 }




int
predictorsize ()
{
  int STORAGESIZE = 0;
  int inter = 0;
 
 

  STORAGESIZE += NBANK * (1 << LOGG) * (CWIDTH + UWIDTH + TBITS ) * ASSOC;
#ifndef SC
  STORAGESIZE += ALTWIDTH;
  //the use_alt counter
#endif
  STORAGESIZE += (1 << LOGB) + (BIMWIDTH - 1) * (1 << (LOGB - HYSTSHIFT));
  STORAGESIZE += m[NHIST];	//the history bits 
  STORAGESIZE += PHISTWIDTH;	//phist
  STORAGESIZE += 12;		//the TICK counter
if (UWIDTH ==1)  STORAGESIZE += 12;		//the TICKH counter
  
  STORAGESIZE += 3 * 7 * ((NHIST/4) * (1<<LOGCOUNT) +1);		//counters COUNT50 COUNT16_31 COUNT8_31 Count50 Count16_31 Count8_15
STORAGESIZE += 8;		//CountMiss11
STORAGESIZE += 36;		// for the random number generator
fprintf (stderr, " (TAGE %d) ", STORAGESIZE);
#ifdef LOOPPREDICTOR

inter = (1 << LOGL) * (2 * WIDTHNBITERLOOP + LOOPTAG + 4 + 4 + 1);
fprintf (stderr, " (LOOP %d) ", inter);
STORAGESIZE += inter;
#endif
#ifdef SC
  

  

inter = WIDTHRESP * 2 * (1 << LOGSIZEUPS);	//the update threshold counters
 
inter += WIDTHRES;
inter += (PERCWIDTH) *2 * (1 << (LOGBIAS)) ;	// BiasPC and BiasPCLMAP, 
inter += (PERCWIDTH) * 2;	//BiasLMAP
#ifdef OTHERTABLES
 
inter += ((1<<7) + (1<<6) + (1<<9) ) * PERCWIDTH;

#endif
STORAGESIZE += inter;
fprintf(stderr,"(SC %d)", inter); inter =0;
  
#ifdef SCMEDIUM


  
inter +=   (GNB ) * (1 << (LOGGNB)) * (PERCWIDTH);
inter +=   (ANB ) * (1 << (LOGANB)) * (PERCWIDTH);
inter += Gm[0];
inter +=   (BNB ) * (1 << (LOGBNB)) * (PERCWIDTH) ;
inter += Bm[0];
inter +=   (FNB ) * (1 << (LOGFNB)) * (PERCWIDTH);
inter += Fm[0];
inter +=   (PNB ) * (1 << (LOGPNB)) * (PERCWIDTH) ;
 inter += Pm[0];

  
STORAGESIZE += inter;
fprintf(stderr,"(SCGH %d)", inter); inter =0;
//The two forms of IMLI
#ifdef SCIMLI  
inter +=  (1 << LOGINB) * PERCWIDTH;	
inter +=  LOGBIAS;
inter += 10; // LastBackPC
inter +=  (1 << LOGINB) * PERCWIDTH;
inter +=  LOGBIAS;
inter += 10;			// LastBack
//The two other forms of IMLI
inter +=  (1 << LOGINB) * PERCWIDTH;	
inter +=  LOGBIAS;
inter += 14; // SLastBackPC
inter +=  (1 << LOGINB) * PERCWIDTH;
inter +=  LOGBIAS;
inter += 14;			// SLastBack
#ifdef EXTRAW
inter += 2* PERCWIDTH * (1 << LOGSIZEUPS); // the 2 EXTRAW   table multipliers
#endif

STORAGESIZE += inter;
fprintf(stderr,"(IMLI %d)", inter); inter =0;
#endif  



  
#ifdef LOCALH
inter +=
			      (LNB ) * (1 << (LOGLNB)) * (PERCWIDTH);

inter += NLOCAL * Lm[0];

#ifdef LOCALS
inter +=
					     (SNB ) * (1 << (LOGSNB)) * (PERCWIDTH);
inter += NSECLOCAL * (Sm[0]);


#endif
#ifdef LOCALT
 
inter +=
							    (QNB ) * (1 << (LOGQNB)) * (PERCWIDTH) ;
inter += NQLOCAL * Qm[0];
inter +=
									   (TNB ) * (1 << (LOGTNB)) * (PERCWIDTH) ;
inter += NTLOCAL * Tm[0];
  
#endif
    
#ifdef EXTRAW
inter += 2* PERCWIDTH * (1 << LOGSIZEUPS); // the 2 EXTRAW   table multipliers
#endif

STORAGESIZE += inter;
fprintf(stderr,"(LOCAL %d)", inter); inter =0;
#endif


 
#endif
#endif
#ifdef PRINTSIZE


fprintf (stderr, " (TOTAL %d, %d Kbits)\n  ", STORAGESIZE, STORAGESIZE/1024);
//fprintf (stdout, " (TOTAL %d %d Kbits)\n  ", STORAGESIZE, STORAGESIZE/1024);
#endif


return (STORAGESIZE);


}






class CBP2025
{
public:
  int LSUM;

  cbp_hist_t active_hist; // running history always updated accurately
  std::unordered_map<uint64_t /*key*/, cbp_hist_t/*val*/> pred_time_histories;
  // Begin LOOPPREDICTOR State
  bool predloop;  // loop predictor prediction
  int LIB;
  int LI;
  int LHIT;           //hitting way in the loop predictor
  int LTAG;           //tag on the loop predictor
  bool LVALID;        // validity of the loop predictor prediction
  // End LOOPPREDICTOR State
  CBP2025(void)
  {

    reinit (active_hist);
   
#ifdef PRINTSIZE
    predictorsize ();
#endif
  }
  void setup()
  {
  }

  void terminate()
  {
  }
  uint64_t  get_unique_inst_id(uint64_t  seq_no, uint8_t piece) const
  {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0x000F);
  }



  void reinit (cbp_hist_t& current_hist)
  {


    if ((LOGASSOC!=1) || (PSK==0)){
#if (REPSK==1)
                   
      printf ("Sorry REPSK only with associativity 2 and PSK activated\n"); exit(1);
                    
#endif
    }
    LVALID= false;           
       
#ifdef OPTGEOHIST


#if (NHIST==28)
#define NNHIST 36
#endif



    int mm[NNHIST+1] ;   
    mm[1] = MINHIST;
   

    for (int i = 2; i <= NNHIST; i++)
      {
	mm[i] =
	  (int) (((double) MINHIST *
		  pow ((double) (MAXHIST) / (double) MINHIST,
		       (double) (i - 1) / (double) ((NNHIST - 1)))) + 0.5);
      }
    for (int i = 2; i <= NNHIST; i++)
      if (mm[i] <= mm[i - 1] +1)
	mm[i] = mm[i - 1] + 1;

#if (NHIST==28)
    int PT = 1;
    for (int i = 1; i <= 7; i += 2)
      {
	m[PT] = mm[i];
	PT++;

      }
               
    for (int i = 9; i <= 28; i++)

      {
	m[PT] = mm[i];
	PT++;

      }
    PT = NHIST;

    for (int i = NNHIST; i >= 30; i -= 2)
      {
	m[PT] = mm[i];
	PT--;


      }
#endif

#if (NHIST==16)
    int PT = 1;
    for (int i = 1; i <= 5; i += 2)
      {
	m[PT] = mm[i];
	PT++;

      }
               
    for (int i = 7; i < 17; i++)

      {
	m[PT] = mm[i];
	PT++;

      }
    PT = NHIST;

    for (int i = NNHIST; i >= 18; i -= 2)
      {
	m[PT] = mm[i];
	PT--;


      }
#endif

#else
    m[1] = MINHIST;

    for (int i = 2; i <= NHIST; i++)
      {
	m[i] =
	  (int) (((double) MINHIST *
		  pow ((double) (MAXHIST) / (double) MINHIST,
		       (double) (i - 1) / (double) ((NHIST - 1)))) + 0.5);
      }
    for (int i = 3; i <= NHIST; i++)
      if (m[i] <= m[i - 1])
	m[i] = m[i - 1] + 1;
#endif



    for (int i = 1; i <= NHIST; i++)
     
      m[i] *= 5;
    //5 bits per block: 5 is prime with the size of the folded register used in the predictor 
    

    for (int i = 1; i <= NHIST; i++)
      fprintf (stderr,"%d ", m[i]);
    fprintf (stderr,"\n");
    // Tables are interleaved on all the banks
    gtable[1]= new gentry[(1 << (LOGG)) * ASSOC *NBANK];
    for (int i = 2; i <= NHIST; i++)
      gtable[i] = gtable[1];
 


    btable = new bentry[1 << LOGB];
    for (int i = 1; i <= NHIST; i++)
      {
	current_hist.ch_i[i].init (m[i], 23, i - 1);
#if ((TBITS==13) || (TBITS==12))
	current_hist.ch_t[0][i].init (current_hist.ch_i[i].OLENGTH, 13, i);
	current_hist.ch_t[1][i].init (current_hist.ch_i[i].OLENGTH, 11, i + 2);
#endif
#if (TBITS==14) 
	current_hist.ch_t[0][i].init (current_hist.ch_i[i].OLENGTH, 13, i);
	current_hist.ch_t[1][i].init (current_hist.ch_i[i].OLENGTH, 14, i + 2);
#endif
	
	
      }


    current_hist.GH=0; 
    // Needs for computing the "histories" for IMLI and backward/forward histories
    current_hist.LastBack=0;
    current_hist.LastBackPC=0;
    
    //////////////////////IMLI RELATED and backward/Forward history////////////////////////////////////
    current_hist.TaIMLI=0;		// use to monitor the iteration number (based on target locality for backward branches)
    current_hist.BrIMLI=0;		// use to monitor the iteration number (a second version based on backward branch locality))
    current_hist.F_TaIMLI=0;		// use to monitor the iteration number,BHIST if TaIMLI = 0
    current_hist.F_BrIMLI=0;		// use to monitor the iteration number (a second version), FHIST if BrIMLI = 0
    current_hist.BHIST=0;
    current_hist.FHIST=0;
    
    Seed = 0;

    TICK = 0;
    
    Seed = 0;

    for (int i = 0; i < HISTBUFFERLENGTH; i++)
      ghist[0] = 0;
    

    updatethreshold = 23;
               

#ifdef SCMEDIUM
               
    

    for (int j = 0; j < ((1 << LOGINB) - 1); j++)
      {
	if (!(j & 1))
	  {
	    IBIAS[j] = -1;

	  }
      }
    for (int j = 0; j < ((1 << LOGINB) - 1); j++)
      {
	if (!(j & 1))
	  {
	    IIBIAS[j] = -1;

	  }
      }
               

               
#endif

    for (int j = 0; j < (1 << LOGBIAS); j++)
      {
	if (!(j & 1))
	  {
	    BiasPCLMAP[j] = -1;

	  }
      }
               
    
    TICK = 0;

   
    for (int i = 0; i < LNB; i++)
      LGEHL[i] = &LGEHLA[i][0];
    for (int i = 0; i < SNB; i++)
      SGEHL[i] = &SGEHLA[i][0];
    for (int i = 0; i < TNB; i++)
      TGEHL[i] = &TGEHLA[i][0];
   
    
    for (int i = 0; i < LNB; i++)
      for (int j = 0; j < ((1 << LOGLNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      LGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < SNB; i++)
      for (int j = 0; j < ((1 << LOGSNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      SGEHL[i][j] = -1;

	    }
	}
    

    for (int i = 0; i < TNB; i++)
      for (int j = 0; j < ((1 << LOGTNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      TGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i <QNB; i++)
      QGEHL[i] = &QGEHLA[i][0];
    for (int i = 0; i < QNB; i++)
      for (int j = 0; j < ((1 << LOGQNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      QGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i <PNB; i++)
      PGEHL[i] = &PGEHLA[i][0];
    for (int i = 0; i < PNB; i++)
      for (int j = 0; j < ((1 << LOGPNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      PGEHL[i][j] = -1;

	    }
	}


    for (int i = 0; i < GNB; i++)
      GGEHL[i] = &GGEHLA[i][0];
    for (int i = 0; i < ANB; i++)
      AGEHL[i] = &AGEHLA[i][0];
    for (int i = 0; i < FNB; i++)
      FGEHL[i] = &FGEHLA[i][0];
    for (int i = 0; i < BNB; i++)
      BGEHL[i] = &BGEHLA[i][0];
    
    for (int i = 0; i < GNB; i++)
      for (int j = 0; j < ((1 << LOGGNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      GGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < ANB; i++)
      for (int j = 0; j < ((1 << LOGANB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      AGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < FNB; i++)
      for (int j = 0; j < ((1 << LOGFNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      FGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < BNB; i++)
      for (int j = 0; j < ((1 << LOGBNB) - 1); j++)
	{
	  if (!(j & 1))
	    {
	      BGEHL[i][j] = -1;

	    }
	}

  }



  // index function for the bimodal table

  int bindex (uint64_t  PC)
  {
    return ((PC ^ (PC >> LOGB)) & ((1 << (LOGB)) - 1));
  }
  // the index functions for the tagged tables uses path history as in the OBIAS predictor
  //F serves to mix path history: not very important impact

  int F (uint64_t A, int size, int bank)
  {
    int A1, A2;
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << LOGG) - 1));
    A2 = (A >> LOGG);
    if (bank < LOGG)
      A2 = ((A2 << bank) & ((1 << LOGG) - 1)) ^ (A2 >> (LOGG - bank));
    A = A1 ^ A2;
    if (bank < LOGG)
      A = ((A << bank) & ((1 << LOGG) - 1)) ^ (A >> (LOGG - bank));
    return (A);
  }

  // gindex computes a full hash of PC, ghist and phist
  int gindex (unsigned int PC, int bank, uint64_t hist, const tage_index_t& ch_i) 
  {
    uint index;
    int logg = LOGG ;
    int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
    index = PC ^ (PC >> (abs (logg - bank) + 1)) ^
      ch_i[bank].comp ^ (ch_i[bank].comp>> logg) ^
      F (hist, M, bank);
    uint32_t X =
      (index ^ (index >> logg) ^ (index >> 2 * logg)) & ((1 << logg) - 1);



               
    return (X);
  }

  //  tag computation
  
  uint16_t gtag (unsigned int PC, int bank, uint64_t hist,const tage_tag_t& ch0, const tage_tag_t& ch1) 
  
  {
    int tag = PC ^ (PC >> 2);
    int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
    tag = (tag >> 1) ^ ((tag & 1) << 10) ^ F (hist, M, bank);
    tag ^= ch0[bank].comp ^ (ch1[bank].comp << 1);
    tag ^= tag >> TBITS;
    tag ^= (tag >> (TBITS - 2));

    return tag & ((1 << TBITS) - 1);
  }

  // up-down saturating counter
  void ctrupdate (int8_t & ctr, bool taken, int nbits)
  {
    if (taken)
      {
	if (ctr < ((1 << (nbits - 1)) - 1))
	  ctr++;
      }
    else
      {
	if (ctr > -(1 << (nbits - 1)))
	  ctr--;
      }
  }


                
  bool getbim ()
  {
    BIM =
      (btable[BI].pred) ? (btable[BI >> HYSTSHIFT].hyst) : -1 -
      (btable[BI >> HYSTSHIFT].hyst);

    TAGECONF =(btable[BI >> HYSTSHIFT].hyst); // used when OTHERTABLES

    ALTCONF= TAGECONF;
    HCCONF =TAGECONF;

    return (btable[BI].pred !=0);
  }

  void baseupdate (bool Taken)
  {
    int8_t inter = BIM;
    ctrupdate (inter, Taken, BIMWIDTH);
    btable[BI].pred = (inter >= 0);
    btable[BI >> HYSTSHIFT].hyst = (inter >= 0) ? inter : -inter - 1;

  };
  uint32_t MYRANDOM ()
  {

    //This pseudo-random function: just to be sure that the simulator is deterministic
    // results are within +- 0.002 MPKI in average with some larger difference on individual benchmarks
    Seed++;
    Seed +=  active_hist.phist;
    Seed = (Seed >> 21) + (Seed << 11);
    Seed +=  active_hist.ptghist;
    Seed = (Seed >> 10) + (Seed << 22);
    Seed += GTAG[4];
    return (Seed);
  };

  //  TAGE PREDICTION: same code at fetch or retire time but the index and tags must recomputed
  void Tagepred (uint64_t  PC, const cbp_hist_t& hist_to_use)
  //void Tagepred (uint64_t  PC)
  {


    HitBank = 0;
    AltBank = 0;
    HCpredBank = 0;
    
    for (int i = 1; i <= NHIST; i++)
      {
	GI[i] = gindex (PC, i,hist_to_use.phist, hist_to_use.ch_i);
	GTAG[i] = gtag (PC, i, hist_to_use.phist,hist_to_use.ch_t[0], hist_to_use.ch_t[1]);
      }
		    


		    
    BANK1= (PC ^(PC >>4) ^  (hist_to_use.phist & ((1<<m[1])-1))) % NBANK;

    for (int i = 1; i <= NHIST; i++)
      GI[i] +=  ((BANK1+i) % NBANK) * (1<<(LOGG)) ;

    for (int i = 1; i <= NHIST; i++)

	
      GI[i] *= ASSOC;

    BI = PC & ((1 << LOGB) - 1);
    for (int i = 1; i <= NHIST; i++)
      {
	for (int j=0;j<ASSOC; j++) GGI[j][i]= GI[i];
	if (PSK==1){
	  for (int j=1;j<ASSOC;j++)    GGI[j][i]^= ((GTAG[i]) & 0xff) << (1);
	  //skewed associative !!
	}

      }
 


    alttaken = getbim ();
    HCpred = alttaken;
    tage_pred = alttaken;
    LongestMatchPred = alttaken;
    //Look for the bank with longest matching history
    for (int i = NHIST; i > 0; i--)
      {
	for (int j = 0; j < ASSOC; j++)
	  {

	    if (gtable[i][GGI[j][i] + j].tag == GTAG[i])
	      {
		HitBank = i;
		HitAssoc = j;

		LongestMatchPred =
		  (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr >= 0);
		TAGECONF = (abs (2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr + 1)) >> 1;
     
		break;
	      }


	  }
	if (HitBank > 0)
	  break;
      }
    //should be noted that when LongestMatchPred is not low conf then alttaken is the 2nd not-low conf:  not a critical path, needed only on update.
    for (int i = HitBank - 1; i > 0; i--)
      {
	for (int j = 0; j < ASSOC; j++)
	  if (gtable[i][GGI[j][i] + j].tag == GTAG[i])
	    {
	      
	      {
		AltAssoc = j;
		AltBank = i;
		break;
	      }
	    }
	if (AltBank > 0)
	  break;

      }
    if (HitBank>0){
                    
      if ( abs (2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr + 1) ==1)
	{
	  for (int i = HitBank - 1; i > 0; i--)
	    {
	      for (int j = 0; j < ASSOC; j++)
		if (gtable[i][GGI[j][i] + j].tag == GTAG[i])
		  {
		    if (abs (2 * gtable[i][GGI[j][i] + j].ctr + 1) != 1)
		      //slightly better to pick alternate prediction as not low confidence
		      {
			HCpredBank = i;
                                   
			HCpredAssoc = j;
			HCpred= (gtable[i][GGI[j][i] + j].ctr >=0);
			HCCONF= abs (2 * gtable[i][GGI[j][i] + j].ctr + 1)>> 1;
			break;
		      }
		  }
	      if (HCpredBank > 0)
		break;

	    }
	}
     
      else {
	HCpredBank = HitBank;
	HCpredAssoc = HitAssoc;
	HCpred = LongestMatchPred;
        HCCONF= TAGECONF;                 
                                                  
      }
     
    }
               
    //computes the prediction and the alternate prediction

    if (HitBank > 0)
      {
	if (AltBank > 0)
	  {
	    alttaken = (gtable[AltBank][GGI[AltAssoc][AltBank] + AltAssoc].ctr >= 0);
	    ALTCONF= abs (2*gtable[AltBank][GGI[AltAssoc][AltBank] + AltAssoc].ctr+1) >>1;
	  }
                   
     



#ifndef SC
	//if the entry is recognized as a newly allocated entry and 
	//USE_ALT_ON_NA is positive  use the alternate prediction
	bool Huse_alt_on_na = (use_alt_on_na >= 0);

	if ((!Huse_alt_on_na)
	    || (abs (2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr + 1) >
		1))
	  tage_pred = LongestMatchPred;
	else
	  tage_pred = HCpred;

#else
	tage_pred = LongestMatchPred;
#endif
      }

               
               
  }
  uint64_t  get_local_index(uint64_t  PC) const
  {
    return ((PC ^ (PC >>2)) & (NLOCAL-1));
  }

  uint64_t  get_second_local_index(uint64_t  PC) const
  {
    return (((PC ^ (PC >>5))) & (NSECLOCAL-1));
  }

  uint64_t  get_third_local_index(uint64_t  PC) const
  {
    return (((PC ^ (PC >>5))) & (NTLOCAL-1));
  }

  

  uint64_t  get_fourth_local_index(uint64_t  PC) const
  {
    return  (((PC ^ (PC >>(3)))) & (NQLOCAL-1)); // different hash for the history
  }

  //compute the prediction
  bool predict (uint64_t  seq_no, uint8_t piece, uint64_t  PC)
  // checkpoint current hist
  {
    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), active_hist);
    const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, active_hist, true/*pred_time_predict*/);
    return pred_taken;
  }

  bool predict_using_given_hist (uint64_t  seq_no, uint8_t piece, uint64_t  PCBRANCH, const cbp_hist_t& hist_to_use, const bool pred_time_predict)
  {
    // computes the TAGE table addresses and the partial tags
    uint64_t  PCBR= PCBRANCH >>2;;
    Tagepred (PCBR, hist_to_use);
    pred_taken = tage_pred;
    predSC = pred_taken;
    predTSC = pred_taken;

#ifndef SC

    return pred_taken;
#endif

    // Let us  compute the SC prediction
    SUMSC = 0;
    ////// These extra counters seem to bring a marginal  gain of 0.006 MPKI  when only pure SC, not useful when other info
#ifndef SCMEDIUM
    SUMSC += incval(BiasGEN);
    SUMSC+= incval (BiasLM[LongestMatchPred]);
    SUMSC+= incval (BiasAP[HCpred]);
#endif
    //////

               
    SUMSC += incval (BiasLMAP[INDBIASLMAP]);
    // x 2: a little bit better
    SUMSC += 2*incval (BiasPC[INDBIASPC]);
    SUMSC += incval (BiasPCLMAP[INDBIASPCLMAP]);
#ifdef OTHERTABLES
    SUMSC += incval (BiasAlt[INDBIASALTTAGE]);
    SUMSC += incval (BiasBIM[INDBIASBIM]);
    SUMSC += incval (BiasLMHCALT[INDBIASLMHCALT]);
#endif 




    predTSC= (SUMSC>=0);
    // when predTSC is correct we do not allocate any new entry

    uint64_t  PC= PCBR;
    SUMGHIST=0;
#ifdef SCGH
    SUMGHIST += Gpredict (PC, hist_to_use.GH, Gm, GGEHL, GNB, LOGGNB);
    SUMGHIST += Gpredict (PC ^ (LongestMatchPred ^ (HCpred<<1)), hist_to_use.GH, Am, AGEHL, ANB, LOGANB);
    predTSC= (SUMSC+SUMGHIST>=0);
   //This is the contribution of "pure" global history

    SUMGHIST += Gpredict (PC ^ (PC >>3), hist_to_use.FHIST, Fm, FGEHL, FNB, LOGFNB);
    SUMGHIST += Gpredict (PC ^ (PC >>3), hist_to_use.PHIST, Pm, PGEHL, PNB, LOGPNB);
    if ((hist_to_use.TaIMLI <12) && (hist_to_use.BrIMLI <12) && (hist_to_use.TaSIMLI <12) && (hist_to_use.BrSIMLI <12))
      SUMGHIST += Gpredict (PC ^ (PC >> 6), hist_to_use.BHIST, Bm, BGEHL, BNB, LOGBNB);
#endif    
#ifdef SCMEDIUM
    SUMFULL = 0;
#ifdef SCIMLI    

    SUMFULL += incval(IIBIAS[INDBIASIMLIBR]);
    SUMFULL += incval(IBIAS[INDBIASIMLITA]);
    SUMFULL += incval(IISBIAS[INDSBIASIMLIBR]);
    SUMFULL += incval(ISBIAS[INDSBIASIMLITA]);

#endif
#endif



    
    LSUM=0;
               
    
#ifdef LOCALH
    
    
    LSUM += Gpredict (PC, hist_to_use.L_shist[get_local_index(PC)], Lm, LGEHL, LNB, LOGLNB);
#ifdef LOCALS
    LSUM += Gpredict (PC, hist_to_use.S_slhist[get_second_local_index(hist_to_use.PCBLOCK)], Sm, SGEHL, SNB, LOGSNB);
#endif
#ifdef LOCALT
    LSUM += Gpredict (PC, hist_to_use.T_slhist[get_third_local_index(PC)], Tm, TGEHL, TNB, LOGTNB);
    LSUM += Gpredict (PC, hist_to_use.Q_slhist[get_fourth_local_index(PC)], Qm, QGEHL, QNB, LOGQNB);
#endif

#endif


#ifdef EXTRAW
     LSUM = (((1<<PERCWIDTH)+WL[INDUPDS])*((1<<PERCWIDTH)+ WL0[INDUPDS0])*LSUM)>> (2*PERCWIDTH-1);
       SUMFULL = (((1<<PERCWIDTH)+WI[INDUPDS])*((1<<PERCWIDTH)+ WI0[INDUPDS0])*SUMFULL)>> (2*PERCWIDTH-1);

#endif


    SUMSC+= LSUM+ SUMFULL + SUMGHIST;
    bool SCPRED = (SUMSC >= 0);
#ifdef FORCEONHIGHCONF
    int THRES = (Pupdatethreshold[INDUPDS] + Supdatethreshold[INDUPDS0] + updatethreshold);
    pred_taken= (TAGECONF!=3)  || (abs(SUMSC)>=THRES/2) ? SCPRED: LongestMatchPred;
#else    
    pred_taken =(SUMSC>=0);
#endif
#ifdef LOOPPREDICTOR
    predloop = getloop (PC, hist_to_use);   // loop prediction
    pred_taken = ((hist_to_use.WITHLOOP >= 0) && (LVALID)) ? predloop : pred_taken;
#endif
    predSC = (SUMSC>=0);
               
    return pred_taken;

  }

  void history_update (uint64_t  seq_no, uint8_t piece, uint64_t  PC, int brtype, bool taken, uint64_t  nextPC)
  {
    HistoryUpdate (PC, brtype, taken, nextPC);
  }

  void TrackOtherInst (uint64_t  PC, int brtype, bool taken, uint64_t  nextPC)
  {
    HistoryUpdate (PC, brtype, taken, nextPC);
  }

  void HistoryUpdate (uint64_t  PCBRANCH, int brtype, bool taken, uint64_t  branchTarget)
  {


    auto& Y = active_hist.ptghist;

    auto& H = active_hist.ch_i;
    auto& G = active_hist.ch_t[0];
    auto& J = active_hist.ch_t[1];
    auto& X = active_hist.phist;

		
    uint64_t  PCBR = (PCBRANCH>>2);

    if ((taken) || (PCBRANCH > branchTarget)){
	active_hist.GH= ( active_hist.GH << 1)^ PCBR;
    active_hist.GH <<= 1;
    uint64_t  Successor = (branchTarget>>4) ^ branchTarget>>2;
    active_hist.GH ^= Successor;
    }
    else
      {
	active_hist.GH <<= 1;
      }
    if (brtype & 1)
      {
	uint64_t  PC=PCBR;
	active_hist.L_shist[get_local_index(PC)] = (active_hist.L_shist[get_local_index(PC)] << 1) + (taken);
	active_hist.S_slhist[get_second_local_index(active_hist.PCBLOCK)] = ((active_hist.S_slhist[get_second_local_index(active_hist.PCBLOCK)] << 1) + taken);
	active_hist.T_slhist[get_third_local_index(PC)] = ((active_hist.T_slhist[get_third_local_index(PC)] << 1) + taken) ^ (PC & 15);
	active_hist.Q_slhist[get_fourth_local_index(PC)] = (active_hist.Q_slhist[get_fourth_local_index(PC)] << 3) + (((active_hist.GH>>2) & 3)<<1)+ taken;


#ifdef LOOPPREDICTOR
	// only for conditional branch
	if (LVALID)
	  {
	    if (pred_taken != predloop)
	      ctrupdate (active_hist.WITHLOOP, (predloop == pred_taken), 7);
	  }

	loopupdate(PC, pred_taken, false/*alloc*/, active_hist.ltable);
#endif

      }
    if (taken) active_hist.PCBLOCK = branchTarget >>2;
#ifdef SCMEDIUM
    if ((branchTarget >> 8) !=  active_hist.PrevRegion) {active_hist.PHIST= (active_hist.PHIST<<1) ^ (branchTarget>>8); active_hist.PrevRegion = (branchTarget >>8);}
    
    if (taken)
      if (branchTarget > PCBRANCH)
	active_hist.FHIST = (active_hist.FHIST << 3) ^ (branchTarget >> 2) ^ (PCBRANCH >>1);
    // Caution: this  is quite  different from the Micro 2015 paper
    // rely on close targets  and on close  branches : capture loop nests, see Tage Cookbook slide set
#define LOGREGSIZE 6
    //Log of the region size for IMLI    
    if ((brtype & 2) == 0)
      //not indirect or return
      {
	if (taken)
	  {
	    if (branchTarget < PCBRANCH)
	      {
		// allows to  finish an iteration at different points
		if (((branchTarget & 65535)>> LOGREGSIZE) ==active_hist.LastBack)
		  {
		    if (active_hist.TaIMLI < ((1 << LOGBIAS) - 1))
		      active_hist.TaIMLI++;
		  }
		else
		  {   

		    active_hist.TaIMLI = 0;
		  }
		if (((PCBRANCH & 65535)>> LOGREGSIZE) == active_hist.LastBackPC)

		  {
		    if (active_hist.BrIMLI < ((1 << LOGBIAS) - 1))
		      active_hist.BrIMLI++;
		  }
		else
		  {
                                        

		    active_hist.BrIMLI = 0;

		  }
		active_hist.LastBack = (branchTarget & 65535)>> LOGREGSIZE;
		active_hist.LastBackPC = (PCBRANCH & 65535)>> LOGREGSIZE;

		//Small REGSIZE 4 bytes
		// allows to  finish an iteration at different points
		if (((branchTarget & 65535)) ==active_hist.SLastBack)
		  {
		    if (active_hist.TaSIMLI < ((1 << LOGBIAS) - 1))
		      active_hist.TaSIMLI++;
		  }
		else
		  {   
		    active_hist.BHIST = (active_hist.BHIST << 1) ^ active_hist.SLastBack;                                    		    active_hist.TaSIMLI = 0;
		  }
		if (((PCBRANCH & 65535)) == active_hist.SLastBackPC)

		  {
		    if (active_hist.BrSIMLI < ((1 << LOGBIAS) - 1))
		      active_hist.BrSIMLI++;
		  }
		else
		  {
                                        
		    active_hist.BHIST = (active_hist.BHIST << 1) ^ active_hist.SLastBackPC; 
		    active_hist.BrSIMLI = 0;

		  }
		active_hist.SLastBack = (branchTarget & 65535);
		active_hist.SLastBackPC = (PCBRANCH & 65535);
		

	      }
	  }
      }

    else
      { active_hist.TaIMLI = 0; active_hist.BrIMLI = 0; active_hist.TaSIMLI = 0; active_hist.BrSIMLI = 0;
      }
#endif
   
    int T =       PCBR ^ (PCBR >> 2) ^ (PCBR >> 4) ^ (branchTarget>>1) ^ taken;
    int PATH =  ((PCBR ^ (PCBR >> 2)))  ^ (branchTarget >> 3) ;
    active_hist.phist = (active_hist.phist << 5) ^ PATH;
    active_hist.phist = (active_hist.phist & ((1 << 27) - 1));
    for (int t = 0; t < 5; t++)
      {
	int DIR = (T & 1);
	T >>= 1;
	int PATHBIT = PATH;
	PATH >>= 1;
	Y--;
	ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;
	for (int i = 1; i <= NHIST; i++)
	  {

	    H[i].update (ghist, Y);
	    G[i].update (ghist, Y);
	    J[i].update (ghist, Y);
	  }

      }
   

    active_hist.F_TaIMLI = (active_hist.TaIMLI == 0) || (active_hist.BrIMLI==active_hist.TaIMLI) ? (active_hist.BHIST & 63) : active_hist.TaIMLI;
    active_hist.F_BrIMLI = (active_hist.BrIMLI == 0) ? (active_hist.FHIST & 63) : active_hist.BrIMLI;

    active_hist.F_TaSIMLI = (active_hist.TaSIMLI == 0) || (active_hist.BrSIMLI==active_hist.TaSIMLI) ? (active_hist.BHIST & 7) : active_hist.TaSIMLI;
    active_hist.F_BrSIMLI = (active_hist.BrSIMLI == 0) ? (active_hist.FHIST & 7) : active_hist.BrSIMLI;

  }
     
  //END UPDATE  HISTORIES
     

  // PREDICTOR UPDATE

        
  void update (uint64_t  seq_no, uint8_t piece, uint64_t  PC, bool resolveDir, bool predDir, uint64_t  branchTarget)
  {
    const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
    const auto& pred_time_history = pred_time_histories.at(pred_hist_key);
    const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, pred_time_history, false/*pred_time_predict*/);
            
    // remove checkpointed hist
    update(PC, resolveDir, pred_taken, branchTarget, pred_time_history);
    pred_time_histories.erase(pred_hist_key);
  }

  void update (uint64_t  PCBRANCH, bool resolveDir, bool pred_taken, uint64_t  branchTarget, const cbp_hist_t& hist_to_use)
  {
	  


    uint64_t  PCBR = PCBRANCH>>2;
    uint64_t  PC= PCBR; // don't ask why :-)
    

    bool DONE = false;

#ifdef SC

#ifdef LOOPPREDICTOR
    if(pred_taken != resolveDir)  // incorrect loophhist updates in spec_update
      {
	// fix active hist.ltable and active_hist.WITHLOOP
	active_hist.ltable = hist_to_use.ltable;
	active_hist.WITHLOOP = hist_to_use.WITHLOOP;
	if (LVALID)
	  {
	    if (pred_taken != predloop)
	      ctrupdate (active_hist.WITHLOOP, (predloop == resolveDir), 7);
	  }
	loopupdate (PC, resolveDir, (pred_taken != resolveDir), active_hist.ltable);
      }
#endif
	    
    bool SCPRED = (SUMSC >= 0);
    int THRES = (Pupdatethreshold[INDUPDS] + Supdatethreshold[INDUPDS0] + updatethreshold);

											      
    if ((SCPRED != resolveDir) || ((abs (SUMSC) < THRES)))

      {
	if ((LSUM >= 0) != (SUMSC - LSUM >= 0))
	  ctrupdate (WL[INDUPDS], ((LSUM >= 0) == resolveDir),
		     PERCWIDTH);
	if ((LSUM >= 0) != (SUMSC - LSUM >= 0))
	  ctrupdate (WL0[INDUPDS0], ((LSUM >= 0) == resolveDir),
		     PERCWIDTH);
	  
	if ((SUMFULL >= 0) != (SUMSC - SUMFULL >= 0))
	  ctrupdate (WI[INDUPDS], ((SUMFULL >= 0) == resolveDir),
		     PERCWIDTH);
	if ((SUMFULL >= 0) != (SUMSC - SUMFULL >= 0))
	  ctrupdate (WI0[INDUPDS0], ((SUMFULL >= 0) == resolveDir),
		     PERCWIDTH);

	  

#define OPTTHRESUPD
#ifdef OPTTHRESUPD
	if (abs (SUMSC) >= (THRES / 2))
#endif
	  {        
	    if (SCPRED != resolveDir)
	      {
		Pupdatethreshold[INDUPDS] += 1;
		Supdatethreshold[INDUPDS0] += 1;
		updatethreshold += 1;

	      }
	    else // if (abs (SUMSC) < THRES)
	      {
		Pupdatethreshold[INDUPDS] -= 1;
		Supdatethreshold[INDUPDS0] -= 1;
		updatethreshold -= 1;
	      }


	    if (updatethreshold >= (1 << (WIDTHRES)))
	      updatethreshold = (1 << (WIDTHRES)) - 1;
	    if (updatethreshold < 0)
	      updatethreshold = 0;
	    if (Pupdatethreshold[INDUPDS] >= (1 << WIDTHRESP))
	      Pupdatethreshold[INDUPDS] = (1 << WIDTHRESP) - 1;
	    if (Pupdatethreshold[INDUPDS] < 0)
	      Pupdatethreshold[INDUPDS] = 0;
	    if (Supdatethreshold[INDUPDS0] >= (1 << WIDTHRESP))
	      Supdatethreshold[INDUPDS0] = (1 << WIDTHRESP) - 1;
	    if (Supdatethreshold[INDUPDS0] < 0)
	      Supdatethreshold[INDUPDS0] = 0;
	    

	  }


	ctrupdate (BiasGEN, resolveDir, PERCWIDTH);
	ctrupdate (BiasAP[HCpred], resolveDir, PERCWIDTH);
	ctrupdate (BiasLM[LongestMatchPred], resolveDir, PERCWIDTH);
	ctrupdate (BiasLMAP[INDBIASLMAP], resolveDir, PERCWIDTH);

	ctrupdate (BiasPC[INDBIASPC], resolveDir, PERCWIDTH);
	ctrupdate (BiasPCLMAP[INDBIASPCLMAP], resolveDir, PERCWIDTH);

#ifdef OTHERTABLES
	ctrupdate (BiasAlt[INDBIASALTTAGE],  resolveDir, PERCWIDTH);
	ctrupdate(BiasBIM[INDBIASBIM],  resolveDir, PERCWIDTH);
	ctrupdate(BiasLMHCALT[INDBIASLMHCALT],  resolveDir, PERCWIDTH);

#endif 

#ifdef SCMEDIUM

	ctrupdate (IBIAS[INDBIASIMLITA], resolveDir, PERCWIDTH);
	ctrupdate (IIBIAS[INDBIASIMLIBR], resolveDir, PERCWIDTH);
	ctrupdate (ISBIAS[INDSBIASIMLITA], resolveDir, PERCWIDTH);
	ctrupdate (IISBIAS[INDSBIASIMLIBR], resolveDir, PERCWIDTH);



	Gupdate (PC, resolveDir, hist_to_use.GH, Gm, GGEHL, GNB, LOGGNB);
	Gupdate (PC ^ (LongestMatchPred ^ (HCpred<<1)), resolveDir, hist_to_use.GH, Am, AGEHL, ANB, LOGANB);
	
	Gupdate (PC ^ (PC >>3), resolveDir, hist_to_use.FHIST, Fm, FGEHL, FNB, LOGFNB);
	//    if ((hist_to_use.TaIMLI <12) && (hist_to_use.BrIMLI <12) && (hist_to_use.TaSIMLI <12) && (hist_to_use.BrSIMLI <12))
	Gupdate (PC ^ (PC >>3), resolveDir, hist_to_use.PHIST, Pm, PGEHL, PNB, LOGPNB);
	if ((hist_to_use.TaIMLI <12) && (hist_to_use.BrIMLI <12) && (hist_to_use.TaSIMLI <12) && (hist_to_use.BrSIMLI <12))
	  Gupdate (PC ^ (PC >> 6), resolveDir, hist_to_use.BHIST, Bm, BGEHL, BNB, LOGBNB);

#ifdef LOCALH

	Gupdate (PC, resolveDir, hist_to_use.L_shist[get_local_index(PC)], Lm, LGEHL, LNB, LOGLNB);
#endif                       
#ifdef LOCALS
	Gupdate (PC, resolveDir, hist_to_use.S_slhist[get_second_local_index(hist_to_use.PCBLOCK)], Sm,
		 SGEHL, SNB, LOGSNB);
#endif
#ifdef LOCALT

	Gupdate (PC , resolveDir, hist_to_use.T_slhist[get_third_local_index(PC)], Tm, TGEHL, TNB, LOGTNB);
	Gupdate (PC , resolveDir, hist_to_use.Q_slhist[get_fourth_local_index(PC)], Qm, QGEHL, QNB, LOGQNB);
#endif

#endif



      }

#endif

    //TAGE UPDATE
    bool ALLOC = (HitBank < NHIST);
    ALLOC &= (LongestMatchPred!=resolveDir);
    ALLOC &= (predTSC != resolveDir) ;

               

               

    
    //////////////////////////////////////////////////   

    if (HitBank > 0)
      {

	bool PseudoNewAlloc =
	  (abs (2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr + 1) <= 1);
	// an entry is considered as newly allocated if its prediction counter is weak

	if (PseudoNewAlloc)
	  {

#ifndef SC
	    if (LongestMatchPred == resolveDir)
	      ALLOC = false;

	    if (LongestMatchPred != HCpred)
	      {

		ctrupdate (use_alt_on_na, (HCpred == resolveDir), ALTWIDTH);
		// pure TAGE only

	      }
#endif
	  }
      }

               
    /////////////////////////
#ifdef FILTERALLOCATION
    //filter allocations: all of this is done at update, not on the critical path
    // try to evaluate if the misprediction rate is above 1/9th

    if ((tage_pred != resolveDir) || ((MYRANDOM () & 31) <4))
      ctrupdate (CountMiss11, (tage_pred != resolveDir), 8);
               

    if (HitBank > 0)
      {
	bool PseudoNewAlloc = (TAGECONF==0);
                    
                    
	if (PseudoNewAlloc)
	  {
	    // Here we count correct/wrong weak counters to guide allocation
	    for (int i= HitBank/4; i<=NHIST/4;i++)
	      {

		ctrupdate (COUNT50[INDCOUNT][i], (resolveDir == LongestMatchPred), 7);
		ctrupdate (Count50[i], (resolveDir == LongestMatchPred), 7);
		// more or less than 50 % good predictions on weak counters
		if ((LongestMatchPred != resolveDir) || ((MYRANDOM () & 31) > 1))
		  {
		    ctrupdate (COUNT16_31[INDCOUNT][i], (resolveDir == LongestMatchPred), 7);ctrupdate (Count16_31[i], (resolveDir == LongestMatchPred), 7);
		  }
		// more or less than 16/31  good predictions on weak counters
		if ((LongestMatchPred != resolveDir) || ((MYRANDOM () & 31) > 3))
		  {
		    ctrupdate (COUNT8_15[INDCOUNT][i], (resolveDir == LongestMatchPred), 7);ctrupdate (Count8_15[i], (resolveDir == LongestMatchPred), 7);
		  }
		// more or less than 16/31  good predictions on weak counters
	      }
                         
	  }


      }
    //  when allocating a new entry is unlikely to result in a good prediction, rarely allocate    
    if (TAGECONF<2){
      if ((COUNT50[INDCOUNT][(HitBank+1)/4] < 0) & (Count50[(HitBank+1)/4] < 0))
	{ ALLOC &= ((MYRANDOM () & ((1 << (4-TAGECONF)) - 1))==0); }
      else 
	//the future allocated entry is not that likely to be correct
	if ((COUNT16_31[INDCOUNT][(HitBank+1)/4] < 0) & (Count16_31[(HitBank+1)/4]<0)){
	  ALLOC &= ((MYRANDOM () & ((1 << (2-TAGECONF)) - 1)) == 0);
	}
	else
	  if ((COUNT8_15[INDCOUNT][(HitBank+1)/4] < 0) & (Count8_15[(HitBank+1)/4]<0)){
	    ALLOC &= ((MYRANDOM () & ((1 << (1-TAGECONF)) - 1)) == 0);
            
	  }
    }
    // The benefit is essentially to decrease the number of allocations
#endif
               
    if (ALLOC)
      {

	int MaxNALLOC = (TAGECONF) +  (! ((COUNT50[INDCOUNT][(HitBank+1)/4] < 0) & (Count50[(HitBank+1)/4] < 0))) + (!((COUNT16_31[INDCOUNT][(HitBank+1)/4] < 0) & (Count16_31[(HitBank+1)/4]<0))) + (!((COUNT8_15[INDCOUNT][(HitBank+1)/4] < 0) & (Count8_15[(HitBank+1)/4]<0)));


	int NA = 0;
	int DEP = HitBank + 1;
	int Penalty = 0;
	DEP += ((MYRANDOM () & 1) == 0);
	DEP += ((MYRANDOM () & 3) == 0);
	if (DEP == HitBank)
	  DEP = HitBank + 1;

	bool First = true;
	bool Test = false;
                    
	for (int i = DEP; i <= NHIST; i++)
	  {
	    bool done = false;
	    uint j = (MYRANDOM () % ASSOC);
	    {
	      bool REP[2]= {false};
	      int IREP[2]={0};
	      bool MOVE[2]= {false};
	      for (int J = 0; J < ASSOC; J++)
		{
		  j++;
		  j = j % ASSOC;
		  if (gtable[i][GGI[j][i] + j].u == 0)  {
		    REP[j]= true;
		    IREP[j]= GGI[j][i] + j;
		  }
		  else if (REPSK==1)
		    {
		      if (PSK==1)
			IREP[j] = (GGI[j][i] ^ (((gtable[i][GGI[j][i] + j].tag ) & 0xff) << (1))) + (j ^1) ;
		      REP[j]  =   (gtable[i][IREP[j]].u == 0);
		      MOVE[j]= true;
		    }

                         
		  if (REP[j])
                              
		    if (
			((UWIDTH == 1) && ((((MYRANDOM () & ((1 << (abs (2 * gtable[i][GGI[j][i] + j].ctr + 1) >> 1)) - 1)) == 0)))||
			 (TICKH >= BORNTICK / 2))
			||(UWIDTH == 2))
		      {
			done = true;
			if (MOVE[j])
			  {    gtable[i][IREP[j]].u = gtable[i][GGI[j][i] + j].u;
			    gtable[i][IREP[j]].tag=   gtable[i][GGI[j][i] + j].tag ;
			    gtable[i][IREP[j]].ctr = gtable[i][GGI[j][i] + j].ctr;
                                     
                                     
			  }
                                   
                                        
                                   
			gtable[i][GGI[j][i] + j].tag = GTAG[i];
#ifndef FORCEU
			gtable[i][GGI[j][i] + j].u = 0;
#else

			gtable[i][GGI[j][i] + j].u =  ((UWIDTH ==2) || (TICKH >= BORNTICK/2)) & (First ? 1: 0);
#endif
			gtable[i][GGI[j][i] + j].ctr = (resolveDir) ? 0 : -1;
                                   

                                  
                                   
			NA++;
			if ((i >= 3) || (!First))   MaxNALLOC--;
			First = false;
			i += 2;
			i -= ((MYRANDOM () & 1) == 0);
			i += ((MYRANDOM () & 1) == 0);
			i += ((MYRANDOM () & 3) == 0);
			break;

		      }
		}
	      if (MaxNALLOC < 0)
		break;
	      if (!done)
		{
#ifdef FORCEU

		  for (int j = 0; j < ASSOC; j++)
		    {
		      {
			// some just allocated entries  have been set to useful
			if ((MYRANDOM () & ((1<<(1+LOGASSOC+REPSK))-1))==0)
			  if (abs (2 * gtable[i][GGI[j][i] + j].ctr + 1) == 1)
			    if (gtable[i][GGI[j][i] + j].u ==1)
			      gtable[i][GGI[j][i] + j].u--;
			if (REPSK==1)
			  if ((MYRANDOM () & ((1<<(1+LOGASSOC+REPSK))-1))==0)
			    if (abs (2 * gtable[i][IREP[j]].ctr + 1) == 1)
			      if (gtable[i][IREP[j]].u ==1)
				gtable[i][IREP[j]].u--;
                                                            

		      }

		    }
#endif
		  Penalty++;
		}
            
	    }

	    //////////////////////////////////////////////               
	  }

	// we set two counts to monitor: "time to reset u" and "almost time reset u": TICKH is useful only if UWIDTH =1        
#ifndef PROTECTRECENTALLOCUSEFUL
	TICKH += Penalty -  NA;
	TICK += Penalty - 2 * NA;
#else
	TICKH += 2*Penalty - 3*NA;
	TICK += Penalty - (2+ 2*(CountMiss11>=0))*NA;
#endif
	if (TICKH < 0)
	  TICKH = 0;
	if (TICKH >= BORNTICK)
	  TICKH = BORNTICK;





	if (TICK < 0)
	  TICK = 0;
	if (TICK >= BORNTICK)
	  {
	    for (int j = 0;
		 j < ASSOC * (1 << LOGG)*NBANK; j++)
	      {
		//this is not realistic: in a real processor:    gtable[1][j].u >>= 1;  
		if (gtable[1][j].u > 0)
		  gtable[1][j].u--;
	      }
	    TICK = 0;
	    TICKH = 0;
	  }



      }

    //update TAGE predictions

               

    if (HitBank > 0)
      {
#ifdef UPDATEALTONWEAKMISP
	// This protection, when prediction is low confidence 
	if (TAGECONF == 0)
	  {
	    if (LongestMatchPred != resolveDir)
	      {
		if (AltBank !=HCpredBank)  ctrupdate (gtable[AltBank][GGI[AltAssoc][AltBank] + AltAssoc].ctr,
						      resolveDir, CWIDTH);
		if (HCpredBank > 0)
		  {
		    ctrupdate (gtable[HCpredBank][GGI[HCpredAssoc][HCpredBank] + HCpredAssoc].ctr,
			       resolveDir, CWIDTH);

		  }

		else
		  baseupdate (resolveDir);
	      }
	  }
          
#endif
	ctrupdate (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr,
		   resolveDir, CWIDTH);

      }
    else
      baseupdate (resolveDir);
    ////////: note that here it is alttaken that is used: the second hitting entry

    if (LongestMatchPred != alttaken)
      {

	if (LongestMatchPred == resolveDir)
	  {
#ifdef PROTECTRECENTALLOCUSEFUL
	    //if (TICKH >= BORNTICK)
	    if (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u == 0)
	      gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u++;
	    // Recent useful will survive one smart reset
#endif
	    if (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u < (1 << UWIDTH) - 1)
	      gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u++;

	  }
	else
	  {
	    if (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u > 0)
	      {
		if (predSC == resolveDir) 
		  gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u--;;

	      }

	  }
      }
    
     
   
  }


#define GINDEX (((long long) PC) ^ bhist ^ (bhist >> (8 - i)) ^ (bhist >> (16 - 2 * i)) ^ (bhist >> (24 - 3 * i)) ^ (bhist >> (32 - 3 * i)) ^ (bhist >> (40 - 4 * i))^(bhist >> (48 - 4 * i))) & ((1 << (logs /*- (i >= (NBR - 2))*/)) - 1)
  int Gpredict (uint64_t  PC, uint64_t BHIST, int *length, int8_t ** tab, int NBR, int logs)
  {
    int PERCSUM = 0;
    for (int i = 0; i < NBR; i++)
      {
	uint64_t bhist = BHIST & ((long long) ((1 << length[i]) - 1));
	uint64_t index = GINDEX;

	int8_t ctr = tab[i][index];

	PERCSUM += (2 * ctr + 1);
      }
    return ((PERCSUM));
  }
  void Gupdate (uint64_t  PC, bool taken, uint64_t BHIST, int *length,
                int8_t ** tab, int NBR, int logs)
  {



    for (int i = 0; i < NBR; i++)
      {
	uint64_t bhist = BHIST & ((long long) ((1 << length[i]) - 1));
	uint64_t index = GINDEX;


	ctrupdate (tab[i][index], taken, PERCWIDTH);
      }

  }

#ifdef LOOPPREDICTOR
  int lindex (uint64_t PC)
  {
    return (((PC ^ (PC >> 2)) & ((1 << (LOGL - 2)) - 1)) << 2);
  }


  //loop prediction: only used if high confidence
  //skewed associative 4-way
  //At fetch time: speculative
#define CONFLOOP 15
  bool getloop (uint64_t PC, const cbp_hist_t& hist_to_use)
  {
    const auto& ltable = hist_to_use.ltable;
    LHIT = -1;

    LI = lindex (PC);
    LIB = ((PC >> (LOGL - 2)) & ((1 << (LOGL - 2)) - 1));
    LTAG = (PC >> (LOGL - 2)) & ((1 << 2 * LOOPTAG) - 1);
    LTAG ^= (LTAG >> LOOPTAG);
    LTAG = (LTAG & ((1 << LOOPTAG) - 1));

    for (int i = 0; i < 4; i++)
      {
	int index = (LI ^ ((LIB >> i) << 2)) + i;

	if (ltable[index].TAG == LTAG)
	  {
	    LHIT = i;
	    LVALID = ((ltable[index].confid == CONFLOOP)
		      || (ltable[index].confid * ltable[index].NbIter > 128));


	    if (ltable[index].CurrentIter + 1 == ltable[index].NbIter)
	      return (!(ltable[index].dir));
	    return ((ltable[index].dir));

	  }
      }

    LVALID = false;
    return (false);
  }



  void loopupdate (uint64_t PC, bool Taken, bool ALLOC, std::vector<lentry>& ltable)
  {
    if (LHIT >= 0)
      {
	int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
	//already a hit 
	if (LVALID)
	  {
	    if (Taken != predloop)
	      {
		// free the entry
		ltable[index].NbIter = 0;
		ltable[index].age = 0;
		ltable[index].confid = 0;
		ltable[index].CurrentIter = 0;
		return;

	      }
	    else if ((predloop != tage_pred) || ((MYRANDOM () & 7) == 0))
	      if (ltable[index].age < CONFLOOP)
		ltable[index].age++;
	  }

	ltable[index].CurrentIter++;
	ltable[index].CurrentIter &= ((1 << WIDTHNBITERLOOP) - 1);
	//loop with more than 2** WIDTHNBITERLOOP iterations are not treated correctly; but who cares :-)
	if (ltable[index].CurrentIter > ltable[index].NbIter)
	  {
	    ltable[index].confid = 0;
	    ltable[index].NbIter = 0;
	    //treat like the 1st encounter of the loop 
	  }
	if (Taken != ltable[index].dir)
	  {
	    if (ltable[index].CurrentIter == ltable[index].NbIter)
	      {
		if (ltable[index].confid < CONFLOOP)
		  ltable[index].confid++;
		if (ltable[index].NbIter < 3)
		  //just do not predict when the loop count is 1 or 2     
		  {
		    // free the entry
		    ltable[index].dir = Taken;
		    ltable[index].NbIter = 0;
		    ltable[index].age = 0;
		    ltable[index].confid = 0;
		  }
	      }
	    else
	      {
		if (ltable[index].NbIter == 0)
		  {
		    // first complete nest;
		    ltable[index].confid = 0;
		    ltable[index].NbIter = ltable[index].CurrentIter;
		  }
		else
		  {
		    //not the same number of iterations as last time: free the entry
		    ltable[index].NbIter = 0;
		    ltable[index].confid = 0;
		  }
	      }
	    ltable[index].CurrentIter = 0;
	  }

      }
    else if (ALLOC)

      {
	uint64_t X = MYRANDOM () & 3;

	if ((MYRANDOM () & 3) == 0)
	  for (int i = 0; i < 4; i++)
	    {
	      int loop_hit_way_loc = (X + i) & 3;
	      int index = (LI ^ ((LIB >> loop_hit_way_loc) << 2)) + loop_hit_way_loc;
	      if (ltable[index].age == 0)
		{
		  ltable[index].dir = !Taken;
		  // most of mispredictions are on last iterations
		  ltable[index].TAG = LTAG;
		  ltable[index].NbIter = 0;
		  ltable[index].age = 7;
		  ltable[index].confid = 0;
		  ltable[index].CurrentIter = 0;
		  break;

		}
	      else
		ltable[index].age--;
	      break;
	    }
      }
  }
#endif    // LOOPPREDICTOR

  

};


#endif
#undef UINT64

//static CBP2025 cbp2025;
