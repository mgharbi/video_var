/* --------------------------------------------------------------------------
 * File:    STWarp_nnf.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-03-17
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/


#include "STWarp.hpp"

template <class T>
int STWarp<T>::getPatchCost(const IVideo &A,const IVideo &B, int i_a, int j_a, int k_a, int i_b, int j_b, int k_b, int cutoff) {
  int ans = 0;
  const unsigned char* pA = A.dataReader();
  const unsigned char* pB = B.dataReader();
  int h       = A.getHeight();
  int w       = A.getWidth();
  int nVoxels = A.voxelCount();
  for (int k = 0; k < params.patchSize_time; ++k) 
      for (int j = 0; j < params.patchSize_space; ++j) 
      {
          for (int i = 0; i < params.patchSize_space; ++i) {
              int indexA = i+i_a + h*(j+j_a + w*(k+k_a));
              int indexB = i+i_b + h*(j+j_b + w*(k+k_b));
              for( int l = 0 ; l < A.channelCount() ; l++ ){
                  double c = pA[indexA+l*nVoxels] - pB[indexB+l*nVoxels];
                  ans += c*c;
              }
          }
          if (ans >= cutoff) { return cutoff; }
      }
  return ans;
}

template <class T>
void STWarp<T>::improve_guess(const IVideo &A,const IVideo &B,int i_a, int j_a, int k_a,
        int &ibest, int &jbest, int& kbest, int &cost,
        int ip, int jp, int kp) {
    int d = getPatchCost(A,B, i_a,j_a,k_a,ip,jp,kp, cost);
    if( d < cost) {
        cost = d;
        ibest = ip;
        jbest = jp;
        kbest = kp;
    }
}

template <class T>
WarpingField<T> STWarp<T>::computeNNF() {

    double ratio = 1;

    int org_h = videoA->getHeight();
    int org_w = videoA->getWidth();
    int nF    = videoA->frameCount();
    int h     = org_h*ratio;
    int w     = org_w*ratio;
    if(h<10){
        h = 10;
    }
    if(w<10){
        w = 10;
    }
    IVideo A(h, w, nF, 3);
    IVideo B(h, w, nF, 3);
    
    VideoProcessing::resize(videoA->extractChannel(0,2), &A);
    VideoProcessing::resize(videoB->extractChannel(0,2), &B);

    Video<int> nnf      = Video<int>(h, w, nF, 3);
    Video<int> bestCost = Video<int>(h, w, nF, 1);

    int* pNNF  = nnf.dataWriter();
    int* pCost = bestCost.dataWriter();

    int nVoxels = A.voxelCount();
    int w_eff   = w - params.patchSize_space+1;
    int h_eff   = h - params.patchSize_space + 1;
    int nF_eff  = nF - params.patchSize_time + 1;

    // Random initialization
    fprintf(stderr,"+ NNF initialization with size %dx%d (original %dx%dx%d)...",h,w,org_h,org_w,nF);
    for (int k = 0; k < nF_eff; ++k) 
        for (int j = 0; j < w_eff; ++j) 
            for (int i = 0; i < h_eff; ++i) 
    {
        int index             = i + h*(j+w*k);
        int i_b               = rand()%h_eff;
        int j_b               = rand()%w_eff;
        int k_b               = rand()%nF_eff;
        pNNF[index]           = i_b;
        pNNF[index+nVoxels]   = j_b;
        pNNF[index+2*nVoxels] = k_b;
        pCost[index]          = getPatchCost(A, B, i,j,k,i_b,j_b,k_b);
    }
    fprintf(stderr, "done.\n");

    for (int iter = 0; iter < params.propagationIterations; ++iter) {
        fprintf(stderr, "  - iteration %d/%d\n", iter+1,params.propagationIterations);
        int xstart = 0, xend = w_eff, xchange = 1;
        int ystart = 0, yend = h_eff, ychange = 1;
        int tstart = 0, tend = nF_eff, tchange = 1;
        if (iter % 2 == 1) {
            xstart = xend-1; xend = -1; xchange = -1;
            ystart = yend-1; yend = -1; ychange = -1;
            tstart = tend-1; tend = -1; tchange = -1;
        }
        for (int k_a = tstart; k_a != tend; k_a += tchange) 
            for (int j_a = xstart; j_a != xend; j_a += xchange) 
                for (int i_a = ystart; i_a != yend; i_a += ychange) 
        {
            int index = i_a + h*(j_a+w*k_a);

            // Current (best) guess
            int ibest    = pNNF[index];
            int jbest    = pNNF[index+nVoxels];
            int kbest    = pNNF[index+2*nVoxels];
            int bc = pCost[index];

            // Propagate x
            if ( j_a - xchange > -1 && j_a - xchange <  w_eff) {
                int index_prev = index - h*xchange;
                int ip = pNNF[index_prev];
                int jp = pNNF[index_prev + nVoxels] + xchange;
                int kp = pNNF[index_prev +2*nVoxels];
                if ( jp> -1 && jp <  w_eff) {
                    improve_guess(A,B,i_a, j_a, k_a,
                            ibest, jbest,kbest, bc,
                            ip, jp, kp);
                }
            }
            // Propagate y
            if (i_a -ychange> -1 && (i_a - ychange) <  h_eff) {
                int index_prev = index - ychange;
                int ip = pNNF[index_prev] + ychange;
                int jp = pNNF[index_prev + nVoxels];
                int kp = pNNF[index_prev +2*nVoxels];
                if ( ip > -1 && ip <  h_eff) {
                    improve_guess(A,B,i_a, j_a, k_a,
                            ibest, jbest,kbest, bc,
                            ip, jp, kp);
                }
            }
            // Propagate t
            if ( (k_a - tchange)>-1 && k_a -tchange <  nF_eff) {
                int index_prev = index - tchange*h*w;
                int ip = pNNF[index_prev];
                int jp = pNNF[index_prev + nVoxels];
                int kp = pNNF[index_prev +2*nVoxels] + tchange;
                if ( kp > -1 && kp <  nF_eff) {
                    improve_guess(A,B,i_a, j_a, k_a,
                            ibest, jbest,kbest, bc,
                            ip, jp, kp);
                }
            }

            /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
            int rs_start = INT_MAX;
            int rt_start = INT_MAX;
            if (rs_start > max(w, h)) { rs_start = max(w, h); }
            if (rt_start > nF) { rt_start = nF; }
            for (int mag = rs_start; mag >= 1; mag /= 2){
                int mag_time = mag;
                /* Sampling window */
                int i_min = max(ibest-mag, 0), i_max = min(ibest+mag+1,h_eff);
                int j_min = max(jbest-mag, 0), j_max = min(jbest+mag+1,w_eff);
                int ip    = i_min+rand()%(i_max-i_min);
                int jp    = j_min+rand()%(j_max-j_min);

                    int k_min = max(k_a-mag_time, 0), k_max = min(k_a+mag_time+1,nF_eff);
                    int kp    = k_min+rand()%(k_max-k_min);
                    improve_guess(A,B,i_a, j_a, k_a,
                            ibest, jbest,kbest, bc,
                            ip, jp, kp);

                // for (int mag_time = rt_start; mag_time >= 1; mag_time /= 2){
                //     int k_min = max(k_a-mag_time, 0), k_max = min(k_a+mag_time+1,nF_eff);
                //     int kp    = k_min+rand()%(k_max-k_min);
                //     improve_guess(A,B,i_a, j_a, k_a,
                //             ibest, jbest,kbest, bc,
                //             ip, jp, kp);
                // }
            }
            pNNF[index]           = ibest;
            pNNF[index+nVoxels]   = jbest;
            pNNF[index+2*nVoxels] = kbest;
            pCost[index]          = bc;
        }
    } // end PM-iteration

    // Convert to relative offset
    vector<int> dims(3);
    dims[0] = org_h;
    dims[1] = org_w;
    dims[2] = nF;
    WarpingField<T> nnf2(h, w,  nF, 3);
    nnf2.copy(nnf);
    resampleWarpingField(nnf2, dims);
    WarpingField<T> ret = WarpingField<T>(org_h, org_w, nF, 3);
    T* pRet = ret.dataWriter();
    const T* pNNF2 = nnf2.dataReader();
    nVoxels = ret.voxelCount();
    for (int k = 0; k < nF_eff; ++k) 
        for (int j = 0; j < w_eff/ratio; ++j) 
            for (int i = 0; i < h_eff/ratio; ++i) 
    {
        int index = i + org_h*(j+org_w*k);
        pRet[index] = pNNF2[index + nVoxels] - j;
        pRet[index + nVoxels] = pNNF2[index] - i;
        pRet[index + 2*nVoxels] = pNNF2[index + 2*nVoxels] - k;
    }

    return ret;
}

#pragma mark - Template instantiations
template class STWarp<float>;
template class STWarp<double>;
