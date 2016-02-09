/* --------------------------------------------------------------------------
 * File:    warpingIterationAux.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Implementation belonging to the STWarp class. Compute side terms needed for
 * the warping iteration.
 * 
 * ------------------------------------------------------------------------*/


#include "mcp/STWarp.hpp"
#include <cmath>


template <class T>
void STWarp<T>::computeSmoothCost( const Video<T> &warpDX,
                                   const Video<T> &warpDY,
                                   const Video<T> &warpDT,
                                   const Video<T> &warpField,
                                   Video<T> &smoothCost,
                                   Video<T> &lapl)
{
    double eps     = params.eps; eps *= eps;
    int height     = dimensions[0];
    int width      = dimensions[1];
    int nFrames    = dimensions[2];
    int nVoxels    = height*width*nFrames;

    // Compute local smoothness
    Video<T> brightness(videoA->size());
    brightness.copy(*videoA);
    brightness.collapse();
    Video<T> localSmoothness(brightness.size());
    Video<T> d(brightness.size());

    VideoProcessing::dx(brightness,d,false);
    d.multiply(d);
    localSmoothness.add(d);

    VideoProcessing::dy(brightness,d,false);
    d.multiply(d);
    localSmoothness.add(d);

    VideoProcessing::dt(brightness,d,false);
    d.scalarMultiply(1/params.c);
    d.multiply(d);
    localSmoothness.add(d);

    double sig = params.localSmoothnessPrior;
    sig *= sig;

    // T factor = 1/(sig * sqrt(2*M_PI));
    T* pLocSmooth = localSmoothness.dataWriter();
    for (int i = 0; i < localSmoothness.elementCount(); ++i) {
        pLocSmooth[i] = 1 + params.localSmoothness*
            exp(-pLocSmooth[i]/(2*sig));
    }

    T* pSmoothCost = smoothCost.dataWriter();

    const T* pUx  = warpDX.dataReader();
    const T* pUy  = warpDY.dataReader();
    const T* pUt  = warpDT.dataReader();
    int index = 0;
    T value = 0;
        for(int k=0;k<nFrames;k++)
            for(int j=0;j<width;j++)
                for(int i=0;i<height;i++)
        {
            index = i + height*( j + width*k );
            // Ux Uy Ut | Vx Vy Vt | Wx Wy Wt
            for(int chan = 0; chan <3; chan++){
                pSmoothCost[index+(3*chan)*nVoxels] = 
                     pUx[index+chan*nVoxels]*pUx[index+chan*nVoxels];
                pSmoothCost[index+(3*chan+1)*nVoxels] =
                     pUy[index+chan*nVoxels]*pUy[index+chan*nVoxels];
                pSmoothCost[index+(3*chan+2)*nVoxels] =
                     pUt[index+chan*nVoxels]*pUt[index+chan*nVoxels];
            }

            switch( params.decoupleRegularization ){
                case 0 :
                    // Sum_U,V,W rho(Ux2+Uy2+Ut2)
                    for (int comp = 0; comp < 3; ++comp) { // U,V,W
                        value = 0;
                        for(int c = 0;c<3; c++){
                            value += pSmoothCost[index+(3*comp+c)*nVoxels];
                        }
                        for(int c = 0;c<3; c++){
                            pSmoothCost[index+(3*comp+c)*nVoxels] = value;
                        }
                    }
                    break;
                case 1 :
                    // Sum_x,y,t rho(Ux2+Vx2+Wx2)
                    for (int c = 0; c < 3; ++c) { // x,y,t
                        value = 0;
                        for(int comp = 0;comp<3; comp++){
                            value += pSmoothCost[index+(3*comp+c)*nVoxels];
                        }
                        for(int comp = 0;comp<3; comp++){
                            pSmoothCost[index+(3*comp+c)*nVoxels] = value;
                        }
                    }
                    break;
                case 2 :
                    for (int comp = 0; comp < 3; ++comp) { // U,V,W
                        // Sum_U,V,W rho(Ux2+Uy2)+rho(Ut2)
                        value = 0;
                        for(int c = 0;c<2; c++){
                            value += pSmoothCost[index+(3*comp+c)*nVoxels];
                        }
                        for(int c = 0;c<2; c++){
                            pSmoothCost[index+(3*comp+c)*nVoxels] = value;
                        }
                    }
                    break;
                case 3 :
                    // rho(Sum_U,V,W Ux2+Uy2+Ut2)
                    value = 0;
                    for(int c = 0;c<9; c++){
                        value += pSmoothCost[index+(c)*nVoxels];
                    }
                    for(int c = 0;c<9; c++){
                        pSmoothCost[index+(c)*nVoxels] = value;
                    }
                    break;
                case 4 :
                    //  rho(Sum_U,V,W Ux2+Uy2)+rho(Sum_U,V,W Ut2)
                    value = 0;
                    for (int comp = 0; comp < 3; ++comp) { // U,V,W
                        for(int c = 0;c<2; c++){
                            value += pSmoothCost[index+(3*comp+c)*nVoxels];
                        }
                    }
                    for (int comp = 0; comp < 3; ++comp) { // U,V,W
                        for(int c = 0;c<2; c++){
                            pSmoothCost[index+(3*comp+c)*nVoxels] = value;
                        }
                    }
                    value = 0;
                    for (int comp = 0; comp < 3; ++comp) { // U,V,W
                            value += pSmoothCost[index+(3*comp+2)*nVoxels];
                    }
                    for (int comp = 0; comp < 3; ++comp) { // U,V,W
                            pSmoothCost[index+(3*comp+2)*nVoxels] = value;
                    }
                    break;
                default:
                    // rho(Ux2)+rho(Uy2)+rho(Ut2)+rho(Vy2)+...
                    break;
            }
            for(int smoothChan = 0; smoothChan < 9; smoothChan++){
                value = pSmoothCost[index+smoothChan*nVoxels];
                value = 0.5/sqrt(eps+value);
                pSmoothCost[index+smoothChan*nVoxels] = value;
                // Use local smoothness weighting
                // pSmoothCost[index+smoothChan*nVoxels] *= pLocSmooth[index];
            }

            // NOTE: L2 regularization on w instead of charbonnier
            // for(int smoothChan = 6; smoothChan < 9; smoothChan++){
            //     value = 0.5/sqrt(eps);
            // }
        } // end of smoothness cost computation

    // Laplacian of the current flowField
    weightedLaplacian(warpField,smoothCost,lapl);
}

template <class T>
void STWarp<T>::computeOcclusion( const Video<T> &warpField,
                                  const Video<T> &C,
                                  Video<T> &occlusion){

    Video<T> u = warpField.extractChannel(0);
    Video<T> v = warpField.extractChannel(1);
    Video<T> w = warpField.extractChannel(1);
    occlusion = Video<T>(u.size());
    if(!params.useOcclusion) {
        return;
    }
    Video<T> div(u.size());
    VideoProcessing::dx(u,div,false);

    Video<T> divY(v.size());
    VideoProcessing::dy(v,divY,false);
    div.add(divY);

    Video<T> divT(v.size());
    VideoProcessing::dt(w,divT,false);
    div.add(divT);

    Video<T> proj = C.extractChannel(0);


    double sig1 = params.divPrior; sig1 *= sig1;
    double sig2 = params.occlusionPrior; sig2 *= sig2;
    T* pO       = occlusion.dataWriter();
    T* pD       = div.dataWriter();
    const T* pC = proj.dataReader();
    for (int i = 0; i < occlusion.voxelCount(); ++i) {
        if(pD[i]>0){
            pD[i] = 0;
        }
        pO[i]  = exp(-pD[i]*pD[i]/(2*sig1));
        pO[i] *= exp(-pC[i]*pC[i]/(2*sig2));
    }

    T maxOcc = occlusion.max(-1);
    if(maxOcc>0){
        occlusion.scalarMultiply(1/maxOcc);
    }
}

template <class T>
void STWarp<T>::computeDataCost( const Video<T> &Bx,
                                 const Video<T> &By,
                                 const Video<T> &Bt,
                                 const Video<T> &C,
                                 const Video<T> &dWarpField,
                                 const Video<T> &occlusion,
                                 Video<T> &dataCost)
{
    double eps     = params.eps; eps *= eps;
    int height     = dimensions[0];
    int width      = dimensions[1];
    int nFrames    = dimensions[2];
    int nChannels  = videoA->channelCount();
    int nVoxels    = height*width*nFrames;

    if( params.useFeatures ){
        nChannels -= 3;
    }

    T* pDataCost   = dataCost.dataWriter();

    // const T* pOcc  = occlusion.dataReader();
    const T* pC  = C.dataReader();
    // const unsigned char* pM  = maskA->dataReader();
    const T* pBx = Bx.dataReader();
    const T* pBy = By.dataReader();
    const T* pBt = Bt.dataReader();
    const T* pU  = dWarpField.channelReader(0);
    const T* pV  = dWarpField.channelReader(1);
    const T* pW  = dWarpField.channelReader(2);
    // T occSig = params.occlusionPrior;
    for(int i=0;i<nVoxels;i++){
        T value = 0;
        for(int l=0;l<nChannels;l++) {
        int index = i + nVoxels*l ;
            T v = pC[index] - pBx[index]*pU[i]
                            - pBy[index]*pV[i]
                            - pBt[index]*pW[i];
            v *= v;
            value += v;
        }
        pDataCost[i] = 0.5/sqrt(value+eps);
        // if (pOcc[i]>.5) {
        // } else {
        //     pDataCost[index] = 0;
        // }
        // if(pM[i]==0) {
        //     pDataCost[i] = 0;
        // }
        // pDataCost[index] *= pOcc[i];
        // pDataCost[index] += (1-pOcc[i])*0.5/sqrt(10*10+eps);
    } // end of data cost computation

    if(!params.useFeatures) {
        return;
    }

    // Gradient data term
    for(int i=0;i<nVoxels;i++){
        T value = 0;
        for(int l=nChannels;l<nChannels+3;l++) {
            int index = i + nVoxels*l ;
            T v = pC[index] - pBx[index]*pU[i]
                            - pBy[index]*pV[i]
                            - pBt[index]*pW[i];
            v *= v;
            value += v;
        }
        pDataCost[i+nVoxels] += 0.5/sqrt(value+eps);
    } // end of data cost computation
}

template <class T>
void STWarp<T>::prepareLinearSystem( const Video<T> &Bx,
                          const Video<T> &By,
                          const Video<T> &Bt,
                          const Video<T> &C,
                          const Video<T> &lapl,
                          const Video<T> &dataCost,
                          Video<T> &CBx,
                          Video<T> &CBy,
                          Video<T> &CBt,
                          Video<T> &Bx2,
                          Video<T> &By2,
                          Video<T> &Bt2,
                          Video<T> &Bxy,
                          Video<T> &Bxt,
                          Video<T> &Byt
                          )
{
    double eps     = params.eps; eps *= eps;
    int height     = dimensions[0];
    int width      = dimensions[1];
    int nFrames    = dimensions[2];
    int nVoxels    = height*width*nFrames;

    // System Components:
    // A = [Bx2 Bxy Bxt; Bxy By2 Byt; Bxt Byt Bt2]
    // b = [CBx - lapl(u); CBy + lapl(v); CBt + lapl(w)]
    T values[9], valuesGradient[9];
    T* pCBx = CBx.dataWriter();
    T* pCBy = CBy.dataWriter();
    T* pCBt = CBt.dataWriter();
    T* pBx2 = Bx2.dataWriter();
    T* pBy2 = By2.dataWriter();
    T* pBt2 = Bt2.dataWriter();
    T* pBxy = Bxy.dataWriter();
    T* pBxt = Bxt.dataWriter();
    T* pByt = Byt.dataWriter();
    const T* pC  = C.dataReader();
    const T* pBx = Bx.dataReader();
    const T* pBy = By.dataReader();
    const T* pBt = Bt.dataReader();
    const T* pDataCost   = dataCost.dataReader();
    const T* pLapl       = lapl.dataReader();
    int nChannels  = Bx.channelCount();

    if(params.useFeatures){
        nChannels -= 3;
    }
    for(int i=0;i<nVoxels;i++)
    {
        for( int v = 0; v<9 ; v++) { 
            values[v] = 0;
            valuesGradient[v] = 0;
        }
        for(int l=0;l<nChannels;l++) {
            int index = i + nVoxels*l;
            values[0] +=  pDataCost[i] * pC[index]  * pBx[index] ;
            values[1] +=  pDataCost[i] * pC[index]  * pBy[index] ;
            values[2] +=  pDataCost[i] * pC[index]  * pBt[index] ;

            values[3] +=  pDataCost[i] * pBx[index] * pBx[index] ;
            values[4] +=  pDataCost[i] * pBy[index] * pBy[index] ;
            values[5] +=  pDataCost[i] * pBt[index] * pBt[index] ;

            values[6] +=  pDataCost[i] * pBx[index] * pBy[index] ;
            values[7] +=  pDataCost[i] * pBx[index] * pBt[index] ;
            values[8] +=  pDataCost[i] * pBy[index] * pBt[index] ;
        }
        if(params.useFeatures){
            for(int l=nChannels;l<nChannels+3;l++) {
                int index = i + nVoxels*l;
                valuesGradient[0] +=  pDataCost[i+nVoxels] * pC[index]  * pBx[index] ;
                valuesGradient[1] +=  pDataCost[i+nVoxels] * pC[index]  * pBy[index] ;
                valuesGradient[2] +=  pDataCost[i+nVoxels] * pC[index]  * pBt[index] ;

                valuesGradient[3] +=  pDataCost[i+nVoxels] * pBx[index] * pBx[index] ;
                valuesGradient[4] +=  pDataCost[i+nVoxels] * pBy[index] * pBy[index] ;
                valuesGradient[5] +=  pDataCost[i+nVoxels] * pBt[index] * pBt[index] ;

                valuesGradient[6] +=  pDataCost[i+nVoxels] * pBx[index] * pBy[index] ;
                valuesGradient[7] +=  pDataCost[i+nVoxels] * pBx[index] * pBt[index] ;
                valuesGradient[8] +=  pDataCost[i+nVoxels] * pBy[index] * pBt[index] ;
            }
        }
        for( int v = 0; v<9 ; v++) {
            values[v] /= nChannels;
            if(params.useFeatures){
                valuesGradient[v] /= 3;
                values[v] += params.gamma*valuesGradient[v];
            }
        }
        values[0] += pLapl[i];
        values[1] += pLapl[i+nVoxels];
        values[2] += pLapl[i+2*nVoxels];

        pCBx[i]    = values[0];
        pCBy[i]    = values[1];
        pCBt[i]    = values[2];

        pBx2[i]    = values[3];
        pBy2[i]    = values[4];
        pBt2[i]    = values[5];

        pBxy[i]    = values[6];
        pBxt[i]    = values[7];
        pByt[i]    = values[8];
    } // end of Linear System preparation
}

template <class T>
void STWarp<T>::sor(
         const Video<T> &dataCost,
         const Video<T> &smoothCost,
         const Video<T> &lapl,
         const Video<T> &CBx,
         const Video<T> &CBy,
         const Video<T> &CBt,
         const Video<T> &Bx2,
         const Video<T> &By2,
         const Video<T> &Bt2,
         const Video<T> &Bxy,
         const Video<T> &Bxt,
         const Video<T> &Byt,
         Video<T> &dWarpField
        )
{

    double eps     = params.eps; eps *= eps;
    int height     = dimensions[0];
    int width      = dimensions[1];
    int nFrames    = dimensions[2];
    double* lambda = params.lambda;
    int nVoxels    = height*width*nFrames;

    const T* pSmoothCost = smoothCost.dataReader();

    const T* pCBx = CBx.dataReader();
    const T* pCBy = CBy.dataReader();
    const T* pCBt = CBt.dataReader();
    const T* pBx2 = Bx2.dataReader();
    const T* pBy2 = By2.dataReader();
    const T* pBt2 = Bt2.dataReader();
    const T* pBxy = Bxy.dataReader();
    const T* pBxt = Bxt.dataReader();
    const T* pByt = Byt.dataReader();
    
    T* pU  = dWarpField.channelWriter(0);
    T* pV  = dWarpField.channelWriter(1);
    T* pW  = dWarpField.channelWriter(2);

    vector<T*> pField = vector<T*>(3,NULL);
    pField[0] = pU;
    pField[1] = pV;
    pField[2] = pW;

    int lambdaIndex[3] = {0,0,1};

    // SOR
    dWarpField.reset(0);
    T omega = 1.8;
    for(int sorIt = 0; sorIt<params.solverIterations;sorIt++)
    {
        for(int k=0;k<nFrames;k++)
            for(int j=0;j<width;j++)
                for(int i=0;i<height;i++)
        {
            // Compute laplacian weighting of the incremental flow
            int index = i + height*( j + width*( k ) );
            T weight = 0;
            T uTerm = 0, vTerm = 0, wTerm = 0;
            vector<T> terms(3,0);
            T coef[3] = {0,0,0};
            int indexW = 0;

            pair<NeighborhoodType::iterator, NeighborhoodType::iterator> range = 
                neighborhood.equal_range(index);

            for(int chan = 0; chan<3; chan ++){
                if( j>0 ) {
                    indexW          = index + (3*chan + 0)*nVoxels;
                    weight          = pSmoothCost[indexW-height];
                    weight         *= lambda[2*lambdaIndex[chan]];
                    terms[chan]    += weight*pField[chan][index-height];
                    coef[chan]     += weight;
                }
                if( j<width-1 ) {
                    indexW          = index + (3*chan + 0)*nVoxels;
                    weight          = pSmoothCost[indexW];
                    weight         *= lambda[2*lambdaIndex[chan]];
                    terms[chan]    += weight*pField[chan][index+height];
                    coef[chan]     += weight;
                }
                if( i>0 ) {
                    indexW          = index + (3*chan + 1)*nVoxels;
                    weight          = pSmoothCost[indexW-1];
                    weight         *= lambda[2*lambdaIndex[chan]];
                    terms[chan]    += weight*pField[chan][index-1];
                    coef[chan]     += weight;
                }
                if( i<height-1 ) {
                    indexW          = index + (3*chan + 1)*nVoxels;
                    weight          = pSmoothCost[indexW];
                    weight         *= lambda[2*lambdaIndex[chan]];
                    terms[chan]    += weight*pField[chan][index+1];
                    coef[chan]     += weight;
                }
                if(params.useFlow){
                    for(NeighborhoodType::iterator it = range.first; it != range.second; ++it ){
                        pair<int,int> n = it->second;

                        int indexOther = n.first;
                        if(n.second < 0){ // past connnection
                            indexW          = indexOther + (3*chan + 2)*nVoxels;
                        }else{
                            indexW = index + (3*chan + 2)*nVoxels;
                        }
                        weight          = pSmoothCost[indexW];
                        weight         *= lambda[2*lambdaIndex[chan]+1];
                        terms[chan]    += weight*pField[chan][indexOther];
                        coef[chan]     += weight;
                    }
                }else{
                    if( k>0 ) {
                        int indexPast   = index - height*width;
                        indexW          = indexPast + (3*chan + 2)*nVoxels;
                        weight          = pSmoothCost[indexW];
                        weight         *= lambda[2*lambdaIndex[chan]+1];
                        terms[chan]    += weight*pField[chan][indexPast];
                        coef[chan]     += weight;
                    }
                    if( k<nFrames-1 ) {
                        int indexFuture = index + height*width;
                        indexW          = index + (3*chan + 2)*nVoxels;
                        weight          = pSmoothCost[indexW];
                        weight         *= lambda[2*lambdaIndex[chan]+1];
                        terms[chan]    += weight*pField[chan][indexFuture];
                        coef[chan]     += weight;
                    }
                }
            }

            uTerm = -terms[0];
            vTerm = -terms[1];
            wTerm = -terms[2];

            // We compute (A+M)x = b, M does the laplacian filtering the
            // incremental flow

            // Update u:
            // u[k,i+1] = (1-w)*u[k,i]) + 
            // w/(A[k,k]+M[k,k]) * (b[k])-Sum_{l<k}{A[k,l]*u[l,i+1]}-Sum_{l>k}{A[k,l]*u[l,i]})
            uTerm    += pBxy[index]*pV[index];
            uTerm    += pBxt[index]*pW[index];
            uTerm     = (1-omega)*pU[index] 
                      + omega/( pBx2[index] + coef[0] ) * ( pCBx[index] - uTerm );
            pU[index] = uTerm;

            // Update v
            vTerm    += pBxy[index]*pU[index];
            vTerm    += pByt[index]*pW[index];
            vTerm     = (1-omega)*pV[index] 
                      + omega/( pBy2[index] + coef[1] ) * ( pCBy[index] - vTerm );
            pV[index] = vTerm;

            // Update w
            wTerm    += pBxt[index]*pU[index];
            wTerm    += pByt[index]*pV[index];
            wTerm     = (1-omega)*pW[index] 
                      + omega/( pBt2[index] + coef[2] ) * ( pCBt[index] - wTerm );
            pW[index] = wTerm;
        } // end of SOR-iteration
        
    } // end of SOR
}

/*
 * Compute a per channel weighted laplacian.
 * @param[in] input video whose laplacian is to be computed
 * @param[in] weight weight map
 * @params[out] output the resulting laplacian filtered video
 */
template <class T>
void STWarp<T>::weightedLaplacian(const Video<T>& input, const Video<T>& weight, Video<T>& output) {
    VideoSize sz  = input.size();
    int height    = sz.height;
    int width     = sz.width;
    int nFrames   = sz.nFrames;
    int nVoxels   = input.voxelCount();

    double* lambda = params.lambda;
    int lambdaIndex[3] = {0,0,1};

    T* pOutput       = output.dataWriter();
    const T* pSmoothCost = weight.dataReader();

    vector<const T*> pField = vector<const T*>(input.channelCount(),NULL);
    for (int i = 0; i < input.channelCount(); ++i) {
        pField[i] = input.channelReader(i);
    }

    for(int k=0;k<nFrames;k++)
        for(int j=0;j<width;j++)
            for(int i=0;i<height;i++)
    {
        // Compute laplacian weighting of the incremental flow
        int index = i + height*( j + width*( k ) );
        T weight = 0;
        vector<T> terms(3,0);
        T coef[3] = {0,0,0};
        int indexW = 0;

        pair<NeighborhoodType::iterator, NeighborhoodType::iterator> range = 
            neighborhood.equal_range(index);

        for(int chan = 0; chan<3; chan ++){
            if( j>0 ) {
                indexW          = index + (3*chan + 0)*nVoxels;
                weight          = pSmoothCost[indexW-height];
                weight         *= lambda[2*lambdaIndex[chan]];
                terms[chan]    += weight*pField[chan][index-height];
                coef[chan]     += weight;
            }
            if( j<width-1 ) {
                indexW          = index + (3*chan + 0)*nVoxels;
                weight          = pSmoothCost[indexW];
                weight         *= lambda[2*lambdaIndex[chan]];
                terms[chan]    += weight*pField[chan][index+height];
                coef[chan]     += weight;
            }
            if( i>0 ) {
                indexW          = index + (3*chan + 1)*nVoxels;
                weight          = pSmoothCost[indexW-1];
                weight         *= lambda[2*lambdaIndex[chan]];
                terms[chan]    += weight*pField[chan][index-1];
                coef[chan]     += weight;
            }
            if( i<height-1 ) {
                indexW          = index + (3*chan + 1)*nVoxels;
                weight          = pSmoothCost[indexW];
                weight         *= lambda[2*lambdaIndex[chan]];
                terms[chan]    += weight*pField[chan][index+1];
                coef[chan]     += weight;
            }
            if(params.useFlow){
                for(NeighborhoodType::iterator it = range.first; it != range.second; ++it ){
                    pair<int,int> n = it->second;
                    int indexOther = n.first;
                    if(n.second < 0){ // past connnection
                        indexW          = indexOther + (3*chan + 2)*nVoxels;
                    }else{
                        indexW = index + (3*chan + 2)*nVoxels;
                    }
                    weight          = pSmoothCost[indexW];
                    weight         *= lambda[2*lambdaIndex[chan]+1];
                    terms[chan]    += weight*pField[chan][indexOther];
                    coef[chan]     += weight;
                }
            }else{
                if( k>0 ) {
                    int indexPast   = index - height*width;
                    indexW          = indexPast + (3*chan + 2)*nVoxels;
                    weight          = pSmoothCost[indexW];
                    weight         *= lambda[2*lambdaIndex[chan]+1];
                    terms[chan]    += weight*pField[chan][indexPast];
                    coef[chan]     += weight;
                }
                if( k<nFrames-1 ) {
                    int indexFuture = index + height*width;
                    indexW          = index + (3*chan + 2)*nVoxels;
                    weight          = pSmoothCost[indexW];
                    weight         *= lambda[2*lambdaIndex[chan]+1];
                    terms[chan]    += weight*pField[chan][indexFuture];
                    coef[chan]     += weight;
                }
            }
            pOutput[index + nVoxels*chan] = terms[chan] - coef[chan]*pField[chan][index];
        }
    }


}

template class STWarp<float>;
template class STWarp<double>;
