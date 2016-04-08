#include "mcp/NNReconstruction.hpp"
#include <cassert>

Video<nnf_data_t> NNReconstruction::reconstruct() {
    Video<float> buffer(db_->size());
    Video<nnf_data_t> out(db_->size());

    int h  = buffer.getHeight();
    int w  = buffer.getWidth();
    int nF = buffer.frameCount();
    int nC = buffer.channelCount();

    int psz_space = params_.patch_size_space;
    int psz_time  = params_.patch_size_time;

    int nVoxels = h*w*nF;

    Video<int> aggregation_count(h,w,nF,1);

    for (int pt = 0; pt < nF-psz_time+1; ++pt)
    for (int px = 0; px < w-psz_space+1; ++px)
    for (int py = 0; py < h-psz_space+1; ++py)
    {
        int voxel = py + h*(px + w*pt);
        // get patch weight
        for(int k = 0 ; k < params_.knn ; ++k) 
        {
            float weight = w_->dataReader()[voxel+k*nVoxels];
            int target_x = nnf_->dataReader()[voxel+0*nVoxels+k*3*nVoxels];
            int target_y = nnf_->dataReader()[voxel+1*nVoxels+k*3*nVoxels];
            int target_t = nnf_->dataReader()[voxel+2*nVoxels+k*3*nVoxels];
            assert(target_x < db_->getWidth()-psz_space+1);
            assert(target_y < db_->getHeight()-psz_space+1);
            assert(target_t < db_->frameCount()-psz_time+1);

            // Loop through the patch pixel values and copy them over
            for (int t = 0; t < psz_time; ++t)
            for (int x = 0; x < psz_space; ++x)
            for (int y = 0; y < psz_space; ++y)
            {
                for (int c = 0; c < nC; ++c) {
                    buffer.at(voxel+y,x,t,c) +=  
                        weight*((float)db_->at(target_y+y,target_x+x,target_t+t,c));
                }
                aggregation_count.at(voxel+y,x,t,0) += 1;
            } // patch pixel loop
        } // knn loop
    }

    // Normalize by the number of patches influencing each pixel
    for (int pt = 0; pt < nF; ++pt)
    for (int px = 0; px < w; ++px)
    for (int py = 0; py < h; ++py)
    for (int c = 0; c < nC; ++c)
    {
        int voxel = py + h*(px + w*pt);
        out.at(voxel,0,0,c) = params_.knn*buffer.at(voxel,0,0,c) / aggregation_count.at(voxel);
    }

    return out;
}
