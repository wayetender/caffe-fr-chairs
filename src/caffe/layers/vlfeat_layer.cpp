#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
// extern "C" {
//   #include <vl/generic.h>
//   #include <vl/hog.h>
// }
#include <omp.h>

namespace caffe {
  
bool is_outside_the_box(VlSiftKeypoint key, VLFeatParameter param) {
  if (!(param.has_box_minx() && param.has_box_maxx() && param.has_box_miny() && param.has_box_maxy()))
    return true;
  if (key.x <= param.box_maxx() && key.x >= param.box_minx() && key.y <= param.box_maxy() && key.y >= param.box_miny())
    return false;
  return true;  
}
  
template <typename Dtype>
void VLFeatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top)
{
   if (this->layer_param_.vlfeat_param().descriptor_type() == "HOG") {
    CHECK(bottom[0]->shape()[1] == 1 || bottom[0]->shape()[1] == 3) << "HOG can only be computed for a 3-channel (RGB) or 1-channel image";
    num_orientations_ = this->layer_param_.vlfeat_param().hog_num_orientations();
    hog_cell_size_ = this->layer_param_.vlfeat_param().hog_cell_size();
    
    if (this->layer_param_.vlfeat_param().hog_variant() == "UOCTTI")
      hog_variant_ = VlHogVariantUoctti;      
    else if (this->layer_param_.vlfeat_param().hog_variant() == "DalalTriggs")
      hog_variant_ = VlHogVariantDalalTriggs;
    else
      LOG(ERROR) << "Unknown HOG variant " << this->layer_param_.vlfeat_param().hog_variant();
    
    hog_extractor_ = vl_hog_new(hog_variant_, num_orientations_, VL_FALSE);
    
    vl_hog_put_image(hog_extractor_, bottom[0]->cpu_data(), bottom[0]->shape()[3], bottom[0]->shape()[2], bottom[0]->shape()[1], hog_cell_size_);
    out_width_ = vl_hog_get_width(hog_extractor_) ;
    out_height_ = vl_hog_get_height(hog_extractor_) ;
    out_channels_ = vl_hog_get_dimension(hog_extractor_) ;
   } else if (this->layer_param_.vlfeat_param().descriptor_type() == "SIFT") {
    CHECK_EQ(bottom[0]->shape()[1], 1) << "SIFT can only be computed for a 1-channel image";
    num_octaves_ = -1;
    levels_per_octave_ = 3;
    first_octave_ = 0;
    sift_dim_ = 128;
    ds_factor_ = 4;
    sift_extractor_ =  vl_sift_new(bottom[0]->shape()[3], bottom[0]->shape()[2], num_octaves_, levels_per_octave_, first_octave_);
    out_channels_ = sift_dim_ + 5; // descriptor plus the coordinates plus the angle sin and cos and the scale
    out_width_ = ceil(bottom[0]->shape()[3] / ds_factor_);
    out_height_ = ceil(bottom[0]->shape()[2] / ds_factor_);
    if (this->layer_param_.vlfeat_param().has_fixed_angle())
      LOG(INFO) << "Setting angle of all keypoints to " << this->layer_param_.vlfeat_param().fixed_angle();
    if (this->layer_param_.vlfeat_param().has_fixed_sigma())
      LOG(INFO) << "Setting sigma of all keypoints to " << this->layer_param_.vlfeat_param().fixed_sigma();
    if (this->layer_param_.vlfeat_param().has_fixed_displacement())
      LOG(INFO) << "Setting sigma of all dx and dy to " << this->layer_param_.vlfeat_param().fixed_displacement();
  } else if (this->layer_param_.vlfeat_param().descriptor_type() == "LBP") {
    CHECK(bottom[0]->shape()[1] == 1 || bottom[0]->shape()[1] == 3) << "HOG can only be computed for a 3-channel (RGB) or 1-channel image";
    num_orientations_ = this->layer_param_.vlfeat_param().hog_num_orientations();
    lbp_cell_size_ = this->layer_param_.vlfeat_param().lbp_cell_size();
    
    lbp_extractor_ = vl_lbp_new(VlLbpUniform, VL_FALSE);
    
    out_width_ = bottom[0]->shape()[3] / lbp_cell_size_ ;
    out_height_ = bottom[0]->shape()[2] / lbp_cell_size_  ;
    out_channels_ = vl_lbp_get_dimension(lbp_extractor_) ;
  } else
    LOG(ERROR) << "Unkonown descriptor_type() " << this->layer_param_.vlfeat_param().descriptor_type();
  omp_set_num_threads(this->layer_param_.vlfeat_param().num_threads());
  LOG(INFO) << "Running with " << this->layer_param_.vlfeat_param().num_threads() << " threads";
}

template <typename Dtype>
void VLFeatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    // Allocate output
    top[0]->Reshape(bottom[0]->shape()[0], out_channels_, out_height_, out_width_);
}

template <typename Dtype>
void VLFeatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top)
{
  int bot_num = bottom[0]->shape()[0];
  int bot_channels = bottom[0]->shape()[1];
  int bot_height = bottom[0]->shape()[2];
  int bot_width = bottom[0]->shape()[3];
  
  int bot_offset = bot_channels * bot_height * bot_width;
  int top_offset = top[0]->shape()[1] * top[0]->shape()[2] * top[0]->shape()[3];
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_mutable_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  caffe_set(top[0]->count(), Dtype(0), top_data);  
  
  LayerParameter layer_param = this->layer_param_;
  #pragma omp parallel for
  for (int n=0; n<bot_num; n++) {
//     LOG(INFO) << "Image " << n;
    const Dtype* curr_bottom_data = bottom_data + bot_offset * n;
    Dtype* curr_bottom_mutable_data = bottom_mutable_data + bot_offset * n;
    Dtype* curr_top_data = top_data + top_offset * n;
    
    if (layer_param.vlfeat_param().descriptor_type() == "HOG") {
      VlHog* hog_extractor = vl_hog_new(hog_variant_, num_orientations_, VL_FALSE);
      vl_hog_put_image(hog_extractor, curr_bottom_data, bot_width, bot_height, bot_channels, hog_cell_size_) ;
      vl_hog_extract(hog_extractor, curr_top_data);  
      vl_hog_delete(hog_extractor) ;
    } else if (layer_param.vlfeat_param().descriptor_type() == "LBP") {
      VlLbp* lbp_extractor = vl_lbp_new(VlLbpUniform, VL_FALSE);
      vl_lbp_process(lbp_extractor, curr_top_data, curr_bottom_mutable_data, bot_width, bot_height, lbp_cell_size_) ;
      vl_lbp_delete(lbp_extractor) ;
    } else if (layer_param.vlfeat_param().descriptor_type() == "SIFT") {
      int err;
      double angles[4];  
      float* descr = new float[out_channels_];
      for (int i=0; i<out_channels_; i++)
        descr[i] = 0.;
      VlSiftFilt* sift_extractor =  vl_sift_new(bot_width, bot_height, num_octaves_, levels_per_octave_, first_octave_);
      err = !VL_ERR_EOF;
      for (int i=0; err != VL_ERR_EOF; i++) {
        // process the current octave
        if (i==0)
          err = vl_sift_process_first_octave(sift_extractor, curr_bottom_data);
        else
          err = vl_sift_process_next_octave(sift_extractor);
        // if there are still octaves left
        if (err != VL_ERR_EOF) {
          // detect keypoints
          vl_sift_detect(sift_extractor);
          // get the list of keypoints
          VlSiftKeypoint const *keys = vl_sift_get_keypoints(sift_extractor);
          int nkeys = vl_sift_get_nkeypoints(sift_extractor);  
//           LOG(INFO) << "Image " << n << " detected " << nkeys << " keypoints at octave " << i;
          //extract descriptors from the keypoints
          for (int nk=0; nk < nkeys; nk++) {
            VlSiftKeypoint curr_key = *(keys + nk);
            int num_found_orientations = vl_sift_calc_keypoint_orientations(sift_extractor, angles, &curr_key);
            if (layer_param.vlfeat_param().has_fixed_angle())
              angles[0] = layer_param.vlfeat_param().fixed_angle();
            if (layer_param.vlfeat_param().has_fixed_sigma())
              curr_key.sigma = layer_param.vlfeat_param().fixed_sigma();
            if (num_found_orientations > 0) {
              //extract the descriptor
              vl_sift_calc_keypoint_descriptor(sift_extractor, descr, &curr_key, angles[0]); 
              // write the descriptor to the top blob
//               LOG(INFO) << "Keypoint " << nk << ": o=" << keys[nk].o << ", ix=" << keys[nk].ix << ", iy=" << keys[nk].iy << ", x=" << keys[nk].x << ", y=" << keys[nk].y << ", s=" << keys[nk].s << ", sigma=" << keys[nk].sigma;
              if (is_outside_the_box(curr_key, layer_param.vlfeat_param())) {
                int curr_x = int(floor(curr_key.x / float(ds_factor_)));
                int curr_y = int(floor(curr_key.y / float(ds_factor_)));
                float dx, dy;
                if (layer_param.vlfeat_param().has_fixed_displacement()) {
                  dx = layer_param.vlfeat_param().fixed_displacement();
                  dy = layer_param.vlfeat_param().fixed_displacement();
                } else {
                  dx = curr_key.x - float(curr_x * ds_factor_);
                  dy = curr_key.y - float(curr_y * ds_factor_);   
                }
                
                // if asked for, mess with the features after the keypoint is already extracted
                if (layer_param.vlfeat_param().has_replace_sigma())
                  curr_key.sigma = layer_param.vlfeat_param().replace_sigma();
                if (layer_param.vlfeat_param().has_replace_angle())
                  angles[0] = layer_param.vlfeat_param().replace_angle();
                
                descr[out_channels_-5] = sin(angles[0]);
                descr[out_channels_-4] = cos(angles[0]);
                descr[out_channels_-3] = dx;
                descr[out_channels_-2] = dy;
                descr[out_channels_-1] = log(curr_key.sigma);
                
  //               //This is just a stupid test: put the actual pixel values to the feature vector
  //               int xi = (int)(curr_key.x);
  //               int yi = (int)(curr_key.y);
  //               int bot_ind = yi*bottom[0]->shape()[3] + xi;
  //               descr[out_channels_-1] = bottom_data[bot_ind];
                
                for (int c=0; c < out_channels_; c++) {
                  int top_ind = (c * top[0]->shape()[2] + curr_y) * top[0]->shape()[3] + curr_x;
                  curr_top_data[top_ind] = descr[c];
                }    
              }
            } else
              LOG(WARNING) << "Found 0 orientations for keypoint " << nk;     
          }  
        }          
      }
      delete[] descr;
      vl_sift_delete(sift_extractor);
    }
  }
}

template <typename Dtype>
void VLFeatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{  
  LOG(FATAL) << "Backward VLFeat not implemented.";
}
  
#ifdef CPU_ONLY
STUB_GPU(VLFeatLayer);
#endif

INSTANTIATE_CLASS(VLFeatLayer);
REGISTER_LAYER_CLASS(VLFeat);

}  // namespace caffe
