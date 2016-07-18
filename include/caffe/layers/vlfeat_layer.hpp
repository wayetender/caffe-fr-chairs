#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

extern "C" {
  #include <vl/generic.h>
  #include <vl/hog.h>
  #include <vl/sift.h>
  #include <vl/lbp.h>
}

namespace caffe {

/*
 * @brief Interface to some functions of the VLFeat library: HOG, SIFT and LBP.
 *
 */
template <typename Dtype>
class VLFeatLayer : public Layer<Dtype> {
 public:
  explicit VLFeatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VLFeat"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  VlHog* hog_extractor_;
  VlSiftFilt* sift_extractor_;
  VlLbp* lbp_extractor_;
  int out_width_;
  int out_height_;
  int out_channels_;
  int hog_cell_size_;
  int num_orientations_;
  int ds_factor_;
  VlHogVariant hog_variant_;
  int num_octaves_;
  int levels_per_octave_;
  int first_octave_ ;
  int sift_dim_;
  int lbp_cell_size_;

};

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_
