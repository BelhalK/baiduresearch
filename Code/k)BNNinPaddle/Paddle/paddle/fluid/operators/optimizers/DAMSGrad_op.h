/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "Eigen/Dense"

#include<iostream>

namespace paddle {
namespace operators {

template <typename T>
class DAMSGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
	auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    if (param_var->IsType<framework::LoDTensor>()) {
	  auto *param_out_tensor = ctx.Output<framework::Tensor>("ParamOut");
	  auto *moment1_out_tensor = ctx.Output<framework::Tensor>("Moment1Out");
	  	  
      param_out_tensor->mutable_data<T>(ctx.GetPlace());
      moment1_out_tensor->mutable_data<T>(ctx.GetPlace());

    
      // Actually, all tensors are LoDTensor except SelectedRows.
      if (grad_var->IsType<framework::LoDTensor>()) {

        auto param = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Param"));
        auto grad = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Grad"));
        auto moment1 = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Moment1"));
        auto moment2 = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Moment2"));


      
	    auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
	    auto moment1_out = framework::EigenVector<T>::Flatten(*moment1_out_tensor);
    
    
        T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));

		auto *lr = learning_rate->data<T>();


//        auto aux_indicator_tensor = *ctx.Input<framework::Tensor>("AuxIndicator");
//        std::vector<float> aux_indicator_tensor_buffer;
//        TensorToVector(aux_indicator_tensor,&aux_indicator_tensor_buffer);

        moment1_out = moment1 * beta1 + (1-beta1) * grad;
        param_out = param - lr[0] * moment1_out / moment2.sqrt();

      } else {
        PADDLE_THROW("Unsupported Variable Type of Grad");
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Parameter");
    }
  }
};
}  // namespace operators
}  // namespace paddle
