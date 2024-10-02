/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef TYPES_HH_
#define TYPES_HH_

#include <Core/Types.hh>
#include <Flow/Vector.hh>
#include <Search/Types.hh>
#include <Speech/Types.hh>

namespace Math {
template<typename T>
class CudaMatrix;
template<typename T>
class CudaVector;
}  // namespace Math

namespace Nn {

template<typename T>
class Types {
public:
    typedef Math::CudaMatrix<T> NnMatrix;
    typedef Math::CudaVector<T> NnVector;
};

typedef Flow::Vector<f32>            FeatureVector;
typedef Flow::DataPtr<FeatureVector> FeatureVectorRef;

typedef Search::Score Score;

}  // namespace Nn

#endif /* TYPES_HH_ */
