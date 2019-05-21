/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#ifndef _FLF_EPSILON_REMOVAL_HH
#define _FLF_EPSILON_REMOVAL_HH

#include "FlfCore/Lattice.hh"
#include "Network.hh"

namespace Flf {

/*
 * Fast epsilon-removal,
 * making use of the fact that lattices are ALWAYS acyclic
 */
ConstLatticeRef fastRemoveEpsilons(ConstLatticeRef l);
ConstLatticeRef removeEpsilons(ConstLatticeRef l);
NodeRef         createEpsilonRemovalNode(const std::string& name, const Core::Configuration& config);

/*
 * Remove arcs of length 0
 */
ConstLatticeRef fastRemoveNullArcs(ConstLatticeRef l);
NodeRef         createNullArcsRemovalNode(const std::string& name, const Core::Configuration& config);

}  // namespace Flf
#endif  // _FLF_EPSILON_REMOVAL_HH
