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

#include "LabelScorer.hh"

namespace Nn {

LabelScorer::LabelScorer(const Core::Configuration& config)
        : Core::Component(config) {}

std::vector<std::optional<LabelScorer::ScoreWithTime>> LabelScorer::getScoresWithTime(const std::vector<LabelScorer::Request>& requests) {
    std::vector<std::optional<LabelScorer::ScoreWithTime>> results;
    results.reserve(requests.size());
    for (auto& request : requests) {
        results.push_back(getScoreWithTime(request));
    }

    return results;
}

}  // namespace Nn
