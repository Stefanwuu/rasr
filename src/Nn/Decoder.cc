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

#include "Decoder.hh"
#include <Mm/Module.hh>
#include "Core/ReferenceCounting.hh"
#include "Nn/LabelHistory.hh"

namespace Nn {

/*
 * =============================
 * === Decoder =================
 * =============================
 */

Decoder::Decoder(const Core::Configuration& config)
        : Core::Component(config) {
}

void Decoder::reset() {
    encoderOutputBuffer_.clear();
    segmentEnd_ = false;
}

void Decoder::addEncoderOutput(FeatureVectorRef encoderOutput) {
    encoderOutputBuffer_.push_back(encoderOutput);
}

void Decoder::signalSegmentEnd() {
    segmentEnd_ = true;
}

/*
 * =============================
 * === NoOpDecoder =============
 * =============================
 */

NoOpDecoder::NoOpDecoder(const Core::Configuration& config)
        : Core::Component(config), Precursor(config) {}

Core::Ref<LabelHistory> NoOpDecoder::getStartHistory() {
    StepLabelHistory startHistory;
    startHistory.currentStep = 0ul;
    return Core::ref(&startHistory);
}

void NoOpDecoder::extendHistory(Core::Ref<LabelHistory> history, LabelIndex labelIndex, bool isLoop) {
    auto stepHistory = dynamic_cast<StepLabelHistory*>(history.get());
    ++stepHistory->currentStep;
}

std::optional<Score> NoOpDecoder::getDecoderScore(Core::Ref<const LabelHistory> history, LabelIndex labelIndex, bool isLoop) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*history);
    if (encoderOutputBuffer_.size() <= stepHistory.currentStep) {
        return {};
    }
    return encoderOutputBuffer_.at(stepHistory.currentStep)->at(labelIndex);
}

/*
 * =============================
 * = LegacyFeatureScorerDecoder
 * =============================
 */

LegacyFeatureScorerDecoder::LegacyFeatureScorerDecoder(const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config),
          featureScorer_(Mm::Module::instance().createFeatureScorer(config)),
          scoreCache_() {}

void LegacyFeatureScorerDecoder::reset() {
    Precursor::reset();
    featureScorer_.reset();
    scoreCache_.clear();
}

void LegacyFeatureScorerDecoder::addEncoderOutput(FeatureVectorRef encoderOutput) {
    auto feature = Core::ref(new Mm::Feature(*encoderOutput));
    if (featureScorer_->isBuffered() and not featureScorer_->bufferFilled()) {
        featureScorer_->addFeature(feature);
    }
    else {
        scoreCache_.push_back(featureScorer_->getScorer(feature));
    }
}

void LegacyFeatureScorerDecoder::signalSegmentEnd() {
    Precursor::signalSegmentEnd();
    while (!featureScorer_->bufferEmpty()) {
        scoreCache_.push_back(featureScorer_->flush());
    }
}

Core::Ref<LabelHistory> LegacyFeatureScorerDecoder::getStartHistory() {
    StepLabelHistory startHistory;
    startHistory.currentStep = 0ul;
    return Core::ref(&startHistory);
}

void LegacyFeatureScorerDecoder::extendHistory(Core::Ref<LabelHistory> history, LabelIndex labelIndex, bool isLoop) {
    auto stepHistory = dynamic_cast<StepLabelHistory*>(history.get());
    ++stepHistory->currentStep;
}

std::optional<Score> LegacyFeatureScorerDecoder::getDecoderScore(Core::Ref<const LabelHistory> history, LabelIndex labelIndex, bool isLoop) {
    const auto& stepHistory = dynamic_cast<const StepLabelHistory&>(*history);
    if (scoreCache_.size() <= stepHistory.currentStep) {
        return {};
    }
    // Retrieve score from score cache at the right index
    return scoreCache_.at(stepHistory.currentStep)->score(labelIndex);
}

}  // namespace Nn