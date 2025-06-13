//
//  character-tokenizer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//
#pragma once

#include "tokenizer.h"
#include <unordered_map>
#include <vector>
#include <string>

class CharacterTokenizer : public Tokenizer {
public:
    CharacterTokenizer();
    virtual ~CharacterTokenizer();

    std::vector<int> tokenize(const std::string& text) const override;
    std::string detokenize(const std::vector<int>& tokens) const override;
    size_t vocabSize() const override;

private:
    void buildVocabulary();  // explicitly build character mappings

    std::unordered_map<char, int> char2idx_; // char → index explicitly
    std::vector<char> idx2char_;             // index → char explicitly
};
