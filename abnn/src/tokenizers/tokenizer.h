//
//  tokenizer.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#pragma once

#include <vector>
#include <string>

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // Explicitly converts text to token IDs
    virtual std::vector<int> tokenize(const std::string& text) const = 0;

    // Explicitly converts token IDs back to text
    virtual std::string detokenize(const std::vector<int>& tokens) const = 0;

    // Explicitly returns vocabulary size (useful for embedding layers)
    virtual size_t vocabSize() const = 0;
};
