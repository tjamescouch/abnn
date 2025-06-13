//
//  character-tokenizer.cpp
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#include "character-tokenizer.h"
#include <stdexcept>
#include <set>
#include "logger.h"
#include <iostream>

// Constructor explicitly builds vocabulary
CharacterTokenizer::CharacterTokenizer() {
    buildVocabulary();
}

CharacterTokenizer::~CharacterTokenizer() {}

// Explicitly builds vocabulary from standard ASCII characters
void CharacterTokenizer::buildVocabulary() {
    // Explicitly build ASCII character vocabulary (can be customized)
    std::set<char> charset;
    for (int c = 32; c < 127; ++c) { // printable ASCII explicitly
        charset.insert(static_cast<char>(c));
    }
    //charset.insert('\n');  // explicitly include newline
    
    int idx = 0;
    for (char c : charset) {
        char2idx_[c] = idx++;
        idx2char_.push_back(c);
    }
    
    while (idx2char_.size() < 128) {
        idx2char_.push_back('?');
    }
}

// Explicitly converts string to sequence of token IDs
std::vector<int> CharacterTokenizer::tokenize(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());

    //Logger::log << "Tokenizing text: " << text << std::endl;
    
    for (char c : text) {
        auto it = char2idx_.find(c);
        if (it != char2idx_.end()) {
            tokens.push_back(it->second);
        } else {
            std::cout << "Character not in vocabulary: " << c << std::endl;
            
            tokens.push_back(char2idx_.find('?')->second);
        }
    }
    
    return tokens;
}

// Explicitly converts sequence of token IDs back to string
std::string CharacterTokenizer::detokenize(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());

    for (int idx : tokens) {
        if (idx >= 0 && idx < static_cast<int>(idx2char_.size())) {
            text.push_back(idx2char_[idx]);
        } else {
            throw std::runtime_error("Invalid token ID in detokenization.");
        }
    }
    return text;
}

// Explicitly returns vocabulary size
size_t CharacterTokenizer::vocabSize() const {
    return idx2char_.size();
}
