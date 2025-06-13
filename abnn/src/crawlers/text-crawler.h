//
//  text-crawler.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-21.
//

#pragma once

#include <vector>
#include <string>
#include <random>

class TextCrawler {
public:
    explicit TextCrawler(const std::string& corpusDirectory,
                         size_t sequenceLength,
                         size_t samplesPerFile);
    ~TextCrawler();

    void loadNextFile(); // explicitly loads next file randomly

    std::string getRandomSequence(); // explicitly gets random sequence from current file

    size_t currentFileSize() const;

private:
    void loadFile(const std::string& filepath); // explicitly load a single file

    std::vector<std::string> filepaths_; // all file paths explicitly loaded from directory
    std::string currentFileContent_;     // explicitly current loaded file content
    size_t sequenceLength_;              // explicitly desired sequence length
    size_t samplesPerFile_;              // explicitly how many samples per file
    size_t currentSampleCount_;          // explicitly tracks current sample count per file

    std::mt19937 generator_;             // random generator explicitly
    std::uniform_int_distribution<size_t> distribution_; // explicit random distribution

    void resetDistribution();            // explicitly updates distribution bounds
};
