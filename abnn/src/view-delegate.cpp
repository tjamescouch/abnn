//
//  view-delegate.cpp
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//

#include "view-delegate.h"
#include "model-config.h"
#include <iostream>
#include <filesystem>
#include <mach-o/dyld.h>
#include "configuration-manager.h"

//const char* modelFilename = "ocr.yml";
//const char* modelFilename = "simple-ocr.yml";
//const char* modelFilename = "ocr-with-dropout.yml";
//const char* modelFilename = "ocr-with-batch-normalization.yml";
//const char* modelFilename = "ocr-complete.yml";
const char* modelFilename = "feed-forward.yml";
//const char* modelFilename = "residual-connection.yml";
//const char* modelFilename = "gelu.yml";
//const char* modelFilename = "multi-dense-layer.yml";
//const char* modelFilename = "single-dense-layer.yml";
//const char* modelFilename = "self-attention.yml";
//const char* modelFilename = "multi-head-attention.yml";
//const char* modelFilename = "layer-norm.yml";
//const char* modelFilename = "sequence-length.yml";
//const char* modelFilename = "passthru.yml";
// const char* modelFilename = "transformer-complete.yml";

#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate(MTL::Device* pDevice)
: MTK::ViewDelegate()
, _pDevice(pDevice)
, _pNeuralEngine(nullptr)
, _pDataManager(nullptr)
{
    static ModelConfig config = ModelConfig::loadFromFile(getDefaultModelFilePath());
    config.filename = modelFilename;
    
    ConfigurationManager::instance().setConfig(&config);

    _pDataManager = (new DataManager())->configure(&config);
    
    // Instantiate NeuralEngine using the updated constructor with DataManager
    _pNeuralEngine = new NeuralEngine(_pDevice, config, _pDataManager);

    Logger::log << "✅ NeuralEngine loaded with model: " << config.name << std::endl;
}

ViewDelegate::~ViewDelegate()
{
    delete _pNeuralEngine;
}

void ViewDelegate::drawInMTKView(MTK::View* pView)
{
    pView->setDepthStencilPixelFormat(MTL::PixelFormatDepth32Float);
    pView->setClearDepth(1.0);
}

void ViewDelegate::drawableSizeWillChange(MTK::View* pView, CGSize size)
{
    // Handle resize events if needed
}

NeuralEngine* ViewDelegate::getNeuralEngine()
{
    return _pNeuralEngine;
}

std::string ViewDelegate::getDefaultModelFilePath() {
    namespace fs = std::filesystem;

    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        throw std::runtime_error("❌ Executable path buffer too small.");
    }

    fs::path executablePath = fs::canonical(path);
    fs::path resourcePath = executablePath.parent_path().parent_path() / "Resources" / modelFilename;

    if (!fs::exists(resourcePath)) {
        throw std::runtime_error("❌ Could not find configuration yml at " + resourcePath.string());
    }
    Logger::log << "📂 Loaded file " << modelFilename << std::endl;

    return resourcePath.string();
}

#pragma endregion ViewDelegate }
