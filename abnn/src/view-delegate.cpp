//
//  view-delegate.cpp
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//

#include "brain-engine.h"
#include "view-delegate.h"
#include "model-config.h"
#include <iostream>
#include <filesystem>
#include <mach-o/dyld.h>
#include "configuration-manager.h"
#include "logger.h"
#include "functional-dataset.h"

const char* modelFilename = "simple.yml";


#pragma mark - ViewDelegate
#pragma region ViewDelegate {

ViewDelegate::ViewDelegate(MTL::Device* pDevice)
: MTK::ViewDelegate()
, _pDevice(pDevice)
, _pBrainEngine(nullptr)
{
    _pBrainEngine = new BrainEngine(_pDevice, 256, 256);

    auto stim = std::make_shared<FunctionalDataset>(
                    /*nInput=*/256,
                    /*dtSec =*/ 0.001,//dt,
                    /*freqHz=*/1.0);
    _pBrainEngine->set_stimulus(stim);
    _pBrainEngine->start_async();

    std::cout << "✅ BrainEngine loaded" << std::endl;
}

ViewDelegate::~ViewDelegate()
{
    delete _pBrainEngine;
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

BrainEngine* ViewDelegate::getBrainEngine()
{
    return _pBrainEngine;
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
    std::cout << "📂 Loaded file " << modelFilename << std::endl;

    return resourcePath.string();
}

#pragma endregion ViewDelegate }
