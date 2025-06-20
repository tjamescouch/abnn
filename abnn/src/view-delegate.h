//
//  view-delegate.h
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//
#ifndef VIEW_DELEGATE_H
#define VIEW_DELEGATE_H

#pragma region Declarations {

#include "common.h"
#include "model-config.h"
#include "brain-engine.h"


class ViewDelegate : public MTK::ViewDelegate {
public:
    ViewDelegate(MTL::Device* pDevice);
    virtual ~ViewDelegate();

    bool loadModelFromFile(const std::string& filePath);
    void drawInMTKView(MTK::View* pView) override;
    void drawableSizeWillChange(MTK::View* pView, CGSize size) override;

    BrainEngine* getBrainEngine();
    std::string getDefaultModelFilePath();

private:
    MTL::Device* _pDevice;
    BrainEngine* _pBrainEngine;
};

#endif
