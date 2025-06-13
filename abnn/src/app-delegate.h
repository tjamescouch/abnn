//
//  app-delegate.h
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//

#ifndef APP_DELEGATE_H
#define APP_DELEGATE_H

#pragma region Declarations {

#include "common.h"

#import "view-delegate.h"


class AppDelegate : public NS::ApplicationDelegate
{
    public:
        ~AppDelegate();

        NS::Menu* createMenuBar();

        virtual void applicationWillFinishLaunching( NS::Notification* pNotification ) override;
        virtual void applicationDidFinishLaunching( NS::Notification* pNotification ) override;
        virtual bool applicationShouldTerminateAfterLastWindowClosed( NS::Application* pSender ) override;
    
    NeuralEngine* getNeuralEngine() {
        return _pViewDelegate->getNeuralEngine();
    }

    private:
        NS::Window* _pWindow;
        MTK::View* _pMtkView;
        MTL::Device* _pDevice;
        ViewDelegate* _pViewDelegate = nullptr;
};

#pragma endregion Declarations }
#endif
