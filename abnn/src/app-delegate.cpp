
//
//  app-delegate.cpp
//  MetalNN
//
//  Created by James Couch on 2024-12-07.
//

#include "app-kit-bridge.h"
#import "app-delegate.h"
#include <mach-o/dyld.h>
#include <filesystem>
#include <fstream>

#pragma mark - AppDelegate
#pragma region AppDelegate {


AppDelegate::~AppDelegate()
{
    _pMtkView->release();
    _pWindow->release();
    _pDevice->release();
    delete _pViewDelegate;
}

NS::Menu* AppDelegate::createMenuBar()
{
    using NS::StringEncoding::UTF8StringEncoding;
    
    NS::Menu* pMainMenu = NS::Menu::alloc()->init();
    NS::MenuItem* pAppMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pAppMenu = NS::Menu::alloc()->init( NS::String::string( "Appname", UTF8StringEncoding ) );
    
    NS::String* appName = NS::RunningApplication::currentApplication()->localizedName();
    NS::String* quitItemName = NS::String::string( "Quit ", UTF8StringEncoding )->stringByAppendingString( appName );
    SEL quitCb = NS::MenuItem::registerActionCallback( "appQuit", [](void*,SEL,const NS::Object* pSender){
        auto pApp = NS::Application::sharedApplication();
        pApp->terminate( pSender );
    } );
    
    NS::MenuItem* pAppQuitItem = pAppMenu->addItem( quitItemName, quitCb, NS::String::string( "q", UTF8StringEncoding ) );
    pAppQuitItem->setKeyEquivalentModifierMask( NS::EventModifierFlagCommand );
    pAppMenuItem->setSubmenu( pAppMenu );
    
    NS::MenuItem* pWindowMenuItem = NS::MenuItem::alloc()->init();
    NS::Menu* pWindowMenu = NS::Menu::alloc()->init( NS::String::string( "Window", UTF8StringEncoding ) );
    
    SEL closeWindowCb = NS::MenuItem::registerActionCallback( "windowClose", [](void*, SEL, const NS::Object*){
        auto pApp = NS::Application::sharedApplication();
        pApp->windows()->object< NS::Window >(0)->close();
    } );
    NS::MenuItem* pCloseWindowItem = pWindowMenu->addItem( NS::String::string( "Close Window", UTF8StringEncoding ), closeWindowCb, NS::String::string( "w", UTF8StringEncoding ) );
    pCloseWindowItem->setKeyEquivalentModifierMask( NS::EventModifierFlagCommand );
    
    pWindowMenuItem->setSubmenu( pWindowMenu );
    
    pMainMenu->addItem( pAppMenuItem );
    pMainMenu->addItem( pWindowMenuItem );
    
    pAppMenuItem->release();
    pWindowMenuItem->release();
    pAppMenu->release();
    pWindowMenu->release();
    
    return pMainMenu->autorelease();
}

void AppDelegate::applicationWillFinishLaunching( NS::Notification* pNotification )
{
    NS::Menu* pMenu = createMenuBar();
    NS::Application* pApp = reinterpret_cast< NS::Application* >( pNotification->object() );
    pApp->setMainMenu( pMenu );
    pApp->setActivationPolicy( NS::ActivationPolicy::ActivationPolicyRegular );
}

void AppDelegate::applicationDidFinishLaunching( NS::Notification* pNotification )
{
    namespace fs = std::filesystem;
    CGRect frame = (CGRect){ {10, 10}, {640, 640} };

    
    _pWindow = NS::Window::alloc()->init(frame,
                                         NS::WindowStyleMaskClosable | NS::WindowStyleMaskTitled | NS::WindowStyleMaskResizable,
                                         NS::BackingStoreBuffered,
                                         false);
    
    _pDevice = MTL::CreateSystemDefaultDevice();
    
    
    
    
    _pMtkView = MTK::View::alloc()->init( frame, _pDevice );
    _pMtkView->setColorPixelFormat( MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB );
    _pMtkView->setClearColor( MTL::ClearColor::Make( 0.0, 0.0, 0.0, 1.0 ) );
    
    _pViewDelegate = new ViewDelegate( _pDevice );
    _pMtkView->setDelegate( _pViewDelegate );
    
    _pWindow->setContentView( _pMtkView );
    _pWindow->setTitle( NS::String::string( "Metal Compute", NS::StringEncoding::UTF8StringEncoding ) );
    
    _pWindow->makeKeyAndOrderFront( nullptr );
    
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        throw std::runtime_error("❌ Executable path buffer too small.");
    }
    
    fs::path executablePath = fs::canonical(path);
    fs::path resourcePath = executablePath.parent_path().parent_path() / "Resources";
    
    setMenuActionHandlers(
        [this] { this->getBrainEngine()->run(1000); },
        [this, resourcePath] { this->getBrainEngine()->saveModel(resourcePath / "model.bnn"); },
        [this, resourcePath] { this->getBrainEngine()->loadModel(resourcePath / "model.bnn"); }
    );

    setupMenus();
    
    setupTextField((void*)_pWindow);
    
    NS::Application* pApp = reinterpret_cast< NS::Application* >( pNotification->object() );
    pApp->activateIgnoringOtherApps( true );
}

bool AppDelegate::applicationShouldTerminateAfterLastWindowClosed( NS::Application* pSender )
{
    return true;
}


#pragma endregion AppDelegate }

